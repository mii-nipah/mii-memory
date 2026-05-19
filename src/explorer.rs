use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::model::MemoryMode;
use crate::store::{BrowseOptions, MemoryStore, StoreSignature};

const INDEX_HTML: &str = include_str!("explorer/index.html");
const POLL_INTERVAL: Duration = Duration::from_millis(750);
const MAX_REQUEST_BYTES: usize = 16 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(15);

pub fn serve(database_path: PathBuf, host: &str, port: u16) -> Result<()> {
    let bind = format!("{host}:{port}");
    let listener =
        TcpListener::bind(&bind).with_context(|| format!("failed to bind explorer to {bind}"))?;
    let address = listener.local_addr()?;
    eprintln!("mii-memory explorer listening on http://{address}");
    serve_with_listener(listener, database_path)
}

pub fn serve_with_listener(listener: TcpListener, database_path: PathBuf) -> Result<()> {
    let database_path = Arc::new(database_path);
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("explorer accept failed: {error}");
                continue;
            }
        };

        let database_path = Arc::clone(&database_path);
        thread::spawn(move || {
            if let Err(error) = handle_connection(stream, &database_path) {
                eprintln!("explorer connection error: {error}");
            }
        });
    }

    Ok(())
}

#[derive(Debug)]
struct Request {
    method: String,
    path: String,
    query: HashMap<String, Vec<String>>,
}

fn handle_connection(mut stream: TcpStream, database_path: &Path) -> Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(READ_TIMEOUT))?;

    let request = match read_request(&mut stream) {
        Ok(request) => request,
        Err(error) => {
            write_response(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                error.to_string().as_bytes(),
            )?;
            return Ok(());
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => write_response(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        ),
        ("GET", "/api/memories") => serve_memories(&mut stream, database_path, &request.query),
        ("GET", "/api/tags") => serve_tags(&mut stream, database_path, &request.query),
        ("GET", "/api/events") => serve_events(stream, database_path),
        ("GET", _) => write_response(&mut stream, 404, "text/plain; charset=utf-8", b"not found"),
        _ => write_response(
            &mut stream,
            405,
            "text/plain; charset=utf-8",
            b"method not allowed",
        ),
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    let read = reader.read_line(&mut request_line)?;
    if read == 0 {
        bail!("empty request");
    }

    let mut parts = request_line.split_whitespace();
    let method = parts.next().context("missing HTTP method")?.to_string();
    let target = parts.next().context("missing HTTP target")?.to_string();

    let mut total = read;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        total += read;
        if total > MAX_REQUEST_BYTES {
            bail!("request headers too large");
        }
        if read == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }

    let (path, query) = parse_target(&target);
    Ok(Request {
        method,
        path,
        query,
    })
}

fn parse_target(target: &str) -> (String, HashMap<String, Vec<String>>) {
    let mut parts = target.splitn(2, '?');
    let path = parts.next().unwrap_or("/").to_string();
    let mut query: HashMap<String, Vec<String>> = HashMap::new();

    if let Some(raw_query) = parts.next() {
        for pair in raw_query.split('&').filter(|pair| !pair.is_empty()) {
            let mut kv = pair.splitn(2, '=');
            let key = decode_form(kv.next().unwrap_or(""));
            let value = decode_form(kv.next().unwrap_or(""));
            query.entry(key).or_default().push(value);
        }
    }

    (path, query)
}

fn decode_form(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(decoded) => output.push(decoded),
                    Err(_) => output.push(bytes[index]),
                }
                index += 3;
            }
            other => {
                output.push(other);
                index += 1;
            }
        }
    }

    String::from_utf8_lossy(&output).into_owned()
}

#[derive(Debug, Serialize)]
struct MemoriesResponse {
    memories: Vec<crate::store::MemoryEntry>,
    signature: StoreSignature,
}

#[derive(Debug, Serialize)]
struct TagsResponse {
    tags: Vec<crate::store::TagSummary>,
    signature: StoreSignature,
}

fn serve_memories(
    stream: &mut TcpStream,
    database_path: &Path,
    query: &HashMap<String, Vec<String>>,
) -> Result<()> {
    let store = MemoryStore::open(database_path)?;
    let text = query.get("text").and_then(|values| values.first().cloned());
    let mode = query
        .get("mode")
        .and_then(|values| values.first())
        .map(|value| MemoryMode::from_str(value))
        .transpose()
        .ok()
        .flatten();
    let tags = query.get("tag").cloned().unwrap_or_default();
    let limit = query
        .get("limit")
        .and_then(|values| values.first())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let offset = query
        .get("offset")
        .and_then(|values| values.first())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    let memories = store.browse(BrowseOptions {
        text,
        tags,
        mode,
        limit,
        offset,
    })?;
    let signature = store.signature()?;
    let response = MemoriesResponse {
        memories,
        signature,
    };
    let body = serde_json::to_vec(&response)?;
    write_response(stream, 200, "application/json", &body)
}

fn serve_tags(
    stream: &mut TcpStream,
    database_path: &Path,
    query: &HashMap<String, Vec<String>>,
) -> Result<()> {
    let store = MemoryStore::open(database_path)?;
    let filter = query
        .get("filter")
        .and_then(|values| values.first())
        .cloned();
    let tags = store.list_tags(filter.as_deref())?;
    let signature = store.signature()?;
    let response = TagsResponse { tags, signature };
    let body = serde_json::to_vec(&response)?;
    write_response(stream, 200, "application/json", &body)
}

fn serve_events(mut stream: TcpStream, database_path: &Path) -> Result<()> {
    let headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: keep-alive\r\n\
        X-Accel-Buffering: no\r\n\r\n";
    stream.write_all(headers.as_bytes())?;
    stream.write_all(b"event: ready\ndata: {}\n\n")?;
    stream.flush()?;

    let store = MemoryStore::open(database_path)?;
    let mut signature = store.signature().ok();

    loop {
        thread::sleep(POLL_INTERVAL);
        let current = match store.signature() {
            Ok(value) => Some(value),
            Err(_) => continue,
        };
        if current != signature {
            signature = current.clone();
            let payload = serde_json::to_string(&current)?;
            if stream
                .write_all(format!("event: update\ndata: {payload}\n\n").as_bytes())
                .is_err()
            {
                break;
            }
            if stream.flush().is_err() {
                break;
            }
        } else if stream.write_all(b": keep-alive\n\n").is_err() || stream.flush().is_err() {
            break;
        }
    }

    Ok(())
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "OK",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {len}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        len = body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

#[cfg(all(test, has_embedded_embeddings))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Instant;

    use crate::store::SetMemory;
    use tempfile::tempdir;

    fn read_http_response(stream: &mut TcpStream) -> (u16, String, Vec<u8>) {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 4096];
        let started = Instant::now();
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                Err(_) => break,
            }
            if started.elapsed() > Duration::from_secs(5) {
                break;
            }
        }
        let text = String::from_utf8_lossy(&buffer).into_owned();
        let mut split = text.splitn(2, "\r\n\r\n");
        let head = split.next().unwrap_or("").to_string();
        let body = split.next().unwrap_or("").as_bytes().to_vec();
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);
        (status, head, body)
    }

    fn spawn_explorer(database_path: PathBuf) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind explorer");
        let address = listener.local_addr().expect("local address");
        thread::spawn(move || {
            let _ = serve_with_listener(listener, database_path);
        });
        address
    }

    fn http_get(address: std::net::SocketAddr, path: &str) -> (u16, String, Vec<u8>) {
        let mut stream = TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set timeout");
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).expect("write");
        read_http_response(&mut stream)
    }

    #[test]
    fn explorer_serves_index_and_api() -> Result<()> {
        let directory = tempdir()?;
        let database_path = directory.path().join("explorer.db");
        {
            let mut store = MemoryStore::open(&database_path)?;
            store.set(SetMemory {
                content: "Explorer ready".to_string(),
                mode: MemoryMode::Global,
                mode_ref: None,
                tags: vec!["explorer".to_string()],
                expiration_condition: None,
                expiration_value: None,
                metadata: Some("{\"note\":\"hi\"}".to_string()),
            })?;
        }

        let address = spawn_explorer(database_path.clone());

        let (status, _, body) = http_get(address, "/");
        assert_eq!(status, 200);
        assert!(String::from_utf8_lossy(&body).contains("mii-memory explorer"));

        let (status, _, body) = http_get(address, "/api/memories");
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(value["memories"][0]["content"], "Explorer ready");
        assert_eq!(value["memories"][0]["tags"][0], "explorer");

        let (status, _, body) = http_get(address, "/api/tags");
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(value["tags"][0]["tag"], "explorer");

        let (status, _, body) = http_get(address, "/api/memories?tag=missing");
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(value["memories"].as_array().unwrap().len(), 0);

        let (status, _, _) = http_get(address, "/nope");
        assert_eq!(status, 404);

        Ok(())
    }

    #[test]
    fn explorer_events_emit_updates_when_memories_change() -> Result<()> {
        let directory = tempdir()?;
        let database_path = directory.path().join("events.db");
        {
            let _ = MemoryStore::open(&database_path)?;
        }

        let address = spawn_explorer(database_path.clone());

        let mut stream = TcpStream::connect(address).expect("connect events");
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.write_all(
            b"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n",
        )?;

        // Read until we see the ready event.
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];
        let deadline = Instant::now() + Duration::from_secs(5);
        while !String::from_utf8_lossy(&buffer).contains("event: ready") {
            if Instant::now() > deadline {
                panic!("did not receive ready event");
            }
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                Err(_) => break,
            }
        }

        // Mutate the DB from a different process-like connection.
        {
            let mut store = MemoryStore::open(&database_path)?;
            store.set(SetMemory {
                content: "live update".to_string(),
                mode: MemoryMode::Global,
                mode_ref: None,
                tags: vec!["live".to_string()],
                expiration_condition: None,
                expiration_value: None,
                metadata: None,
            })?;
        }

        let deadline = Instant::now() + Duration::from_secs(5);
        while !String::from_utf8_lossy(&buffer).contains("event: update") {
            if Instant::now() > deadline {
                panic!(
                    "did not receive update event; got: {}",
                    String::from_utf8_lossy(&buffer)
                );
            }
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                Err(_) => break,
            }
        }

        Ok(())
    }
}
