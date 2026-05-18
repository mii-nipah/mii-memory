use std::io::{BufRead, Write};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::model::{ExpirationCondition, MemoryMode};
use crate::store::{MemoryStore, SearchOptions, SetMemory, infer_mode_ref};

pub fn serve(mut store: MemoryStore, input: impl BufRead, mut output: impl Write) -> Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Value = serde_json::from_str(&line).context("failed to parse JSON request")?;
        if request.get("jsonrpc").is_some() {
            if let Some(response) = handle_json_rpc(&mut store, request)? {
                writeln!(output, "{}", response)?;
                output.flush()?;
            }
        } else {
            let response = handle_direct_command(&mut store, request)?;
            writeln!(output, "{}", response)?;
            output.flush()?;
        }
    }

    Ok(())
}

fn handle_direct_command(store: &mut MemoryStore, request: Value) -> Result<Value> {
    let command = request
        .get("command")
        .or_else(|| request.get("method"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("MCP request requires command or method"))?;

    let arguments = request.get("arguments").cloned().unwrap_or(request);
    invoke_command(store, &command, arguments)
}

fn handle_json_rpc(store: &mut MemoryStore, request: Value) -> Result<Option<Value>> {
    let id = request.get("id").cloned();
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("JSON-RPC request requires method"))?;

    if method.starts_with("notifications/") {
        return Ok(None);
    }

    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "mii-memory",
                "version": env!("CARGO_PKG_VERSION"),
            },
        }),
        "tools/list" => json!({ "tools": tool_definitions() }),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("tools/call requires params.name"))?;
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let command_result = invoke_command(store, name, arguments)?;
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&command_result)?,
                }],
                "isError": false,
            })
        }
        other => invoke_command(
            store,
            other,
            request.get("params").cloned().unwrap_or_else(|| json!({})),
        )?,
    };

    Ok(Some(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })))
}

fn invoke_command(store: &mut MemoryStore, command: &str, arguments: Value) -> Result<Value> {
    match command {
        "memory_set" | "set" => {
            let input: MemorySetPayload = serde_json::from_value(arguments)?;
            let mode_ref = infer_mode_ref(input.mode, None)?;
            let id = store.set(SetMemory {
                content: input.content,
                mode: input.mode,
                mode_ref,
                tags: input.tags,
                expiration_condition: input.expiration_condition,
                expiration_value: input.expiration_value,
                metadata: input.metadata,
            })?;
            Ok(json!({ "id": id }))
        }
        "memory_get" | "get" => {
            let input: MemoryGetPayload = serde_json::from_value(arguments)?;
            let results = store.get(SearchOptions {
                query: input.query,
                positive_tags: input.positive_tags.unwrap_or_default(),
                negative_tags: input.negative_tags,
                limit: input.limit.unwrap_or(10),
                offset: input.offset.unwrap_or(0),
                mode: input.mode,
                mode_ref: None,
            })?;
            Ok(json!({ "memories": results }))
        }
        "list_tags" | "list-tags" => {
            let input: ListTagsPayload = serde_json::from_value(arguments)?;
            Ok(json!({ "tags": store.list_tags(input.filter.as_deref())? }))
        }
        "alert_set" | "alert-set" => {
            let input: AlertSetPayload = serde_json::from_value(arguments)?;
            let id = store.set_alert(infer_alert_session_ref(input.session_ref)?, input.content)?;
            Ok(json!({ "id": id }))
        }
        "alerts_get" | "alerts-get" | "alerts" => {
            let input: AlertsGetPayload = serde_json::from_value(arguments)?;
            let alerts = store.get_alerts(infer_alert_session_ref(input.session_ref)?)?;
            Ok(json!({ "alerts": alerts }))
        }
        other => Err(anyhow!("unsupported MCP command: {other}")),
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "memory_set",
            "description": "Store a memory with one or more tags.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string" },
                    "mode": { "type": "string", "enum": ["global", "workspace", "session"], "default": "global" },
                    "tags": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
                    "expiration_condition": { "type": "string", "enum": ["time", "usage", "file_exist", "file_pristine", "period"] },
                    "expiration_value": { "type": "string" },
                    "metadata": { "type": "string" }
                },
                "required": ["content", "tags"]
            }
        },
        {
            "name": "memory_get",
            "description": "Retrieve relevant memories by query, positive tags, and negative tags.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "tag": { "type": "array", "items": { "type": "string" } },
                    "p_tag": { "type": "array", "items": { "type": "string" } },
                    "n_tag": { "type": "array", "items": { "type": "string" } },
                    "limit": { "type": "integer", "minimum": 1 },
                    "offset": { "type": "integer", "minimum": 0 },
                    "mode": { "type": "string", "enum": ["global", "workspace", "session"] }
                },
                "required": ["query"]
            }
        },
        {
            "name": "list_tags",
            "description": "List available memory tags, optionally filtered by text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": { "type": "string" }
                }
            }
        },
        {
            "name": "alert_set",
            "description": "Store a one-shot alert for the current agent session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"]
            }
        },
        {
            "name": "alerts_get",
            "description": "Return and clear one-shot alerts for the current agent session.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ])
}

fn infer_alert_session_ref(explicit: Option<String>) -> Result<String> {
    if let Some(explicit) = explicit
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(explicit);
    }

    Ok(std::env::var("MII_MEMORY_SESSION")
        .or_else(|_| std::env::var("MCP_SESSION_ID"))
        .unwrap_or_else(|_| "default".to_string()))
}

#[derive(Debug, Deserialize)]
struct MemorySetPayload {
    content: String,
    #[serde(default)]
    mode: MemoryMode,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    expiration_condition: Option<ExpirationCondition>,
    #[serde(default)]
    expiration_value: Option<String>,
    #[serde(default)]
    metadata: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MemoryGetPayload {
    query: String,
    #[serde(default, alias = "tag", alias = "p_tag", alias = "p-tag")]
    positive_tags: Option<Vec<String>>,
    #[serde(default, alias = "n_tag", alias = "n-tag")]
    negative_tags: Vec<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    mode: Option<MemoryMode>,
}

#[derive(Debug, Deserialize)]
struct ListTagsPayload {
    #[serde(default)]
    filter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlertSetPayload {
    content: String,
    #[serde(default)]
    session_ref: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlertsGetPayload {
    #[serde(default)]
    session_ref: Option<String>,
}
