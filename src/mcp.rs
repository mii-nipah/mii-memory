use std::io::{BufRead, Write};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::model::{ExpirationCondition, MemoryMode};
use crate::store::{MemoryStore, SearchOptions, SetMemory, infer_mode_ref};

pub fn serve(mut store: MemoryStore, input: impl BufRead, mut output: impl Write) -> Result<()> {
    let context = ServerContext::new();

    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Value = serde_json::from_str(&line).context("failed to parse JSON request")?;
        if request.get("jsonrpc").is_some() {
            if let Some(response) = handle_json_rpc(&mut store, &context, request)? {
                writeln!(output, "{}", response)?;
                output.flush()?;
            }
        } else {
            let response = handle_direct_command(&mut store, &context, request)?;
            writeln!(output, "{}", response)?;
            output.flush()?;
        }
    }

    Ok(())
}

struct ServerContext {
    session_ref: String,
}

impl ServerContext {
    fn new() -> Self {
        Self {
            session_ref: Uuid::new_v4().to_string(),
        }
    }
}

fn handle_direct_command(
    store: &mut MemoryStore,
    context: &ServerContext,
    request: Value,
) -> Result<Value> {
    let command = request
        .get("command")
        .or_else(|| request.get("method"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("MCP request requires command or method"))?;

    let arguments = request.get("arguments").cloned().unwrap_or(request);
    invoke_command(store, context, &command, arguments)
}

fn handle_json_rpc(
    store: &mut MemoryStore,
    context: &ServerContext,
    request: Value,
) -> Result<Option<Value>> {
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
            let command_result = invoke_command(store, context, name, arguments)?;
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
            context,
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

fn invoke_command(
    store: &mut MemoryStore,
    context: &ServerContext,
    command: &str,
    arguments: Value,
) -> Result<Value> {
    match command {
        "memory_set" | "set" => {
            let input: MemorySetPayload = serde_json::from_value(arguments)?;
            let mode_ref = infer_mcp_mode_ref(context, input.mode)?;
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
            let mode_ref = input
                .mode
                .map(|mode| infer_mcp_mode_ref(context, mode))
                .transpose()?
                .flatten();
            let results = store.get(SearchOptions {
                query: input.query,
                positive_tags: input.positive_tags.unwrap_or_default(),
                negative_tags: input.negative_tags,
                limit: input.limit.unwrap_or(10),
                offset: input.offset.unwrap_or(0),
                mode: input.mode,
                mode_ref,
            })?;
            Ok(json!({ "memories": results }))
        }
        "list_tags" | "list-tags" => {
            let input: ListTagsPayload = serde_json::from_value(arguments)?;
            Ok(json!({ "tags": store.list_tags(input.filter.as_deref())? }))
        }
        "alert_set" | "alert-set" => {
            let input: AlertSetPayload = serde_json::from_value(arguments)?;
            let id = store.set_alert(context.session_ref.clone(), input.content)?;
            Ok(json!({ "id": id }))
        }
        "alerts_get" | "alerts-get" | "alerts" => {
            let _: AlertsGetPayload = serde_json::from_value(arguments)?;
            let alerts = store.get_alerts(context.session_ref.clone())?;
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

fn infer_mcp_mode_ref(context: &ServerContext, mode: MemoryMode) -> Result<Option<String>> {
    match mode {
        MemoryMode::Global => Ok(None),
        MemoryMode::Workspace => infer_mode_ref(mode, None),
        MemoryMode::Session => Ok(Some(context.session_ref.clone())),
    }
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
}

#[derive(Debug, Deserialize)]
struct AlertsGetPayload {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_alerts_use_process_session_and_expire_on_read() -> Result<()> {
        let store = MemoryStore::in_memory()?;
        let input = br#"
{"command":"alert_set","arguments":{"content":"remember this"}}
{"command":"alerts_get","arguments":{}}
{"command":"alerts_get","arguments":{}}
"#;
        let mut output = Vec::new();

        serve(store, &input[..], &mut output)?;

        let lines = String::from_utf8(output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["alerts"][0]["content"], "remember this");
        assert_eq!(lines[2]["alerts"].as_array().unwrap().len(), 0);
        Ok(())
    }

    #[test]
    fn mcp_session_memories_are_scoped_to_process_session() -> Result<()> {
        let context = ServerContext {
            session_ref: "test-session".to_string(),
        };

        assert_eq!(
            infer_mcp_mode_ref(&context, MemoryMode::Session)?,
            Some("test-session".to_string())
        );
        assert_eq!(infer_mcp_mode_ref(&context, MemoryMode::Global)?, None);
        Ok(())
    }
}
