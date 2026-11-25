//! JSON-RPC 2.0 lines for the agents spoken to over stdio (ACP, Codex's app
//! server): one message per line.

use serde_json::{Value, json};

fn line(message: Value) -> String {
    let mut line = message.to_string();
    line.push('\n');
    line
}

/// A request.
pub(crate) fn request_line(id: u64, method: &str, params: Value) -> String {
    line(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
}

/// A reply to the agent's request `id`.
pub(crate) fn reply_line(id: &Value, result: Value) -> String {
    line(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// A refusal of the agent's request `id` (method not served).
pub(crate) fn error_line(id: &Value, message: &str) -> String {
    line(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": -32601, "message": message},
    }))
}
