//! A tiny scripted MCP server over stdio, for tests. Built only with the
//! `test-server` feature.
//!
//! Flags:
//! - `--protocol <version>`: answer `initialize` with this version.
//! - `--hang-init`: never answer `initialize`.
//!
//! Tools (two `tools/list` pages): `echo`, `write.file`, `no_annotations`,
//! then `fail`, `crash`, `sleep`, `rich`, `ping_client`, `env_probe` and one
//! with an 80-character name.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

fn send(out: &mut impl Write, message: &Value) {
    let _ = writeln!(out, "{message}");
    let _ = out.flush();
}

fn text(t: &str) -> Value {
    json!({"content": [{"type": "text", "text": t}]})
}

fn page_one() -> Value {
    json!([
        {
            "name": "echo",
            "description": "Echo the text back",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            "annotations": {"readOnlyHint": true, "idempotentHint": true}
        },
        {
            "name": "write.file",
            "description": "Pretend to write a file",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "annotations": {"destructiveHint": false, "idempotentHint": true}
        },
        {
            "name": "no_annotations",
            "description": "Has no annotations",
            "inputSchema": {"type": "object"}
        }
    ])
}

fn page_two() -> Value {
    json!([
        {"name": "fail", "description": "Always reports isError", "inputSchema": {"type": "object"},
         "annotations": {"readOnlyHint": true}},
        {"name": "crash", "description": "Exits the server", "inputSchema": {"type": "object"}},
        {"name": "sleep", "description": "Never answers", "inputSchema": {"type": "object"},
         "annotations": {"readOnlyHint": true}},
        {"name": "rich", "description": "Mixed content", "inputSchema": {"type": "object"},
         "annotations": {"readOnlyHint": true}},
        {"name": "ping_client", "description": "Pings the client first", "inputSchema": {"type": "object"},
         "annotations": {"readOnlyHint": true}},
        {"name": "env_probe", "description": "Reports its environment", "inputSchema": {"type": "object"},
         "annotations": {"readOnlyHint": true}},
        {"name": "a".repeat(80), "inputSchema": {"type": "object"}}
    ])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let protocol = args
        .iter()
        .position(|a| a == "--protocol")
        .and_then(|i| args.get(i + 1).cloned());
    let hang_init = args.iter().any(|a| a == "--hang-init");

    eprintln!("fake-mcp: started");
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut out = std::io::stdout();
    let mut initialized = false;

    while let Some(Ok(line)) = lines.next() {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let method = msg["method"].as_str().unwrap_or_default().to_string();
        let id = msg.get("id").cloned();
        let Some(id) = id else {
            if method == "notifications/initialized" {
                initialized = true;
            }
            continue;
        };
        let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
        let error = |code: i64, message: &str| json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}});

        match method.as_str() {
            "initialize" => {
                if hang_init {
                    continue;
                }
                let requested = msg["params"]["protocolVersion"]
                    .as_str()
                    .unwrap_or_default();
                let version = protocol.clone().unwrap_or_else(|| requested.to_string());
                send(
                    &mut out,
                    &reply(json!({
                        "protocolVersion": version,
                        "capabilities": {"tools": {"listChanged": false}},
                        "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
                        "instructions": "synthetic test server",
                    })),
                );
            }
            "tools/list" => {
                if !initialized {
                    send(&mut out, &error(-32002, "not initialized"));
                    continue;
                }
                let page = match msg["params"]["cursor"].as_str() {
                    None => json!({"tools": page_one(), "nextCursor": "page-2"}),
                    Some("page-2") => json!({"tools": page_two()}),
                    Some(_) => {
                        send(&mut out, &error(-32602, "bad cursor"));
                        continue;
                    }
                };
                send(&mut out, &reply(page));
            }
            "tools/call" => {
                let name = msg["params"]["name"].as_str().unwrap_or_default();
                let arguments = &msg["params"]["arguments"];
                match name {
                    "echo" => send(
                        &mut out,
                        &reply(text(arguments["text"].as_str().unwrap_or_default())),
                    ),
                    "write.file" | "no_annotations" => send(&mut out, &reply(text("ok"))),
                    "fail" => send(
                        &mut out,
                        &reply(json!({
                            "content": [{"type": "text", "text": "disk is full"}],
                            "isError": true
                        })),
                    ),
                    "crash" => {
                        eprintln!("fake-mcp: crashing on purpose");
                        std::process::exit(3);
                    }
                    "sleep" => {}
                    "rich" => send(
                        &mut out,
                        &reply(json!({
                            "content": [
                                {"type": "text", "text": "summary"},
                                {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo="}
                            ],
                            "structuredContent": {"rows": 3}
                        })),
                    ),
                    "ping_client" => {
                        // Noise the client must tolerate, then a request.
                        let _ = writeln!(out, "this line is not json");
                        send(
                            &mut out,
                            &json!({"jsonrpc": "2.0", "method": "notifications/message",
                                    "params": {"level": "info", "data": "hello"}}),
                        );
                        send(
                            &mut out,
                            &json!({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"}),
                        );
                        let mut ponged = false;
                        for line in lines.by_ref() {
                            let Ok(line) = line else { break };
                            let Ok(answer) = serde_json::from_str::<Value>(&line) else {
                                continue;
                            };
                            if answer["id"] == "srv-1" && answer.get("result").is_some() {
                                ponged = true;
                                break;
                            }
                        }
                        let t = if ponged { "pong received" } else { "no pong" };
                        send(&mut out, &reply(text(t)));
                    }
                    "env_probe" => send(
                        &mut out,
                        &reply(json!({
                            "content": [],
                            "structuredContent": {
                                "token": std::env::var("FAKE_MCP_TOKEN").ok(),
                                "has_cargo_manifest_dir": std::env::var_os("CARGO_MANIFEST_DIR").is_some(),
                            }
                        })),
                    ),
                    _ => send(&mut out, &error(-32602, &format!("unknown tool: {name}"))),
                }
            }
            "ping" => send(&mut out, &reply(json!({}))),
            _ => send(&mut out, &error(-32601, "method not found")),
        }
    }
}
