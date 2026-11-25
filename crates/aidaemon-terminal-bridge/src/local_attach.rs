//! The local attach listener the attach, share and start CLIs connect to.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LocalAttachEndpoint {
    pub(crate) version: u8,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) secret: String,
    pub(crate) pid: u32,
    pub(crate) updated_at_unix: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct LocalAttachClient {
    pub(crate) session_id: String,
    pub(crate) tx: tokio::sync::mpsc::UnboundedSender<Value>,
}

#[derive(Debug)]
pub(crate) struct LocalAttachAccepted {
    pub(crate) session_id: String,
    pub(crate) resume_code: Option<String>,
    pub(crate) resume_expires_at_unix: Option<i64>,
}

#[derive(Debug)]
pub(crate) struct LocalShareCreated {
    pub(crate) session_id: String,
    pub(crate) code: String,
    pub(crate) expires_at_unix: i64,
}

pub(crate) enum LocalAttachEvent {
    AttachRequest {
        client_id: u64,
        code: String,
        secret: String,
        tx: tokio::sync::mpsc::UnboundedSender<Value>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalAttachAccepted>>,
    },
    StartRequest {
        client_id: u64,
        secret: String,
        agent: String,
        cwd: Option<String>,
        agent_args: Vec<String>,
        tx: tokio::sync::mpsc::UnboundedSender<Value>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalAttachAccepted>>,
    },
    ShareRequest {
        secret: String,
        session_id: Option<String>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalShareCreated>>,
    },
    Input {
        client_id: u64,
        data_b64: String,
    },
    Resize {
        client_id: u64,
        cols: u16,
        rows: u16,
    },
    Redraw {
        client_id: u64,
    },
    Close {
        client_id: u64,
    },
}

pub(crate) fn terminal_bridge_state_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".aidaemon-terminal")
}

pub(crate) fn local_attach_endpoint_path() -> PathBuf {
    terminal_bridge_state_dir().join(LOCAL_ATTACH_ENDPOINT_FILENAME)
}

pub(crate) async fn persist_local_attach_endpoint(port: u16, secret: &str) -> anyhow::Result<()> {
    let dir = terminal_bridge_state_dir();
    tokio::fs::create_dir_all(&dir).await?;
    set_owner_only_permissions(&dir, 0o700)?;

    let endpoint = LocalAttachEndpoint {
        version: 1,
        host: "127.0.0.1".to_string(),
        port,
        secret: secret.to_string(),
        pid: std::process::id(),
        updated_at_unix: chrono::Utc::now().timestamp(),
    };
    let path = local_attach_endpoint_path();
    let raw = serde_json::to_string_pretty(&endpoint)?;
    tokio::fs::write(&path, raw).await?;
    set_owner_only_permissions(&path, 0o600)?;
    Ok(())
}

pub(crate) async fn write_json_line<W>(writer: &mut W, value: &Value) -> anyhow::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

pub(crate) async fn spawn_local_attach_listener(
    events_tx: tokio::sync::mpsc::UnboundedSender<LocalAttachEvent>,
    secret: String,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    persist_local_attach_endpoint(port, &secret).await?;
    info!(port, "Local terminal attach endpoint ready");

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((socket, _)) => {
                    let events_tx = events_tx.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_local_attach_socket(socket, events_tx).await {
                            debug!(error = %err, "Local attach socket closed with error");
                        }
                    });
                }
                Err(err) => {
                    warn!(error = %err, "Local attach listener accept failed");
                    break;
                }
            }
        }
    });
    Ok(())
}

/// Read one line, but never more than one byte past the largest frame: a
/// client that sends no newline cannot make the bridge buffer without end.
/// A line longer than [`LOCAL_ATTACH_MAX_FRAME_BYTES`] comes back over that
/// length, for the caller to refuse.
async fn read_bounded_line<R>(reader: &mut R, line: &mut String) -> std::io::Result<usize>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;
    let mut limited = reader.take(LOCAL_ATTACH_MAX_FRAME_BYTES as u64 + 1);
    limited.read_line(line).await
}

/// Whether `given` is the attach secret, compared in constant time over
/// equal lengths so the comparison leaks nothing of how much matched.
pub(crate) fn same_secret(given: &str, expected: &str) -> bool {
    given.len() == expected.len()
        && given
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

pub(crate) async fn handle_local_attach_socket(
    socket: TcpStream,
    events_tx: tokio::sync::mpsc::UnboundedSender<LocalAttachEvent>,
) -> anyhow::Result<()> {
    let client_id = NEXT_LOCAL_CLIENT_ID.fetch_add(1, Ordering::Relaxed);
    let (read_half, mut write_half) = socket.into_split();
    let mut reader = BufReader::new(read_half);

    let mut first_line = String::new();
    let first_read = read_bounded_line(&mut reader, &mut first_line).await?;
    if first_read == 0 || first_line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
        anyhow::bail!("missing or oversized attach frame");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid attach frame: {}", e))?;
    let frame_type = first_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match frame_type {
        "attach" | "start" => {
            let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();

            if frame_type == "attach" {
                let code = first_value
                    .get("code")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                let secret = first_value
                    .get("secret")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                if code.is_empty() || secret.is_empty() {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"attach requires code and secret.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                events_tx
                    .send(LocalAttachEvent::AttachRequest {
                        client_id,
                        code,
                        secret,
                        tx: outbound_tx.clone(),
                        response_tx,
                    })
                    .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            } else {
                let secret = first_value
                    .get("secret")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                let agent = normalize_agent(first_value.get("agent").and_then(Value::as_str))
                    .unwrap_or_else(|| "codex".to_string());
                let agent_args = requested_agent_args_from_client_hello(&first_value);
                let cwd = first_value
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(|v| v.to_string());
                if secret.is_empty() {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"start requires secret.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                events_tx
                    .send(LocalAttachEvent::StartRequest {
                        client_id,
                        secret,
                        agent,
                        cwd,
                        agent_args,
                        tx: outbound_tx.clone(),
                        response_tx,
                    })
                    .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            }

            let accepted = match response_rx.await {
                Ok(Ok(ok)) => ok,
                Ok(Err(err)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message": err.to_string(),
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                Err(_) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"local attach service did not respond.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
            };

            write_json_line(
                &mut write_half,
                &json!({
                    "type":"attached",
                    "session_id": accepted.session_id,
                    "resume_code": accepted.resume_code,
                    "resume_expires_at_unix": accepted.resume_expires_at_unix,
                }),
            )
            .await?;

            let writer_task = tokio::spawn(async move {
                while let Some(payload) = outbound_rx.recv().await {
                    if write_json_line(&mut write_half, &payload).await.is_err() {
                        break;
                    }
                }
            });

            let mut line = String::new();
            loop {
                line.clear();
                let n = read_bounded_line(&mut reader, &mut line).await?;
                if n == 0 {
                    break;
                }
                if line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
                    break;
                }
                let value: Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match value.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "stdin" => {
                        let data_b64 = value
                            .get("data")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if data_b64.is_empty() {
                            continue;
                        }
                        let _ = events_tx.send(LocalAttachEvent::Input {
                            client_id,
                            data_b64,
                        });
                    }
                    "resize" => {
                        let cols = value
                            .get("cols")
                            .or_else(|| value.get("width"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(PTY_DEFAULT_COLS as u64)
                            .clamp(20, 400) as u16;
                        let rows = value
                            .get("rows")
                            .or_else(|| value.get("height"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(PTY_DEFAULT_ROWS as u64)
                            .clamp(5, 200) as u16;
                        let _ = events_tx.send(LocalAttachEvent::Resize {
                            client_id,
                            cols,
                            rows,
                        });
                    }
                    "redraw" => {
                        let _ = events_tx.send(LocalAttachEvent::Redraw { client_id });
                    }
                    "detach" => break,
                    _ => {}
                }
            }

            let _ = events_tx.send(LocalAttachEvent::Close { client_id });
            writer_task.abort();
            Ok(())
        }
        "share" => {
            let secret = first_value
                .get("secret")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            let session_id = normalize_relay_session_id(
                first_value
                    .get("session_id")
                    .and_then(Value::as_str)
                    .or_else(|| first_value.get("sessionId").and_then(Value::as_str)),
            );
            if secret.is_empty() {
                write_json_line(
                    &mut write_half,
                    &json!({
                        "type":"error",
                        "message":"share requires secret.",
                    }),
                )
                .await?;
                return Ok(());
            }
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            events_tx
                .send(LocalAttachEvent::ShareRequest {
                    secret,
                    session_id,
                    response_tx,
                })
                .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            match response_rx.await {
                Ok(Ok(shared)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"shared",
                            "session_id": shared.session_id,
                            "code": shared.code,
                            "expires_at_unix": shared.expires_at_unix,
                        }),
                    )
                    .await?;
                    Ok(())
                }
                Ok(Err(err)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message": err.to_string(),
                        }),
                    )
                    .await?;
                    Ok(())
                }
                Err(_) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"local share service did not respond.",
                        }),
                    )
                    .await?;
                    Ok(())
                }
            }
        }
        _ => {
            write_json_line(
                &mut write_half,
                &json!({
                    "type":"error",
                    "message":"First frame must be type=attach, type=start, or type=share.",
                }),
            )
            .await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod bounded_tests {
    use super::*;

    #[tokio::test]
    async fn a_line_without_end_is_read_only_to_the_frame_limit() {
        let endless = vec![b'a'; LOCAL_ATTACH_MAX_FRAME_BYTES * 4];
        let mut reader = tokio::io::BufReader::new(&endless[..]);
        let mut line = String::new();
        let n = read_bounded_line(&mut reader, &mut line).await.unwrap();
        assert_eq!(n, LOCAL_ATTACH_MAX_FRAME_BYTES + 1);
        assert!(line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES);
        let mut short = tokio::io::BufReader::new(&b"{\"type\":\"attach\"}\nnext"[..]);
        let mut line = String::new();
        read_bounded_line(&mut short, &mut line).await.unwrap();
        assert_eq!(line, "{\"type\":\"attach\"}\n");
    }

    #[test]
    fn the_secret_matches_only_itself() {
        assert!(same_secret("s3cret-synthetic", "s3cret-synthetic"));
        assert!(!same_secret("s3cret-synthetiX", "s3cret-synthetic"));
        assert!(!same_secret("s3cret", "s3cret-synthetic"));
        assert!(!same_secret("", "s3cret-synthetic"));
    }
}
