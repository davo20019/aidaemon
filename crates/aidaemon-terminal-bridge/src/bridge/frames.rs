//! Encrypted relay frames, session id remaps and session teardown.

use crate::*;

impl TerminalBridge {
    pub(crate) async fn handle_encrypted_frame<S>(
        &mut self,
        frame: &Value,
        relay_session_id: &str,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let mut remove_reason: Option<&'static str> = None;
        let mut status_messages: Vec<String> = Vec::new();
        let mut immediate_payloads: Vec<Value> = Vec::new();
        let max_upload_bytes = self.cfg.max_upload_bytes;
        let inbox_dir = self.cfg.inbox_dir.clone();
        let review_profiles = self.cfg.review_profiles.clone();
        let shell_events_tx = self.shell_events_tx.clone();
        {
            let Some(active) = self.sessions.get_mut(relay_session_id) else {
                return Ok(());
            };
            let crypto = &mut active.crypto;

            let counter = frame.get("c").and_then(|v| v.as_u64()).unwrap_or(0);
            if counter <= crypto.recv_counter {
                return Ok(());
            }
            let iv = b64_decode(frame.get("iv").and_then(|v| v.as_str()).unwrap_or(""))?;
            let ct = b64_decode(frame.get("ct").and_then(|v| v.as_str()).unwrap_or(""))?;
            if iv.len() != 12 || ct.is_empty() {
                return Ok(());
            }
            if ct.len() > MAX_ENCRYPTED_PAYLOAD_BYTES.saturating_mul(2) {
                return Ok(());
            }

            let aad = format!("{}:{}", crypto.session_id, counter);
            let plaintext = match crypto.cipher.decrypt(
                #[allow(deprecated)]
                Nonce::from_slice(&iv),
                Payload {
                    msg: &ct,
                    aad: aad.as_bytes(),
                },
            ) {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };
            if plaintext.len() > MAX_ENCRYPTED_PAYLOAD_BYTES {
                return Ok(());
            }
            crypto.recv_counter = counter;

            let payload: Value = match serde_json::from_slice(&plaintext) {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };
            let kind = payload.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "stdin" => {
                    let data = payload.get("data").and_then(|v| v.as_str()).unwrap_or("");
                    let normalized = normalize_stdin_for_pty(data);
                    let _ = active.shell.write_stdin(&normalized).await;
                }
                "signal" => {
                    let signal = payload
                        .get("signal")
                        .and_then(|v| v.as_str())
                        .unwrap_or("INT")
                        .to_ascii_uppercase();
                    match signal.as_str() {
                        "INT" | "SIGINT" => {
                            let _ = active.shell.write_stdin("\u{3}").await;
                        }
                        "TSTP" | "SIGTSTP" => {
                            let _ = active.shell.write_stdin("\u{1a}").await;
                        }
                        _ => {
                            remove_reason = Some("signal stop");
                        }
                    }
                }
                "control" if payload.get("action").and_then(|v| v.as_str()) == Some("stop") => {
                    remove_reason = Some("stop requested by client");
                }
                "resize" => {
                    let cols = payload
                        .get("cols")
                        .or_else(|| payload.get("width"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(PTY_DEFAULT_COLS as u64)
                        .clamp(20, 400) as u16;
                    let rows = payload
                        .get("rows")
                        .or_else(|| payload.get("height"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(PTY_DEFAULT_ROWS as u64)
                        .clamp(5, 200) as u16;
                    let _ = active.shell.resize(cols, rows);
                }
                "upload_start" => {
                    match Self::begin_image_upload(active, &payload, max_upload_bytes) {
                        Ok(msg) => status_messages.push(msg),
                        Err(err) => {
                            status_messages.push(format!("Image upload error: {}", err));
                        }
                    }
                }
                "upload_chunk" => {
                    match Self::append_image_upload_chunk(active, &payload, max_upload_bytes) {
                        Ok(Some(msg)) => status_messages.push(msg),
                        Ok(None) => {}
                        Err(err) => {
                            status_messages.push(format!("Image upload error: {}", err));
                        }
                    }
                }
                "upload_commit" => match Self::commit_image_upload(active, &payload, &inbox_dir) {
                    Ok(result) => {
                        status_messages.push(result.status_message);
                        if let Some(prompt) = result.prompt_for_agent {
                            match active.shell.write_stdin(&prompt).await {
                                Ok(_) => status_messages
                                    .push("Image context sent to active agent.".to_string()),
                                Err(err) => status_messages.push(format!(
                                    "Image saved, but failed to send it to the active agent: {}",
                                    err
                                )),
                            }
                        } else if let Some(agent_status_message) = result.agent_status_message {
                            status_messages.push(agent_status_message);
                        } else {
                            status_messages.push(
                                "Image saved. No active CLI agent detected, so it was not auto-sent."
                                    .to_string(),
                            );
                        }
                    }
                    Err(err) => {
                        status_messages.push(format!("Image upload error: {}", err));
                    }
                },
                "upload_cancel" => match Self::cancel_image_upload(active, &payload) {
                    Ok(msg) => status_messages.push(msg),
                    Err(err) => {
                        status_messages.push(format!("Image upload error: {}", err));
                    }
                },
                "review_start" => {
                    let request_id = normalize_review_request_id(
                        payload.get("request_id").and_then(|v| v.as_str()),
                    )
                    .unwrap_or_else(next_review_request_id);
                    if let Some(running) = active.review_job.as_ref() {
                        let payload = json!({
                            "kind":"review_error",
                            "request_id": request_id,
                            "code":"review_already_running",
                            "message": format!(
                                "A review is already running (request_id={}). Cancel it before starting another one.",
                                running.request_id
                            ),
                        });
                        active.last_review_progress = Some(payload.clone());
                        active.last_review_result = None;
                        immediate_payloads.push(payload);
                    } else {
                        let reviewer = normalize_agent(
                            payload
                                .get("reviewer")
                                .and_then(|v| v.as_str())
                                .or_else(|| payload.get("agent").and_then(|v| v.as_str())),
                        )
                        .unwrap_or_else(|| "codex".to_string());
                        if let Some(profile) = review_profiles.get(&reviewer).cloned() {
                            let scope = ReviewScope::from_raw(
                                payload.get("scope").and_then(|v| v.as_str()),
                            );
                            let diff_base = ReviewDiffBase::from_raw(
                                payload.get("diff_base").and_then(|v| v.as_str()),
                            );
                            let notes = payload
                                .get("notes")
                                .and_then(|v| v.as_str())
                                .map(str::trim)
                                .filter(|v| !v.is_empty())
                                .map(|v| truncate_with_note(v, REVIEW_MAX_SECTION_CHARS));
                            let cwd = active.cwd.clone();
                            let replay_tail = collect_replay_tail(active, REVIEW_MAX_SECTION_CHARS);
                            let session_id = relay_session_id.to_string();
                            let request_id_for_task = request_id.clone();
                            let reviewer_for_task = reviewer.clone();
                            let tx_for_task = shell_events_tx.clone();

                            let queued_payload = json!({
                                "kind":"review_progress",
                                "request_id": request_id,
                                "stage":"queued",
                                "message": format!("Starting {} review...", reviewer),
                                "reviewer": reviewer,
                                "scope": scope.as_str(),
                                "diff_base": diff_base.as_str(),
                            });
                            active.review_stream_replay.clear();
                            active.review_stream_replay_bytes = 0;
                            active.next_review_stream_seq = 1;
                            active.last_review_progress = Some(queued_payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(queued_payload);

                            let handle = tokio::spawn(async move {
                                run_review_job(
                                    tx_for_task,
                                    session_id,
                                    request_id_for_task,
                                    reviewer_for_task,
                                    profile,
                                    cwd,
                                    scope,
                                    diff_base,
                                    notes,
                                    replay_tail,
                                )
                                .await;
                            });
                            active.review_job = Some(ReviewJob { request_id, handle });
                        } else {
                            let payload = json!({
                                "kind":"review_error",
                                "request_id": request_id,
                                "code":"reviewer_not_configured",
                                "message": format!(
                                    "Reviewer `{}` is not configured. Supported: {}.",
                                    reviewer,
                                    SUPPORTED_TERMINAL_AGENTS.join(", ")
                                ),
                            });
                            active.last_review_progress = Some(payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(payload);
                        }
                    }
                }
                "review_cancel" => {
                    let requested = normalize_review_request_id(
                        payload.get("request_id").and_then(|v| v.as_str()),
                    );
                    if let Some(job) = active.review_job.take() {
                        if let Some(requested_id) = requested {
                            if requested_id != job.request_id {
                                active.review_job = Some(job);
                                status_messages.push(format!(
                                    "Running review has id `{}`; nothing cancelled.",
                                    requested_id
                                ));
                            } else {
                                let request_id = job.request_id.clone();
                                job.handle.abort();
                                let payload = json!({
                                    "kind":"review_error",
                                    "request_id": request_id,
                                    "code":"review_cancelled",
                                    "message":"Review cancelled by user.",
                                });
                                active.last_review_progress = Some(payload.clone());
                                active.last_review_result = None;
                                immediate_payloads.push(payload);
                            }
                        } else {
                            let request_id = job.request_id.clone();
                            job.handle.abort();
                            let payload = json!({
                                "kind":"review_error",
                                "request_id": request_id,
                                "code":"review_cancelled",
                                "message":"Review cancelled by user.",
                            });
                            active.last_review_progress = Some(payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(payload);
                        }
                    } else {
                        let payload = json!({
                            "kind":"review_error",
                            "request_id": requested.unwrap_or_else(next_review_request_id),
                            "code":"review_not_running",
                            "message":"No running review to cancel.",
                        });
                        active.last_review_progress = Some(payload.clone());
                        active.last_review_result = None;
                        immediate_payloads.push(payload);
                    }
                }
                _ => {}
            }
        }

        for message in status_messages {
            let _ = self
                .send_encrypted_json_for_session(
                    relay_session_id,
                    ws_write,
                    json!({"kind":"status","message":message}),
                )
                .await;
        }
        for payload in immediate_payloads {
            let _ = self
                .send_encrypted_json_for_session(relay_session_id, ws_write, payload)
                .await;
        }

        if let Some(reason) = remove_reason {
            self.remove_session(relay_session_id, reason).await;
        }
        Ok(())
    }

    pub(crate) async fn remap_active_session_id(
        &mut self,
        from_session_id: &str,
        to_session_id: &str,
    ) -> anyhow::Result<()> {
        if from_session_id == to_session_id {
            return Ok(());
        }
        if !self.sessions.contains_key(from_session_id) {
            anyhow::bail!("source session does not exist");
        }
        if self.sessions.contains_key(to_session_id) {
            // Telegram may resume an older scope-bound session id; when an explicit
            // /agent resume mapping points to a different active native session,
            // treat the mapping as authoritative and replace the stale target.
            self.remove_session(to_session_id, "session replaced by /agent resume mapping")
                .await;
        }

        let mut active = self
            .sessions
            .remove(from_session_id)
            .ok_or_else(|| anyhow::anyhow!("source session disappeared"))?;
        active.crypto.session_id = to_session_id.to_string();
        self.sessions.insert(to_session_id.to_string(), active);

        for client in self.local_clients.values_mut() {
            if client.session_id == from_session_id {
                client.session_id = to_session_id.to_string();
            }
        }

        if let Some(owner_user_id) = self.owner_user_id()
            && let Err(err) = self
                .host
                .set_last_active_relay_session_id(owner_user_id, to_session_id)
                .await
        {
            warn!(
                error = %err,
                from_session_id = %from_session_id,
                to_session_id = %to_session_id,
                "Failed to persist remapped relay session id"
            );
        }

        Ok(())
    }

    pub(crate) async fn send_plain_json<S>(ws_write: &mut S, value: Value) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        ws_write
            .send(Message::text(serde_json::to_string(&value)?))
            .await?;
        Ok(())
    }

    pub(crate) async fn send_encrypted_json_for_session<S>(
        &mut self,
        session_id: &str,
        ws_write: &mut S,
        payload: Value,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let crypto = &mut active.crypto;
        let plaintext = serde_json::to_vec(&payload)?;
        if plaintext.len() > MAX_ENCRYPTED_PAYLOAD_BYTES {
            return Ok(());
        }

        let counter = crypto
            .send_counter
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session counter overflow"))?;
        crypto.send_counter = counter;
        let aad = format!("{}:{}", crypto.session_id, counter);
        let mut iv = [0u8; 12];
        UnwrapErr(SysRng).fill_bytes(&mut iv);
        let ct = crypto
            .cipher
            .encrypt(
                #[allow(deprecated)]
                Nonce::from_slice(&iv),
                Payload {
                    msg: &plaintext,
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("encrypt failed"))?;

        let envelope = json!({
            "type":"e2ee",
            "session_id": session_id,
            "c": counter,
            "iv": b64_encode(&iv),
            "ct": b64_encode(&ct),
        });
        Self::send_plain_json(ws_write, envelope).await
    }

    pub(crate) async fn remove_session(&mut self, session_id: &str, reason: &str) {
        self.broadcast_local_payload(
            session_id,
            json!({
                "type":"exit",
                "reason": reason,
            }),
        );
        self.local_clients
            .retain(|_, client| client.session_id != session_id);
        if let Some(mut active) = self.sessions.remove(session_id) {
            debug!(reason, session_id, "Stopping local shell process");
            if let Some(job) = active.review_job.take() {
                job.handle.abort();
            }
            active.shell.stop().await;
        }
    }
}
