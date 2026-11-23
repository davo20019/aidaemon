//! Relay messages: client hello, replay and agent bootstrap.

use crate::*;

impl TerminalBridge {
    pub(crate) async fn flush_outbound_queue<S>(
        &mut self,
        outbound_queue: &mut OutboundQueue,
        ws_write: &mut S,
        max_batch: usize,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        for _ in 0..max_batch {
            let Some(queued) = outbound_queue.pop_next() else {
                break;
            };
            self.send_encrypted_json_for_session(&queued.session_id, ws_write, queued.payload)
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn replay_stdout_since<S>(
        &mut self,
        session_id: &str,
        resume_from_seq: u64,
        resume_from_review_stream_seq: u64,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let (
            oldest_seq,
            frames,
            oldest_review_stream_seq,
            review_stream_frames,
            last_review_progress,
            last_review_result,
            running_review_request_id,
        ) = {
            let Some(active) = self.sessions.get(session_id) else {
                return Ok(());
            };
            let oldest = active.replay.front().map(|f| f.seq).unwrap_or(0);
            let frames = active
                .replay
                .iter()
                .filter(|f| f.seq > resume_from_seq)
                .map(|f| (f.seq, f.data.clone()))
                .collect::<Vec<_>>();
            let oldest_review = active
                .review_stream_replay
                .front()
                .map(|f| f.seq)
                .unwrap_or(0);
            let review_stream_frames = active
                .review_stream_replay
                .iter()
                .filter(|f| f.seq > resume_from_review_stream_seq)
                .map(|f| {
                    (
                        f.seq,
                        f.request_id.clone(),
                        f.stream.clone(),
                        f.data.clone(),
                    )
                })
                .collect::<Vec<_>>();
            (
                oldest,
                frames,
                oldest_review,
                review_stream_frames,
                active.last_review_progress.clone(),
                active.last_review_result.clone(),
                active.review_job.as_ref().map(|job| job.request_id.clone()),
            )
        };

        let skipped_stdout_replay = should_skip_stdout_replay(resume_from_seq, oldest_seq, &frames);
        if !frames.is_empty() {
            if skipped_stdout_replay {
                if let Some(message) =
                    build_skipped_stdout_replay_status_message(resume_from_seq, oldest_seq, &frames)
                {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        build_connection_notice_payload(message, "warn", "terminal", 5200),
                    )
                    .await?;
                }
            } else {
                for (seq, chunk) in &frames {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        json!({"kind":"stdout","seq":seq,"data":chunk}),
                    )
                    .await?;
                }
            }
        }

        let skipped_review_stream_replay = should_skip_review_stream_replay(
            resume_from_review_stream_seq,
            oldest_review_stream_seq,
            review_stream_frames.len(),
        );
        if !review_stream_frames.is_empty() {
            if skipped_review_stream_replay {
                if let Some(message) = build_skipped_review_stream_replay_status_message(
                    resume_from_review_stream_seq,
                    oldest_review_stream_seq,
                    review_stream_frames.len(),
                ) {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        build_connection_notice_payload(message, "warn", "review", 5200),
                    )
                    .await?;
                }
            } else {
                for (seq, request_id, stream, data) in &review_stream_frames {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        json!({
                            "kind":"review_stream",
                            "seq": seq,
                            "request_id": request_id,
                            "stream": stream,
                            "data": data,
                        }),
                    )
                    .await?;
                }
            }
        }

        let has_last_progress = last_review_progress.is_some();
        if let Some(payload) = last_review_progress {
            self.send_encrypted_json_for_session(session_id, ws_write, payload)
                .await?;
        }
        if let Some(payload) = last_review_result {
            self.send_encrypted_json_for_session(session_id, ws_write, payload)
                .await?;
        }
        if running_review_request_id.is_some()
            && (frames.is_empty() || skipped_stdout_replay)
            && !has_last_progress
        {
            self.send_encrypted_json_for_session(
                session_id,
                ws_write,
                json!({
                    "kind":"status",
                    "message":"Review is still running. Waiting for result..."
                }),
            )
            .await?;
        }

        Ok(())
    }

    pub(crate) async fn handle_ws_message<S>(
        &mut self,
        msg: Message,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        // tokio-tungstenite 0.29+ wraps text payloads in `Utf8Bytes`. Coerce
        // both arms to `String` for downstream JSON parsing.
        let text: String = match msg {
            Message::Text(t) => t.to_string(),
            Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
            Message::Ping(_) | Message::Pong(_) => return Ok(()),
            Message::Close(_) => return Ok(()),
            _ => return Ok(()),
        };
        let outer: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let event = outer.get("event").and_then(|v| v.as_str()).unwrap_or("");
        match event {
            "heartbeat" => {
                Self::send_plain_json(ws_write, json!({"type":"heartbeat"})).await?;
                return Ok(());
            }
            "status" => {
                if let Some(message) = outer.get("message").and_then(|v| v.as_str()) {
                    debug!(message, "Broker status");
                }
                return Ok(());
            }
            "relay" => {}
            _ => return Ok(()),
        }
        if outer.get("from").and_then(|v| v.as_str()) != Some("client") {
            return Ok(());
        }
        let session_id = outer
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let payload_raw = outer
            .get("payload")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let frame: Value = match serde_json::from_str(&payload_raw) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let frame_type = frame.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match frame_type {
            "heartbeat" => Ok(()),
            "e2ee_client_hello" => {
                self.handle_client_hello(&frame, &session_id, ws_write)
                    .await
            }
            "e2ee" => {
                self.handle_encrypted_frame(&frame, &session_id, ws_write)
                    .await
            }
            _ => Ok(()),
        }
    }

    pub(crate) async fn handle_client_hello<S>(
        &mut self,
        frame: &Value,
        relay_session_id: &str,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let client_pub_b64 = frame
            .get("client_pub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if client_pub_b64.is_empty() || relay_session_id.is_empty() {
            Self::send_plain_json(
                ws_write,
                json!({"type":"error","message":"Invalid client hello payload."}),
            )
            .await?;
            return Ok(());
        }
        if let Ok(owner_user_id) = self.cfg.user_id.parse::<u64>()
            && let Err(err) = self
                .host
                .set_last_active_relay_session_id(owner_user_id, relay_session_id)
                .await
        {
            warn!(
                error = %err,
                relay_session_id,
                "Failed to persist last active relay session id"
            );
        }
        let resume_from_seq = frame
            .get("resume_from_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let resume_from_review_stream_seq = frame
            .get("resume_from_review_stream_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let requested_agent = normalize_agent(frame.get("agent").and_then(|v| v.as_str()));
        let requested_agent_args = requested_agent_args_from_client_hello(frame);
        let requested_relay_session_id = requested_relay_session_id_from_client_hello(frame);
        let telegram_session_id = requested_telegram_session_id_from_client_hello(frame);
        let requested_fresh_start = requested_fresh_start_from_client_hello(frame);
        let mut remap_source_session_id: Option<String> = None;
        let mut remap_reason: Option<&'static str> = None;
        if let Some(explicit_relay_session_id) = requested_relay_session_id.as_deref()
            && explicit_relay_session_id != relay_session_id
            && self.sessions.contains_key(explicit_relay_session_id)
        {
            remap_source_session_id = Some(explicit_relay_session_id.to_string());
            remap_reason = Some("explicit relay_session_id from client hello");
        }
        if remap_source_session_id.is_none()
            && let Some(telegram_session_id) = telegram_session_id.as_deref()
        {
            match self
                .host
                .resolve_relay_for_telegram_session(telegram_session_id)
                .await
            {
                Ok(Some(mapped)) => {
                    if mapped != relay_session_id && self.sessions.contains_key(&mapped) {
                        remap_source_session_id = Some(mapped);
                        remap_reason = Some("telegram session binding");
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    warn!(
                        error = %err,
                        telegram_session_id = %telegram_session_id,
                        "Failed to resolve Telegram-to-relay mapping during client hello"
                    );
                }
            }
        }
        if let Some(from_session_id) = remap_source_session_id.as_deref() {
            if let Err(err) = self
                .remap_active_session_id(from_session_id, relay_session_id)
                .await
            {
                warn!(
                    error = %err,
                    from_session_id = %from_session_id,
                    to_session_id = %relay_session_id,
                    telegram_session_id = telegram_session_id.as_deref().unwrap_or(""),
                    requested_relay_session_id = requested_relay_session_id.as_deref().unwrap_or(""),
                    "Failed to remap native session during client hello"
                );
            } else {
                info!(
                    from_session_id = %from_session_id,
                    to_session_id = %relay_session_id,
                    remap_reason = remap_reason.unwrap_or("unknown"),
                    telegram_session_id = telegram_session_id.as_deref().unwrap_or(""),
                    requested_relay_session_id = requested_relay_session_id.as_deref().unwrap_or(""),
                    "Remapped active native session to client relay session id"
                );
            }
        }
        let cipher = Self::derive_relay_cipher(
            &self.key_material.private_key,
            relay_session_id,
            &client_pub_b64,
        )?;

        // Only tear down an already-bootstrapped session when the client explicitly
        // asks for a fresh start. A plain Mini App reopen resets local replay cursors
        // to zero, so resume_from_seq == 0 alone must not trigger session teardown.
        if requested_fresh_start
            && requested_agent.is_some()
            && resume_from_seq == 0
            && let Some(active) = self.sessions.get(relay_session_id)
            && active.crypto.bootstrapped_agent
        {
            info!(
                relay_session_id,
                old_agent = active.crypto.agent.as_deref().unwrap_or("none"),
                new_agent = requested_agent.as_deref().unwrap_or("none"),
                "Tearing down stale session for explicit fresh agent start"
            );
            self.remove_session(relay_session_id, "fresh agent start")
                .await;
        }

        if let Some(active) = self.sessions.get_mut(relay_session_id) {
            active.crypto.cipher = cipher;
            active.crypto.send_counter = 0;
            active.crypto.recv_counter = 0;
            if !active.crypto.bootstrapped_agent {
                active.crypto.agent = requested_agent.clone();
                active.crypto.agent_args = requested_agent_args.clone();
            }
            if let Some(telegram_session_id) = telegram_session_id.as_deref()
                && let Err(err) = self
                    .host
                    .bind_telegram_session_to_relay(telegram_session_id, relay_session_id)
                    .await
            {
                warn!(
                    error = %err,
                    telegram_session_id = %telegram_session_id,
                    relay_session_id = %relay_session_id,
                    "Failed to persist Telegram-to-relay session mapping"
                );
            }
            Self::send_plain_json(
                ws_write,
                json!({
                    "type":"e2ee_server_ready",
                    "session_id": relay_session_id,
                    "fingerprint": self.key_material.fingerprint,
                }),
            )
            .await?;

            self.send_encrypted_json_for_session(
                relay_session_id,
                ws_write,
                json!({
                    "kind":"status",
                    "message":"Local daemon secure channel ready (reattached). Session is active; if the terminal looks blank, press Enter or send a command to refresh output."
                }),
            )
            .await?;
            self.replay_stdout_since(
                relay_session_id,
                resume_from_seq,
                resume_from_review_stream_seq,
                ws_write,
            )
            .await?;
            self.bootstrap_agent(relay_session_id, ws_write).await?;
            return Ok(());
        }

        let cwd = resolve_session_cwd(
            frame.get("cwd").and_then(|v| v.as_str()),
            &self.cfg.default_cwd,
        );

        let shell = match ShellProcess::spawn(
            &self.cfg.shell,
            &cwd,
            relay_session_id,
            self.shell_events_tx.clone(),
        )
        .await
        {
            Ok(shell) => shell,
            Err(err) => {
                Self::send_plain_json(
                    ws_write,
                    json!({"type":"error","message":format!("Failed to spawn shell: {}", err)}),
                )
                .await?;
                return Ok(());
            }
        };

        let crypto = CryptoSession {
            session_id: relay_session_id.to_string(),
            cipher,
            send_counter: 0,
            recv_counter: 0,
            agent: requested_agent,
            agent_args: requested_agent_args,
            bootstrapped_agent: false,
        };

        self.sessions.insert(
            relay_session_id.to_string(),
            ActiveSession {
                crypto,
                shell,
                cwd,
                replay: VecDeque::new(),
                replay_bytes: 0,
                next_stdout_seq: 1,
                review_stream_replay: VecDeque::new(),
                review_stream_replay_bytes: 0,
                next_review_stream_seq: 1,
                pending_uploads: HashMap::new(),
                review_job: None,
                last_review_progress: None,
                last_review_result: None,
            },
        );
        if let Some(telegram_session_id) = telegram_session_id.as_deref()
            && let Err(err) = self
                .host
                .bind_telegram_session_to_relay(telegram_session_id, relay_session_id)
                .await
        {
            warn!(
                error = %err,
                telegram_session_id = %telegram_session_id,
                relay_session_id = %relay_session_id,
                "Failed to persist Telegram-to-relay session mapping"
            );
        }

        Self::send_plain_json(
            ws_write,
            json!({
                "type":"e2ee_server_ready",
                "session_id": relay_session_id,
                "fingerprint": self.key_material.fingerprint,
            }),
        )
        .await?;

        self.send_encrypted_json_for_session(
            relay_session_id,
            ws_write,
            json!({"kind":"status","message":"Local daemon secure channel ready."}),
        )
        .await?;
        self.replay_stdout_since(
            relay_session_id,
            resume_from_seq,
            resume_from_review_stream_seq,
            ws_write,
        )
        .await?;
        self.bootstrap_agent(relay_session_id, ws_write).await?;
        Ok(())
    }

    pub(crate) async fn bootstrap_agent<S>(
        &mut self,
        session_id: &str,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let crypto = &mut active.crypto;
        let Some(agent) = crypto.agent.clone() else {
            return Ok(());
        };
        if crypto.bootstrapped_agent {
            return Ok(());
        }
        if crypto
            .agent_args
            .iter()
            .any(|arg| !is_safe_agent_bootstrap_arg(arg))
        {
            anyhow::bail!("unsafe agent argument rejected for shell bootstrap");
        }
        let mut command_parts = Vec::with_capacity(1 + crypto.agent_args.len());
        command_parts.push(shell_token(&agent));
        command_parts.extend(crypto.agent_args.iter().map(|arg| shell_token(arg)));
        let command = format!("{}\n", command_parts.join(" "));
        active.shell.write_stdin(&command).await?;
        crypto.bootstrapped_agent = true;
        let args_suffix = if crypto.agent_args.is_empty() {
            String::new()
        } else {
            format!(" {}", crypto.agent_args.join(" "))
        };
        self.send_encrypted_json_for_session(
            session_id,
            ws_write,
            json!({"kind":"status","message":format!("Starting {}{}...", agent, args_suffix)}),
        )
        .await?;
        Ok(())
    }
}
