//! Sessions started and driven from local attach clients.

use crate::*;

impl TerminalBridge {
    pub(crate) fn owner_user_id(&self) -> Option<u64> {
        self.cfg.user_id.parse::<u64>().ok()
    }

    pub(crate) fn allocate_local_relay_session_id(&self) -> String {
        for _ in 0..16 {
            let candidate = format!("native-{}", random_nonce_hex(8));
            if !self.sessions.contains_key(&candidate) {
                return candidate;
            }
        }
        format!("native-{}", random_nonce_hex(12))
    }

    pub(crate) async fn start_local_bridge_session(
        &mut self,
        agent: String,
        cwd_hint: Option<String>,
        agent_args: Vec<String>,
    ) -> anyhow::Result<String> {
        let session_id = self.allocate_local_relay_session_id();
        let cwd = resolve_session_cwd(cwd_hint.as_deref(), &self.cfg.default_cwd);
        let shell = ShellProcess::spawn(
            &self.cfg.shell,
            &cwd,
            &session_id,
            self.shell_events_tx.clone(),
        )
        .await?;

        let mut key = [0u8; 32];
        UnwrapErr(SysRng).fill_bytes(&mut key);
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| anyhow::anyhow!("failed to initialize local relay cipher"))?;

        self.sessions.insert(
            session_id.clone(),
            ActiveSession {
                crypto: CryptoSession {
                    session_id: session_id.clone(),
                    cipher,
                    send_counter: 0,
                    recv_counter: 0,
                    agent: Some(agent),
                    agent_args: agent_args.clone(),
                    bootstrapped_agent: false,
                },
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

        // Native-started sessions still run through the same shell bootstrap path.
        if let Some(active) = self.sessions.get_mut(&session_id)
            && let Some(agent_name) = active.crypto.agent.clone()
        {
            if active
                .crypto
                .agent_args
                .iter()
                .any(|arg| !is_safe_agent_bootstrap_arg(arg))
            {
                anyhow::bail!("unsafe agent argument rejected for shell bootstrap");
            }
            let mut command_parts = Vec::with_capacity(1 + active.crypto.agent_args.len());
            command_parts.push(shell_token(&agent_name));
            command_parts.extend(active.crypto.agent_args.iter().map(|arg| shell_token(arg)));
            active
                .shell
                .write_stdin(&format!("{}\n", command_parts.join(" ")))
                .await?;
            active.crypto.bootstrapped_agent = true;
        }

        if let Some(owner_user_id) = self.owner_user_id()
            && let Err(err) = self
                .host
                .set_last_active_relay_session_id(owner_user_id, &session_id)
                .await
        {
            warn!(
                error = %err,
                session_id = %session_id,
                "Failed to persist last active relay session after native start"
            );
        }

        Ok(session_id)
    }

    pub(crate) async fn handle_local_attach_event(
        &mut self,
        event: LocalAttachEvent,
    ) -> anyhow::Result<()> {
        match event {
            LocalAttachEvent::AttachRequest {
                client_id,
                code,
                secret,
                tx,
                response_tx,
            } => {
                if !crate::local_attach::same_secret(&secret, &self.local_attach_secret) {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local attach authorization failed")));
                    return Ok(());
                }
                let handoff = match self.host.resolve_handoff_code(&code).await {
                    Ok(value) => value,
                    Err(err) => {
                        let _ = response_tx.send(Err(err));
                        return Ok(());
                    }
                };
                let session_id = handoff.relay_session_id.clone();
                if !self.sessions.contains_key(&session_id) {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "Session is no longer active. Start /agent open again from Telegram."
                    )));
                    return Ok(());
                }
                if let Err(err) = self.host.consume_handoff_code(&code).await {
                    let _ = response_tx.send(Err(err));
                    return Ok(());
                }

                self.local_clients.insert(
                    client_id,
                    LocalAttachClient {
                        session_id: session_id.clone(),
                        tx: tx.clone(),
                    },
                );

                let replay = self
                    .sessions
                    .get(&session_id)
                    .map(|active| {
                        active
                            .replay
                            .iter()
                            .rev()
                            .take(64)
                            .map(|frame| frame.data.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                for chunk in replay {
                    let _ = tx.send(json!({"type":"stdout","data":chunk}));
                }
                let _ = tx.send(json!({
                    "type":"status",
                    "message":"Attached to active session. Native terminal is now live.",
                }));
                let _ = response_tx.send(Ok(LocalAttachAccepted {
                    session_id,
                    resume_code: None,
                    resume_expires_at_unix: None,
                }));
            }
            LocalAttachEvent::StartRequest {
                client_id,
                secret,
                agent,
                cwd,
                agent_args,
                tx,
                response_tx,
            } => {
                if !crate::local_attach::same_secret(&secret, &self.local_attach_secret) {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local start authorization failed")));
                    return Ok(());
                }

                let session_id = match self
                    .start_local_bridge_session(agent, cwd, agent_args)
                    .await
                {
                    Ok(id) => id,
                    Err(err) => {
                        let _ = response_tx.send(Err(err));
                        return Ok(());
                    }
                };

                self.local_clients.insert(
                    client_id,
                    LocalAttachClient {
                        session_id: session_id.clone(),
                        tx: tx.clone(),
                    },
                );
                let replay = self
                    .sessions
                    .get(&session_id)
                    .map(|active| {
                        active
                            .replay
                            .iter()
                            .rev()
                            .take(64)
                            .map(|frame| frame.data.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                for chunk in replay {
                    let _ = tx.send(json!({"type":"stdout","data":chunk}));
                }

                let mut resume_code: Option<String> = None;
                let mut resume_expires_at_unix: Option<i64> = None;
                if let Some(owner_user_id) = self.owner_user_id() {
                    match self
                        .host
                        .create_handoff_code(&session_id, owner_user_id)
                        .await
                    {
                        Ok(handoff) => {
                            resume_code = Some(handoff.code);
                            resume_expires_at_unix = Some(handoff.expires_at_unix);
                        }
                        Err(err) => {
                            warn!(
                                error = %err,
                                session_id = %session_id,
                                "Failed to auto-create resume code for native-started session"
                            );
                        }
                    }
                }

                let _ = tx.send(json!({
                    "type":"status",
                    "message":"Native terminal session started. Use `aidaemon share` to generate a new Telegram resume code any time.",
                }));
                let _ = response_tx.send(Ok(LocalAttachAccepted {
                    session_id,
                    resume_code,
                    resume_expires_at_unix,
                }));
            }
            LocalAttachEvent::ShareRequest {
                secret,
                session_id,
                response_tx,
            } => {
                if !crate::local_attach::same_secret(&secret, &self.local_attach_secret) {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local share authorization failed")));
                    return Ok(());
                }
                let Some(owner_user_id) = self.owner_user_id() else {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "Configured user id is not numeric; cannot generate share code."
                    )));
                    return Ok(());
                };

                let resolved_session_id = if let Some(explicit) = session_id {
                    explicit
                } else if let Ok(Some(last)) = self
                    .host
                    .get_last_active_relay_session_id(owner_user_id)
                    .await
                {
                    last
                } else if self.sessions.len() == 1 {
                    self.sessions.keys().next().cloned().unwrap_or_default()
                } else {
                    String::new()
                };

                if resolved_session_id.is_empty()
                    || !self.sessions.contains_key(&resolved_session_id)
                {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "No active native session found. Start one with `aidaemon codex` (or claude/gemini/opencode) first."
                    )));
                    return Ok(());
                }

                let created = match self
                    .host
                    .create_handoff_code(&resolved_session_id, owner_user_id)
                    .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        let _ = response_tx.send(Err(err));
                        return Ok(());
                    }
                };
                let _ = response_tx.send(Ok(LocalShareCreated {
                    session_id: resolved_session_id,
                    code: created.code,
                    expires_at_unix: created.expires_at_unix,
                }));
            }
            LocalAttachEvent::Input {
                client_id,
                data_b64,
            } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                let bytes = match b64_decode_flexible(&data_b64) {
                    Ok(v) => v,
                    Err(_) => return Ok(()),
                };
                if bytes.is_empty() {
                    return Ok(());
                }
                let stdin = String::from_utf8_lossy(&bytes).to_string();
                let Some(active) = self.sessions.get_mut(&client.session_id) else {
                    let _ = client.tx.send(json!({
                        "type":"error",
                        "message":"Session is no longer active.",
                    }));
                    self.local_clients.remove(&client_id);
                    return Ok(());
                };
                if let Err(err) = active.shell.write_stdin(&stdin).await {
                    let _ = client.tx.send(json!({
                        "type":"error",
                        "message": format!("Failed to forward stdin: {}", err),
                    }));
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Resize {
                client_id,
                cols,
                rows,
            } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                if let Some(active) = self.sessions.get_mut(&client.session_id) {
                    let _ = active.shell.resize(cols, rows);
                } else {
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Redraw { client_id } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                if let Some(active) = self.sessions.get_mut(&client.session_id) {
                    let _ = active.shell.write_stdin("\u{000c}").await;
                } else {
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Close { client_id } => {
                self.local_clients.remove(&client_id);
            }
        }
        Ok(())
    }
}
