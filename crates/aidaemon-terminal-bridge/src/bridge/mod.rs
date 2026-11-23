//! The bridge's run loop: connecting to the daemon and reconnecting.

use crate::*;

mod frames;
mod local;
mod record;
mod relay;
mod shell_events;

pub(crate) struct TerminalBridge {
    pub(crate) cfg: RuntimeConfig,
    pub(crate) key_material: KeyMaterial,
    pub(crate) device_id: String,
    pub(crate) sessions: HashMap<String, ActiveSession>,
    pub(crate) host: Arc<dyn BridgeHost>,
    pub(crate) http_client: reqwest::Client,
    pub(crate) shell_events_tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    pub(crate) shell_events_rx: tokio::sync::mpsc::UnboundedReceiver<ShellEvent>,
    pub(crate) local_attach_secret: String,
    pub(crate) local_clients: HashMap<u64, LocalAttachClient>,
    pub(crate) local_events_rx: tokio::sync::mpsc::UnboundedReceiver<LocalAttachEvent>,
}

impl TerminalBridge {
    pub(crate) async fn new(
        cfg: RuntimeConfig,
        host: Arc<dyn BridgeHost>,
        secrets: Arc<dyn SecretStore>,
    ) -> anyhow::Result<Self> {
        let key_material = load_or_create_key_material(secrets.as_ref()).await?;
        let device_id = cfg
            .device_id
            .clone()
            .unwrap_or_else(|| default_device_id(&key_material.fingerprint));
        let http_client = reqwest::Client::builder()
            .user_agent("aidaemon-terminal-bridge/1.0")
            .timeout(Duration::from_secs(12))
            .build()?;
        let (shell_events_tx, shell_events_rx) =
            tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
        let (local_events_tx, local_events_rx) =
            tokio::sync::mpsc::unbounded_channel::<LocalAttachEvent>();
        let local_attach_secret = random_nonce_hex(LOCAL_ATTACH_SECRET_BYTES);
        if let Err(err) =
            spawn_local_attach_listener(local_events_tx.clone(), local_attach_secret.clone()).await
        {
            warn!(
                error = %err,
                "Failed to start local attach endpoint; agent attach from native terminal will be unavailable"
            );
        }
        info!(
            fingerprint = %key_material.fingerprint,
            user_id = %cfg.user_id,
            device_id = %device_id,
            "Terminal bridge initialized"
        );
        Ok(Self {
            cfg,
            key_material,
            device_id,
            sessions: HashMap::new(),
            host,
            http_client,
            shell_events_tx,
            shell_events_rx,
            local_attach_secret,
            local_clients: HashMap::new(),
            local_events_rx,
        })
    }

    pub(crate) async fn run_forever(&mut self) {
        let stable_threshold = Duration::from_secs(RECONNECT_STABLE_SECS);
        let mut reconnect_backoff_ms = RECONNECT_INITIAL_MS;
        loop {
            self.drain_shell_events(512);
            self.drain_local_events(256).await;
            let started = Instant::now();
            match self.connect_once().await {
                Ok(()) => {
                    info!(
                        ran_for_secs = started.elapsed().as_secs(),
                        "Terminal bridge disconnected, reconnecting"
                    );
                }
                Err(err) => {
                    if Self::is_expected_disconnect(&err) {
                        warn!(
                            error = %err,
                            ran_for_secs = started.elapsed().as_secs(),
                            "Terminal bridge disconnected (transient), reconnecting"
                        );
                    } else {
                        error!(
                            error = %err,
                            broker_response = %Self::broker_rejection_body(&err).unwrap_or_default(),
                            ran_for_secs = started.elapsed().as_secs(),
                            "Terminal bridge connection failed"
                        );
                    }
                }
            }
            let ran_for = started.elapsed();
            self.drain_shell_events(2048);
            self.drain_local_events(512).await;

            if ran_for >= stable_threshold {
                reconnect_backoff_ms = RECONNECT_INITIAL_MS;
            }
            let jitter_ms = if RECONNECT_JITTER_MS == 0 {
                0
            } else {
                rand::rng().random_range(0..=RECONNECT_JITTER_MS)
            };
            let sleep_ms = reconnect_backoff_ms.saturating_add(jitter_ms);
            debug!(
                sleep_ms,
                base_backoff_ms = reconnect_backoff_ms,
                jitter_ms,
                "Sleeping before terminal bridge reconnect"
            );
            tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
            reconnect_backoff_ms = Self::next_reconnect_backoff_ms(reconnect_backoff_ms);
        }
    }

    pub(crate) fn next_reconnect_backoff_ms(current: u64) -> u64 {
        current
            .saturating_mul(2)
            .clamp(RECONNECT_INITIAL_MS, RECONNECT_MAX_MS)
    }

    /// Classifies a disconnect as routine for a long-lived broker connection: an idle/proxy
    /// reset, a peer reset, or a normal close. The reconnect loop recovers from all of these
    /// automatically, so they are logged at WARN instead of ERROR to avoid drowning the logs
    /// in false alarms (the broker resets long-lived sockets every few minutes by design).
    pub(crate) fn is_expected_disconnect(err: &anyhow::Error) -> bool {
        use tokio_tungstenite::tungstenite::Error as WsError;
        use tokio_tungstenite::tungstenite::error::ProtocolError;
        match err.downcast_ref::<WsError>() {
            Some(WsError::ConnectionClosed)
            | Some(WsError::AlreadyClosed)
            | Some(WsError::Protocol(ProtocolError::ResetWithoutClosingHandshake)) => true,
            Some(WsError::Io(io)) => matches!(
                io.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::TimedOut
            ),
            _ => false,
        }
    }

    /// The body of an HTTP rejection of the websocket upgrade, which says why
    /// the broker refused (for a 409, which other device holds the slot).
    pub(crate) fn broker_rejection_body(err: &anyhow::Error) -> Option<String> {
        use tokio_tungstenite::tungstenite::Error as WsError;
        let Some(WsError::Http(response)) = err.downcast_ref::<WsError>() else {
            return None;
        };
        let body = response.body().as_deref()?;
        Some(String::from_utf8_lossy(body).chars().take(300).collect())
    }

    pub(crate) async fn connect_once(&mut self) -> anyhow::Result<()> {
        let connect_token = {
            let auth = self.cfg.auth.clone();
            let user_id = self.cfg.user_id.clone();
            let device_id = self.device_id.clone();
            let http_client = self.http_client.clone();
            let host = self.host.clone();
            resolve_connect_token(auth, host, user_id, device_id, http_client).await?
        };
        let mut ws_url = reqwest::Url::parse(&self.cfg.ws_url)?;
        ws_url
            .query_pairs_mut()
            .append_pair("user_id", &self.cfg.user_id)
            .append_pair("device_id", &self.device_id);

        let mut req = ws_url.as_str().into_client_request()?;
        req.headers_mut().insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", connect_token))?,
        );
        debug!(
            has_ws_key = req.headers().contains_key("sec-websocket-key"),
            upgrade = ?req.headers().get("upgrade"),
            connection = ?req.headers().get("connection"),
            "Prepared daemon websocket request headers"
        );
        info!("Connecting terminal bridge to broker");
        let tls = aidaemon_tools::tls::client_config()?;
        let (ws_stream, _) =
            connect_async_tls_with_config(req, None, false, Some(Connector::Rustls(tls))).await?;
        info!("Terminal bridge connected");

        let (mut ws_write, mut ws_read) = ws_stream.split();
        let mut outbound_queue = OutboundQueue::default();
        let mut heartbeat = tokio::time::interval(Duration::from_millis(HEARTBEAT_MS));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        Self::send_plain_json(
            &mut ws_write,
            json!({
                "type": "daemon_hello",
                "alg": "ECDH-P256/AES-256-GCM",
                "daemon_pub": b64_encode(&self.key_material.public_raw),
                "fingerprint": self.key_material.fingerprint,
            }),
        )
        .await?;

        loop {
            tokio::select! {
                biased;
                maybe_msg = ws_read.next() => {
                    match maybe_msg {
                        Some(Ok(msg)) => {
                            if let Err(err) = self.handle_ws_message(msg, &mut ws_write).await {
                                warn!(error=%err, "Terminal bridge message handling error");
                            }
                        }
                        Some(Err(err)) => return Err(anyhow::anyhow!(err)),
                        None => return Ok(()),
                    }
                }
                maybe_local_event = self.local_events_rx.recv() => {
                    let Some(event) = maybe_local_event else {
                        continue;
                    };
                    if let Err(err) = self.handle_local_attach_event(event).await {
                        warn!(error = %err, "Local attach event handling failed");
                    }
                }
                _ = heartbeat.tick() => {
                    let _ = Self::send_plain_json(&mut ws_write, json!({"type":"heartbeat"})).await;
                    let mut exited = Vec::new();
                    for (session_id, active) in self.sessions.iter_mut() {
                        if let Ok(Some(code)) = active.shell.try_wait_code() {
                            exited.push((session_id.clone(), code));
                        }
                    }
                    for (session_id, code) in exited {
                        let _ = self
                            .send_encrypted_json_for_session(
                                &session_id,
                                &mut ws_write,
                                json!({"kind":"exit","code":code}),
                            )
                            .await;
                        self.remove_session(&session_id, "shell exited").await;
                    }
                }
                maybe_event = self.shell_events_rx.recv() => {
                    let Some(event) = maybe_event else {
                        continue;
                    };
                    if let Err(err) = self.handle_shell_event(event, Some(&mut outbound_queue)) {
                        warn!(error = %err, "Failed to relay shell output event");
                        continue;
                    }
                    // Under continuous PTY output (spinner frames, progress redraws), keep draining
                    // outbound payloads immediately so low-priority stdout doesn't get starved.
                    if !outbound_queue.is_empty() {
                        self.flush_outbound_queue(&mut outbound_queue, &mut ws_write, OUTBOUND_FLUSH_BURST)
                            .await?;
                    }
                }
                _ = std::future::ready(()), if !outbound_queue.is_empty() => {
                    self.flush_outbound_queue(&mut outbound_queue, &mut ws_write, OUTBOUND_FLUSH_BURST)
                        .await?;
                }
            }
        }
    }

    pub(crate) fn derive_relay_cipher(
        daemon_private_key: &SecretKey,
        relay_session_id: &str,
        client_pub_b64: &str,
    ) -> anyhow::Result<Aes256Gcm> {
        let client_pub_raw = b64_decode(client_pub_b64)?;
        let client_pub = PublicKey::from_sec1_bytes(&client_pub_raw)?;
        let shared = diffie_hellman(
            daemon_private_key.to_nonzero_scalar(),
            client_pub.as_affine(),
        );

        let mut salt_hash = Sha256::new();
        salt_hash.update(relay_session_id.as_bytes());
        let salt = salt_hash.finalize();
        #[allow(deprecated)]
        let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
        let mut key = [0u8; 32];
        hk.expand(KEY_INFO, &mut key)
            .map_err(|_| anyhow::anyhow!("HKDF key expansion failed"))?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| anyhow::anyhow!("failed to build AES-256-GCM cipher"))?;
        Ok(cipher)
    }
}
