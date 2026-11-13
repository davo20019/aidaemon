// ==========================================================================
// The Client Gateway over a real socket
//
// An app's message runs as a turn on the daemon's own agent (MockProvider)
// through the shared turn runner; its owner-approval card reaches the app
// through the hub's approval tap; the client disconnects mid-turn and
// resumes from its last `seq`, missing nothing and seeing nothing twice.
// ==========================================================================

mod client_gateway {
    use std::sync::Arc;
    use std::time::Duration;

    use aidaemon_client_protocol::*;
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    type Socket = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    struct NoHost;

    #[async_trait::async_trait]
    impl aidaemon_client_gateway::ClientHost for NoHost {
        async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
            Ok(Vec::new())
        }
        async fn history(&self, _: &str, _: usize) -> anyhow::Result<Vec<HistoryMessage>> {
            Ok(Vec::new())
        }
        async fn slash(&self, _: &str, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    struct Daemon {
        url: String,
        token: String,
        posted: Arc<std::sync::atomic::AtomicUsize>,
        pool: sqlx::SqlitePool,
        hub: Arc<crate::channels::ChannelHub>,
        turns: aidaemon_channels::turns::TurnDeps,
        _dir: tempfile::TempDir,
        _files: crate::testing::HarnessFiles,
    }

    /// The agent asks to POST a report (an owner-approval card), then says
    /// it posted it. The card goes out through a real hub.
    async fn daemon() -> Daemon {
        let (port, posted) = super::spawn_approvals_server().await;
        let url = format!("http://localhost:{port}/report");
        let (harness, _scripted) = super::approvals_harness(
            vec![
                super::http_call("POST", &url, Some("report")),
                super::MockProvider::text_response("Posted the synthetic report."),
            ],
            Vec::new(),
            Some(Duration::from_secs(60)),
        )
        .await;
        let session_map = crate::channels::SessionMap::default();
        let hub = Arc::new(crate::channels::ChannelHub::new(
            Vec::new(),
            session_map.clone(),
        ));
        let (approval_tx, approval_rx) = tokio::sync::mpsc::channel(16);
        harness
            .agent
            .owner_approvals()
            .set_broker(crate::tools::ApprovalBroker::new(approval_tx.clone()));
        tokio::spawn(hub.clone().approval_listener(approval_rx));

        let (agent, files) = harness.into_agent();
        let state = files.state.clone();
        let pool = state.pool();
        let agent: Arc<dyn aidaemon_channels::ports::ChannelAgentRuntime> = Arc::new(agent);
        let dir = tempfile::tempdir().unwrap();
        let token_file = dir.path().join("client-token.sha256");
        let token = aidaemon_client_gateway::token::mint(&token_file).unwrap();
        let turns = aidaemon_channels::turns::TurnDeps {
            agent,
            session_map,
            sessions: state.clone(),
            task_registry: Arc::new(crate::tasks::TaskRegistry::new(50)),
            watchdog_stale_threshold_secs: 0,
            watch: Default::default(),
        };
        let gateway =
            aidaemon_client_gateway::ClientGateway::start(aidaemon_client_gateway::GatewayDeps {
                turns: turns.clone(),
                hub: hub.clone(),
                host: Arc::new(NoHost),
                log: aidaemon_store_sqlite::SqliteClientEventLog::open(pool.clone())
                    .await
                    .unwrap(),
                config: aidaemon_client_gateway::ClientGatewayConfig {
                    enabled: true,
                    ..Default::default()
                },
                token_file,
                approvals: Some(approval_tx),
            })
            .await
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = aidaemon_client_gateway::router(gateway);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Daemon {
            url: format!("127.0.0.1:{}", address.port()),
            token,
            posted,
            pool,
            hub,
            turns,
            _dir: dir,
            _files: files,
        }
    }

    async fn connect(daemon: &Daemon, resume_from: Option<u64>) -> Socket {
        let mut request = tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(
            format!("ws://{}/client/v1/ws", daemon.url),
        )
        .unwrap();
        request.headers_mut().insert(
            "authorization",
            format!("Bearer {}", daemon.token).parse().unwrap(),
        );
        let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        let hello = ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: ClientKind::Tui,
                name: "integration-test".into(),
                version: None,
                device: Some("Synthetic Laptop".into()),
            },
            capabilities: Vec::new(),
            resume_from,
            token: None,
        };
        socket
            .send(Message::text(serde_json::to_string(&hello).unwrap()))
            .await
            .unwrap();
        socket
    }

    async fn next(socket: &mut Socket) -> ServerFrame {
        loop {
            let message = tokio::time::timeout(Duration::from_secs(30), socket.next())
                .await
                .expect("a frame in time")
                .expect("the socket is open")
                .unwrap();
            if let Message::Text(text) = message {
                return serde_json::from_str(&text).unwrap();
            }
        }
    }

    async fn command(socket: &mut Socket, request_id: &str, command: Command) {
        let frame = ClientFrame::Command {
            request_id: request_id.into(),
            command,
        };
        socket
            .send(Message::text(serde_json::to_string(&frame).unwrap()))
            .await
            .unwrap();
    }

    /// Every frame the client received, and the durable events it applied
    /// (by `seq`).
    #[derive(Default)]
    struct Applied {
        seqs: Vec<u64>,
        events: Vec<Event>,
        frames: Vec<ServerFrame>,
        /// Frames already handed to a test step.
        taken: std::collections::HashSet<usize>,
    }

    impl Applied {
        fn apply(&mut self, frame: &ServerFrame) {
            if let ServerFrame::Event {
                seq: Some(seq),
                event,
                ..
            } = frame
            {
                self.seqs.push(*seq);
                self.events.push(event.clone());
            }
            self.frames.push(frame.clone());
        }

        fn last(&self) -> u64 {
            self.seqs.last().copied().unwrap_or(0)
        }

        fn take(&mut self, want: &impl Fn(&ServerFrame) -> bool) -> Option<ServerFrame> {
            let at = (0..self.frames.len())
                .find(|at| !self.taken.contains(at) && want(&self.frames[*at]))?;
            self.taken.insert(at);
            Some(self.frames[at].clone())
        }
    }

    /// The first frame not yet taken that `want` matches, received already
    /// (replies and events interleave) or read now.
    async fn until(
        socket: &mut Socket,
        applied: &mut Applied,
        want: impl Fn(&ServerFrame) -> bool,
    ) -> ServerFrame {
        loop {
            if let Some(frame) = applied.take(&want) {
                return frame;
            }
            let frame = next(socket).await;
            applied.apply(&frame);
        }
    }

    fn is_event(frame: &ServerFrame, kind: &str) -> bool {
        matches!(frame, ServerFrame::Event { event, .. } if event.kind() == kind)
    }

    #[tokio::test]
    async fn a_turn_with_an_approval_survives_a_disconnect_without_loss_or_duplicates() {
        let daemon = daemon().await;
        let mut applied = Applied::default();

        // First connection: no cache, so a snapshot, then the message.
        let mut socket = connect(&daemon, None).await;
        assert!(matches!(
            next(&mut socket).await,
            ServerFrame::Welcome { latest_seq: 0, .. }
        ));
        assert!(matches!(
            next(&mut socket).await,
            ServerFrame::Snapshot { seq: 0, .. }
        ));
        command(
            &mut socket,
            "send-1",
            Command::SendMessage {
                session_id: None,
                text: "Post the synthetic report for Alice Rivera".into(),
                attachments: Vec::new(),
            },
        )
        .await;
        let ack = until(&mut socket, &mut applied, |f| {
            matches!(f, ServerFrame::Ack { .. })
        })
        .await;
        let ServerFrame::Ack {
            result: CommandResult::MessageAccepted { session_id, .. },
            ..
        } = ack
        else {
            panic!("{ack:?}");
        };
        until(&mut socket, &mut applied, |f| is_event(f, "turn_started")).await;

        // Gone mid-turn: the card is asked while nobody is connected.
        drop(socket);
        for _ in 0..300 {
            let asked: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM client_events WHERE kind = 'approval_requested'",
            )
            .fetch_one(&daemon.pool)
            .await
            .unwrap();
            if asked > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // Back, from the last seq applied: the card is replayed.
        let mut socket = connect(&daemon, Some(applied.last())).await;
        assert!(matches!(next(&mut socket).await, ServerFrame::Welcome { .. }));
        let asked = until(&mut socket, &mut applied, |f| {
            is_event(f, "approval_requested")
        })
        .await;
        let ServerFrame::Event {
            event: Event::ApprovalRequested { approval },
            ..
        } = asked
        else {
            unreachable!()
        };
        assert_eq!(approval.session_id, session_id);
        assert_eq!(approval.kind, ApprovalKind::Command);
        assert!(approval.options.contains(&ApprovalDecision::ApproveOnce));
        assert_eq!(daemon.posted.load(std::sync::atomic::Ordering::SeqCst), 0);

        // Approve; a retry of the same request gets the same answer.
        let resolve = Command::ResolveApproval {
            approval_id: approval.id.clone(),
            decision: ApprovalDecision::ApproveOnce,
        };
        command(&mut socket, "approve-1", resolve.clone()).await;
        let first = until(&mut socket, &mut applied, |f| {
            matches!(f, ServerFrame::Ack { .. } | ServerFrame::Error { .. })
        })
        .await;
        assert!(
            matches!(
                &first,
                ServerFrame::Ack {
                    result: CommandResult::ApprovalResolved {
                        outcome: ApprovalOutcome::ApprovedOnce,
                        ..
                    },
                    ..
                }
            ),
            "{first:?}"
        );
        command(&mut socket, "approve-1", resolve).await;
        let retry = until(&mut socket, &mut applied, |f| {
            matches!(f, ServerFrame::Ack { .. } | ServerFrame::Error { .. })
        })
        .await;
        assert_eq!(retry, first);

        // Drop again (the turn is finishing meanwhile), then resume once more
        // from whatever was applied last.
        drop(socket);
        for _ in 0..500 {
            let done: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM client_events WHERE kind = 'task_updated' \
                 AND payload LIKE '%\"completed\"%'",
            )
            .fetch_one(&daemon.pool)
            .await
            .unwrap();
            if done > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let mut socket = connect(&daemon, Some(applied.last())).await;
        assert!(matches!(next(&mut socket).await, ServerFrame::Welcome { .. }));
        let finished = |applied: &Applied| {
            applied.events.iter().any(|e| {
                matches!(e, Event::TaskUpdated { task } if task.state == TaskState::Completed)
            })
        };
        while !finished(&applied) {
            let frame = next(&mut socket).await;
            applied.apply(&frame);
        }
        let Some(Event::TurnCompleted { text, state, .. }) = applied
            .events
            .iter()
            .find(|e| matches!(e, Event::TurnCompleted { .. }))
        else {
            panic!("no turn_completed applied");
        };
        assert_eq!(*state, TurnState::Completed);
        assert!(text.contains("Posted the synthetic report"), "{text}");
        assert_eq!(daemon.posted.load(std::sync::atomic::Ordering::SeqCst), 1);

        // No loss, no duplicates: what the client applied over three
        // connections is exactly the log, in order.
        let logged: Vec<i64> =
            sqlx::query_scalar("SELECT seq FROM client_events WHERE owner = ?1 ORDER BY seq")
                .bind(aidaemon_client_gateway::OWNER)
                .fetch_all(&daemon.pool)
                .await
                .unwrap();
        let logged: Vec<u64> = logged.into_iter().map(|s| s as u64).collect();
        assert_eq!(applied.seqs, logged);
        assert_eq!(applied.seqs, (1..=logged.len() as u64).collect::<Vec<_>>());
        let kinds: Vec<&str> = applied.events.iter().map(Event::kind).collect();
        assert_eq!(
            kinds,
            [
                "session_created",
                "user_message",
                "turn_started",
                "task_updated",
                "approval_requested",
                "approval_resolved",
                "turn_completed",
                "task_updated",
            ]
        );
        assert!(applied.events.iter().any(|e| matches!(
            e,
            Event::ApprovalResolved {
                resolved_by: Some(ResolvedBy { channel, device: Some(device) }),
                ..
            } if channel == "app" && device == "Synthetic Laptop"
        )));
    }

    /// The owner's Telegram chat: confirms pairings, and leaves every other
    /// card open (so the paired browser has to answer it).
    struct Telegram {
        cards: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl crate::traits::HubChannel for Telegram {
        fn name(&self) -> String {
            "telegram".into()
        }
        fn capabilities(&self) -> aidaemon_channels::hub::ChannelCapabilities {
            aidaemon_channels::hub::ChannelCapabilities {
                markdown: true,
                inline_buttons: true,
                media: true,
                max_message_len: 4096,
            }
        }
        async fn send_text(&self, _: &str, _: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media(
            &self,
            _: &str,
            _: &aidaemon_channels::hub::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn request_approval(
            &self,
            _: &str,
            command: &str,
            _: crate::types::RiskLevel,
            _: &[String],
            _: crate::types::PermissionMode,
            _: bool,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            self.cards.lock().unwrap().push(command.to_string());
            if command.starts_with("Pair a new device") {
                return Ok(crate::types::ApprovalResponse::AllowOnce);
            }
            std::future::pending().await
        }
    }

    /// Shows a Telegram turn's output nowhere.
    struct TelegramSurface;

    #[async_trait::async_trait]
    impl aidaemon_channels::turns::TurnSurface for TelegramSurface {
        fn platform(&self) -> &'static str {
            "telegram"
        }
        async fn say(&self, _: &str) -> anyhow::Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn show_working(
            &self,
            _: tokio_util::sync::CancellationToken,
            _: Arc<std::sync::atomic::AtomicU64>,
            _: u64,
        ) {
        }
    }

    async fn http_json<T: serde::de::DeserializeOwned>(
        response: reqwest::Response,
    ) -> T {
        assert!(response.status().is_success(), "{}", response.status());
        response.json().await.unwrap()
    }

    /// Phase 4 is done when a paired browser can approve a command that a
    /// Telegram-started turn requested: pair from the Telegram chat
    /// (confirmed there), sign in with the device key, and answer the turn's
    /// card over the socket.
    #[tokio::test]
    async fn a_paired_browser_approves_a_telegram_started_turn() {
        use base64::Engine;
        use p256::ecdsa::signature::Signer;
        use p256::elliptic_curve::Generate;
        const TELEGRAM_SESSION: &str = "telegram:synthetic-user-1";
        let daemon = daemon().await;
        let telegram = Arc::new(Telegram {
            cards: std::sync::Mutex::default(),
        });
        daemon.hub.register_channel(telegram.clone()).await;
        let runner = aidaemon_channels::turns::TurnRunner::new(daemon.turns.clone());
        runner.register_session(TELEGRAM_SESSION, "telegram").await;
        let http = reqwest::Client::new();
        let base = format!("http://{}", daemon.url);

        // The owner sends /pair_device in Telegram (the offer is confirmed
        // there); the browser redeems it with its own key.
        let offer: PairingOffer = http_json(
            http.post(format!("{base}/client/v1/pairing/offers"))
                .bearer_auth(&daemon.token)
                .json(&OfferRequest {
                    confirm_in_session: Some(TELEGRAM_SESSION.into()),
                })
                .send()
                .await
                .unwrap(),
        )
        .await;
        let key = p256::ecdsa::SigningKey::generate_from_rng(&mut rand::rng());
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let redeemed: Enrollment = http_json(
            http.post(format!("{base}/client/v1/enrollments/redeem"))
                .json(&RedeemRequest {
                    offer_id: offer.offer_id.clone(),
                    code: offer.code.clone(),
                    device_name: "Synthetic Phone".into(),
                    device_kind: ClientKind::Web,
                    public_key: b64.encode(key.verifying_key().to_sec1_point(false).as_bytes()),
                    scopes: Vec::new(),
                })
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(redeemed.fingerprint, offer.fingerprint);
        let mut device_id = None;
        for _ in 0..500 {
            let enrollment: Enrollment = http_json(
                http.get(format!("{base}/client/v1/enrollments/{}", offer.offer_id))
                    .send()
                    .await
                    .unwrap(),
            )
            .await;
            if enrollment.state == EnrollmentState::Approved {
                device_id = enrollment.device_id;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let device_id = device_id.expect("the owner confirmed in Telegram");
        assert!(telegram.cards.lock().unwrap()[0].contains("Synthetic Phone"));

        // Sign in: a challenge signed with the device key.
        let challenge: Challenge = http_json(
            http.post(format!("{base}/client/v1/sessions/challenge"))
                .json(&ChallengeRequest {
                    device_id: device_id.clone(),
                })
                .send()
                .await
                .unwrap(),
        )
        .await;
        let signature: p256::ecdsa::Signature =
            key.sign(challenge_message(&challenge.challenge_id, &challenge.nonce).as_bytes());
        let grant: AccessGrant = http_json(
            http.post(format!("{base}/client/v1/sessions"))
                .json(&SessionRequest {
                    challenge_id: challenge.challenge_id,
                    signature: b64.encode(signature.to_bytes()),
                })
                .send()
                .await
                .unwrap(),
        )
        .await;

        // The browser's socket: the token in `hello`, as a browser sends it.
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{}/client/v1/ws", daemon.url))
                .await
                .unwrap();
        let hello = ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: ClientKind::Web,
                name: "aidaemon-web".into(),
                version: None,
                device: None,
            },
            capabilities: Vec::new(),
            resume_from: None,
            token: Some(grant.access_token),
        };
        socket
            .send(Message::text(serde_json::to_string(&hello).unwrap()))
            .await
            .unwrap();
        let mut applied = Applied::default();
        until(&mut socket, &mut applied, |f| {
            matches!(f, ServerFrame::Snapshot { .. })
        })
        .await;

        // A turn started from Telegram asks to POST; the card is in
        // Telegram and in the browser.
        runner
            .submit(
                Arc::new(TelegramSurface),
                aidaemon_channels::turns::Turn {
                    session_id: TELEGRAM_SESSION.into(),
                    text: "Post the synthetic report for Alice Rivera".into(),
                    attachments: Vec::new(),
                    user_role: super::UserRole::Owner,
                    channel_ctx: super::ChannelContext::private("telegram"),
                    dedup_key: Some("tg-1".into()),
                    ingress_timing: None,
                },
            )
            .await;
        let asked = until(&mut socket, &mut applied, |f| {
            is_event(f, "approval_requested")
        })
        .await;
        let ServerFrame::Event {
            event: Event::ApprovalRequested { approval },
            ..
        } = asked
        else {
            unreachable!()
        };
        assert_eq!(approval.session_id, TELEGRAM_SESSION);
        assert_eq!(approval.channel.as_deref(), Some("telegram"));
        assert!(applied.events.iter().any(
            |e| matches!(e, Event::TurnStarted { origin, .. } if origin == "telegram")
        ));
        command(
            &mut socket,
            "approve-from-phone",
            Command::ResolveApproval {
                approval_id: approval.id.clone(),
                decision: ApprovalDecision::ApproveOnce,
            },
        )
        .await;
        let completed = until(&mut socket, &mut applied, |f| {
            is_event(f, "turn_completed")
        })
        .await;
        let ServerFrame::Event {
            event: Event::TurnCompleted { text, .. },
            ..
        } = completed
        else {
            unreachable!()
        };
        assert!(text.contains("Posted the synthetic report"), "{text}");
        assert_eq!(daemon.posted.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(applied.events.iter().any(|e| matches!(
            e,
            Event::ApprovalResolved {
                outcome: ApprovalOutcome::ApprovedOnce,
                resolved_by: Some(ResolvedBy { channel, device: Some(device) }),
                ..
            } if channel == "app" && device == "Synthetic Phone"
        )));
    }
}
