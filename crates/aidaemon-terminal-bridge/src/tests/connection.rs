//! Device ids, broker tokens, the outbound queue, reconnects and relay ciphers.

use super::*;

#[test]
fn default_device_id_is_stable_per_key() {
    let fingerprint = "0123456789abcdef0123456789abcdef";
    let id = default_device_id(fingerprint);
    // A restart keeps the id, so the broker lets it replace the old socket.
    assert_eq!(id, default_device_id(fingerprint));
    assert!(id.ends_with("-0123456789ab"), "{id}");
    assert_eq!(id, sanitize_device_id(&id));
    // A different install (different key) gets a different id.
    assert_ne!(id, default_device_id("fedcba9876543210fedcba9876543210"));
}

#[test]
fn test_daemon_connect_token_mint_url_from_wss() {
    let url =
        daemon_connect_token_mint_url("wss://terminal.aidaemon.ai/v1/ws/daemon").expect("mint url");
    assert_eq!(
        url,
        "https://terminal.aidaemon.ai/v1/daemon/connect-token/daemon-auth"
    );
}

#[test]
fn test_daemon_connect_token_mint_url_rejects_insecure_ws() {
    assert!(daemon_connect_token_mint_url("ws://terminal.aidaemon.ai/v1/ws/daemon").is_err());
}

#[test]
fn test_merge_daemon_bot_tokens_includes_dynamic_telegram_bots_for_active_user() {
    let dynamic_bots = vec![
        RuntimeBot {
            channel_type: "telegram".to_string(),
            bot_token: "dynamic-user-token".to_string(),
            allowed_user_ids: vec!["555000111".to_string()],
        },
        RuntimeBot {
            channel_type: "telegram".to_string(),
            bot_token: "other-user-token".to_string(),
            allowed_user_ids: vec!["999".to_string()],
        },
        RuntimeBot {
            channel_type: "slack".to_string(),
            bot_token: "xoxb-not-telegram".to_string(),
            allowed_user_ids: vec!["555000111".to_string()],
        },
    ];

    let merged = merge_daemon_bot_tokens(
        &[
            "configured-token".to_string(),
            "dynamic-user-token".to_string(),
        ],
        &dynamic_bots,
        "555000111",
    );

    assert_eq!(
        merged,
        vec![
            "configured-token".to_string(),
            "dynamic-user-token".to_string()
        ]
    );
}

#[test]
fn test_outbound_queue_prioritizes_high_frames() {
    let mut queue = OutboundQueue::default();
    let _ = queue.enqueue(
        OutboundPriority::Low,
        QueuedOutboundPayload {
            session_id: "session-low-1".to_string(),
            payload: json!({"kind":"stdout","seq":1}),
        },
    );
    let _ = queue.enqueue(
        OutboundPriority::High,
        QueuedOutboundPayload {
            session_id: "session-high".to_string(),
            payload: json!({"kind":"review_result"}),
        },
    );
    let _ = queue.enqueue(
        OutboundPriority::Low,
        QueuedOutboundPayload {
            session_id: "session-low-2".to_string(),
            payload: json!({"kind":"stdout","seq":2}),
        },
    );

    let first = queue.pop_next().expect("first");
    let second = queue.pop_next().expect("second");
    let third = queue.pop_next().expect("third");

    assert_eq!(first.session_id, "session-high");
    assert_eq!(second.session_id, "session-low-1");
    assert_eq!(third.session_id, "session-low-2");
    assert!(queue.pop_next().is_none());
}

#[test]
fn test_outbound_queue_bounds_low_lane() {
    let mut queue = OutboundQueue::default();
    for idx in 0..=OUTBOUND_LOW_QUEUE_CAP {
        let _ = queue.enqueue(
            OutboundPriority::Low,
            QueuedOutboundPayload {
                session_id: format!("s{idx}"),
                payload: json!({"kind":"stdout","seq": idx}),
            },
        );
    }

    assert_eq!(queue.low.len(), OUTBOUND_LOW_QUEUE_CAP);
    let first = queue.pop_next().expect("first");
    let last = queue.low.back().expect("last");
    assert_eq!(first.session_id, "s1");
    assert_eq!(last.session_id, format!("s{}", OUTBOUND_LOW_QUEUE_CAP));
}

#[test]
fn test_next_reconnect_backoff_ms_doubles_until_cap() {
    assert_eq!(
        TerminalBridge::next_reconnect_backoff_ms(RECONNECT_INITIAL_MS),
        RECONNECT_INITIAL_MS * 2
    );
    assert_eq!(
        TerminalBridge::next_reconnect_backoff_ms(RECONNECT_MAX_MS / 2),
        RECONNECT_MAX_MS
    );
    assert_eq!(
        TerminalBridge::next_reconnect_backoff_ms(RECONNECT_MAX_MS),
        RECONNECT_MAX_MS
    );
}

#[test]
fn test_next_reconnect_backoff_ms_normalizes_zero_input() {
    assert_eq!(
        TerminalBridge::next_reconnect_backoff_ms(0),
        RECONNECT_INITIAL_MS
    );
}

#[test]
fn test_is_expected_disconnect_classifies_routine_resets() {
    use tokio_tungstenite::tungstenite::Error as WsError;
    use tokio_tungstenite::tungstenite::error::ProtocolError;

    // Routine disconnects that the reconnect loop recovers from automatically.
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::Protocol(ProtocolError::ResetWithoutClosingHandshake)
    )));
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::ConnectionClosed
    )));
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::AlreadyClosed
    )));
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::Io(std::io::ErrorKind::ConnectionReset.into())
    )));
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::Io(std::io::ErrorKind::BrokenPipe.into())
    )));
    assert!(TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::Io(std::io::ErrorKind::UnexpectedEof.into())
    )));

    // Genuine failures must remain ERROR-worthy.
    assert!(!TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        "authentication failed: 401 unauthorized"
    )));
    assert!(!TerminalBridge::is_expected_disconnect(&anyhow::anyhow!(
        WsError::Io(std::io::ErrorKind::PermissionDenied.into())
    )));
}

#[test]
fn test_daemon_bootstrap_signing_input() {
    let input = daemon_bootstrap_signing_input("12345", "macbook-pro", 1700000000, "abcd1234");
    assert_eq!(
        input,
        "v1\nuser_id=12345\ndevice_id=macbook-pro\nts=1700000000\nnonce=abcd1234"
    );
}

#[test]
fn test_daemon_bootstrap_signature_candidates_include_hkdf_and_legacy() {
    let signatures = daemon_bootstrap_signature_candidates(
        "123456:telegram-bot-token",
        "12345",
        "macbook-pro",
        1700000000,
        "abcd1234",
    )
    .expect("signature candidates");
    assert_eq!(signatures.len(), 2);
    assert_ne!(signatures[0], signatures[1]);
}

#[tokio::test]
async fn test_mint_connect_token_falls_back_to_legacy_signature() {
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct TestState {
        bot_token: String,
        calls: Arc<AtomicUsize>,
    }

    #[derive(Deserialize)]
    struct MintReq {
        user_id: String,
        device_id: String,
        ts: i64,
        nonce: String,
        sig: String,
    }

    #[derive(Serialize)]
    struct MintResp {
        ok: bool,
        daemon_connect_token: Option<String>,
        error: Option<String>,
        message: Option<String>,
    }

    async fn handler(
        State(state): State<TestState>,
        Json(req): Json<MintReq>,
    ) -> (StatusCode, Json<MintResp>) {
        state.calls.fetch_add(1, Ordering::SeqCst);
        let primary = sign_daemon_bootstrap_proof_hkdf(
            &state.bot_token,
            &req.user_id,
            &req.device_id,
            req.ts,
            &req.nonce,
        )
        .expect("primary signature");
        let legacy = sign_daemon_bootstrap_proof_legacy(
            &state.bot_token,
            &req.user_id,
            &req.device_id,
            req.ts,
            &req.nonce,
        )
        .expect("legacy signature");

        if req.sig == primary {
            return (
                StatusCode::UNAUTHORIZED,
                Json(MintResp {
                    ok: false,
                    daemon_connect_token: None,
                    error: Some("unauthorized".to_string()),
                    message: Some("primary rejected for compatibility test".to_string()),
                }),
            );
        }
        if req.sig == legacy {
            return (
                StatusCode::OK,
                Json(MintResp {
                    ok: true,
                    daemon_connect_token: Some("minted-token".to_string()),
                    error: None,
                    message: None,
                }),
            );
        }
        (
            StatusCode::BAD_REQUEST,
            Json(MintResp {
                ok: false,
                daemon_connect_token: None,
                error: Some("bad_signature".to_string()),
                message: Some("signature did not match expected candidates".to_string()),
            }),
        )
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let state = TestState {
        bot_token: "123456:telegram-bot-token".to_string(),
        calls: calls.clone(),
    };
    let app = Router::new().route("/", post(handler)).with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("http client");
    let token = mint_connect_token_from_bot_proof(
        &client,
        &format!("http://{}/", addr),
        &["123456:telegram-bot-token".to_string()],
        "12345",
        "macbook-pro",
    )
    .await
    .expect("minted token");

    assert_eq!(token, "minted-token");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    server.abort();
}

#[test]
fn test_derive_relay_cipher_round_trip() {
    let daemon_secret = SecretKey::generate_from_rng(&mut UnwrapErr(SysRng));
    let client_secret = SecretKey::generate_from_rng(&mut UnwrapErr(SysRng));
    let relay_session_id = "relay-session-1";

    let client_pub = client_secret.public_key().to_sec1_point(false);
    let client_pub_b64 = b64_encode(client_pub.as_bytes());
    let daemon_cipher =
        TerminalBridge::derive_relay_cipher(&daemon_secret, relay_session_id, &client_pub_b64)
            .expect("daemon cipher");

    let daemon_pub = daemon_secret.public_key();
    let shared = diffie_hellman(client_secret.to_nonzero_scalar(), daemon_pub.as_affine());
    let mut salt_hash = Sha256::new();
    salt_hash.update(relay_session_id.as_bytes());
    let salt = salt_hash.finalize();
    #[allow(deprecated)]
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
    let mut key = [0u8; 32];
    hk.expand(KEY_INFO, &mut key)
        .expect("hkdf expansion should work");
    let client_cipher = Aes256Gcm::new_from_slice(&key).expect("aes cipher");

    let nonce = [7u8; 12];
    let aad = b"relay-session-1:42";
    let plaintext = b"terminal bridge test payload";
    let ciphertext = daemon_cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .expect("encrypt");
    let decrypted = client_cipher
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &ciphertext,
                aad,
            },
        )
        .expect("decrypt");
    assert_eq!(decrypted, plaintext);
}

#[test]
fn test_derive_relay_cipher_session_id_mismatch_breaks_decryption() {
    let daemon_secret = SecretKey::generate_from_rng(&mut UnwrapErr(SysRng));
    let client_secret = SecretKey::generate_from_rng(&mut UnwrapErr(SysRng));

    let client_pub = client_secret.public_key().to_sec1_point(false);
    let client_pub_b64 = b64_encode(client_pub.as_bytes());
    let daemon_cipher = TerminalBridge::derive_relay_cipher(
        &daemon_secret,
        "relay-session-correct",
        &client_pub_b64,
    )
    .expect("daemon cipher");

    let daemon_pub = daemon_secret.public_key();
    let shared = diffie_hellman(client_secret.to_nonzero_scalar(), daemon_pub.as_affine());
    let mut salt_hash = Sha256::new();
    salt_hash.update("relay-session-wrong".as_bytes());
    let salt = salt_hash.finalize();
    #[allow(deprecated)]
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
    let mut key = [0u8; 32];
    hk.expand(KEY_INFO, &mut key)
        .expect("hkdf expansion should work");
    let wrong_client_cipher = Aes256Gcm::new_from_slice(&key).expect("aes cipher");

    let nonce = [3u8; 12];
    let aad = b"relay-session-correct:1";
    let plaintext = b"hello";
    let ciphertext = daemon_cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .expect("encrypt");
    let decrypted = wrong_client_cipher.decrypt(
        Nonce::from_slice(&nonce),
        Payload {
            msg: &ciphertext,
            aad,
        },
    );
    assert!(decrypted.is_err());
}
