//! The setup gateway: the first device to redeem the printed code becomes
//! the owner device, the code is single use and dead once an owner exists,
//! and a signed-in client may only take setup steps.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_client_gateway::ClientGatewayConfig;
use aidaemon_client_gateway::setup::{SetupDeps, SetupGateway, SetupHost};
use aidaemon_client_protocol::{
    ClientKind, Command, CommandResult, ErrorCode, ServerFrame, SetupAction, SetupState, SetupStep,
};
use aidaemon_store_sqlite::SqliteClientEventLog;
use async_trait::async_trait;

struct Host;

#[async_trait]
impl SetupHost for Host {
    async fn state(&self, local: bool) -> SetupState {
        SetupState {
            on_server: !local,
            ..SetupState::default()
        }
    }
    async fn step(&self, action: SetupAction, _local: bool) -> Result<SetupState, String> {
        match action {
            SetupAction::Detect => Ok(SetupState {
                step: SetupStep::Starting,
                ..SetupState::default()
            }),
            _ => Err("not in this test".into()),
        }
    }
}

struct Fixture {
    base: String,
    token: String,
    gateway: Arc<SetupGateway>,
    _dir: tempfile::TempDir,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let log = SqliteClientEventLog::open(pool).await.unwrap();
    let token_file = dir.path().join("token.sha256");
    let token = aidaemon_client_gateway::token::mint(&token_file).unwrap();
    let (_tx, updates) = tokio::sync::watch::channel(SetupState::default());
    let gateway = SetupGateway::start(SetupDeps {
        host: Arc::new(Host),
        updates,
        log,
        config: ClientGatewayConfig::default(),
        token_file,
    })
    .await
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = aidaemon_client_gateway::setup::router(gateway.clone());
    tokio::spawn(async move { axum::serve(listener, router).await });
    Fixture {
        base,
        token,
        gateway,
        _dir: dir,
    }
}

async fn command(f: &Fixture, bearer: &str, command: Command) -> ServerFrame {
    reqwest::Client::new()
        .post(format!("{}/client/v1/commands", f.base))
        .bearer_auth(bearer)
        .json(&aidaemon_client_protocol::ClientFrame::Command {
            request_id: "r1".into(),
            command,
        })
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

fn pair_request(
    f: &Fixture,
    offer: &aidaemon_client_protocol::PairingOffer,
) -> aidaemon_client_core::signin::PairRequest {
    aidaemon_client_core::signin::PairRequest {
        gateway: f.base.clone(),
        offer_id: offer.offer_id.clone(),
        code: offer.code.clone(),
        device_name: "Synthetic Laptop".into(),
        device_kind: ClientKind::Desktop,
        fingerprint: Some(offer.fingerprint.clone()),
    }
}

#[tokio::test]
async fn the_printed_code_pairs_the_first_device_only_and_it_may_only_set_up() {
    let f = fixture().await;
    let offer = f.gateway.console_offer().await.expect("no owner yet");
    let http = reqwest::Client::new();
    let device = aidaemon_client_core::signin::pair(
        &http,
        &pair_request(&f, &offer),
        Duration::from_millis(10),
    )
    .await
    .expect("the first device pairs without a chat to confirm in");
    let grant = aidaemon_client_core::signin::sign_in(&http, &device)
        .await
        .unwrap();

    // Setup steps work; anything else waits for setup to finish.
    let ServerFrame::Ack {
        result: CommandResult::Setup { setup },
        ..
    } = command(&f, &grant.access_token, Command::GetSetup).await
    else {
        panic!("setup state");
    };
    assert!(setup.on_server, "a paired device is not this machine");
    let ServerFrame::Error { error, .. } =
        command(&f, &grant.access_token, Command::GetSnapshot).await
    else {
        panic!("refused");
    };
    assert_eq!(error.code, ErrorCode::Forbidden);

    // An owner device exists: no new code, and the old one is dead.
    assert!(f.gateway.console_offer().await.is_none());
    let again = aidaemon_client_core::signin::pair(
        &http,
        &pair_request(&f, &offer),
        Duration::from_millis(10),
    )
    .await;
    assert!(again.is_err(), "a second device does not pair");
}

#[tokio::test]
async fn signing_in_with_the_local_token_kills_the_printed_code() {
    let f = fixture().await;
    let offer = f.gateway.console_offer().await.unwrap();
    let ServerFrame::Ack {
        result: CommandResult::Setup { setup },
        ..
    } = command(&f, &f.token, Command::GetSetup).await
    else {
        panic!("setup state");
    };
    assert!(!setup.on_server);
    let paired = aidaemon_client_core::signin::pair(
        &reqwest::Client::new(),
        &pair_request(&f, &offer),
        Duration::from_millis(10),
    )
    .await;
    assert!(paired.is_err());
    assert!(matches!(
        command(&f, "acg_wrong", Command::GetSetup).await,
        ServerFrame::Error { .. }
    ));
}
