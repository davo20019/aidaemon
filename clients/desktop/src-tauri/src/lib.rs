//! The AIdaemon desktop app: a Tauri 2 shell around its own copy of the web
//! client (`clients/web`).
//!
//! The shell holds the only connection to the daemon (the shared client core,
//! as `ClientKind::Desktop`); the page gets frames and sends commands through
//! the shell's commands below, so the gateway needs no cross-origin
//! allowance and the device key never enters the web view. The shell also
//! does what a tab cannot: a tray icon with the connection state and the
//! approvals waiting, a native notification for each new approval (the
//! window may be closed: closing it only hides it), the device key in the OS
//! keychain, and starting the local daemon through its service manager when
//! it is not running. Quitting the app leaves the daemon running.

mod daemon;
mod keychain;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_client_core::keystore::KeyStore;
use aidaemon_client_core::link::{self, Auth, ClientIdentity, DropKind, LinkConfig, Update};
use aidaemon_client_core::signin::{self, PairRequest};
use aidaemon_client_core::State;
use aidaemon_client_protocol::{
    Capability, ClientKind, Command, OfferRequest, PairingOffer, ServerFrame, Snapshot,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, WindowEvent};
use tauri_plugin_notification::NotificationExt;
use tokio::sync::mpsc;

use crate::keychain::KeychainStore;

const FRAME_EVENT: &str = "aidaemon://frame";
const LINK_EVENT: &str = "aidaemon://link";
const TRAY_ID: &str = "main";

/// What the shell keeps for the page and the tray.
struct Shell {
    store: Arc<KeychainStore>,
    commands: Mutex<Option<mpsc::UnboundedSender<(String, Command)>>>,
    /// The shell's own view, from the same reducer the TUI uses: what the
    /// page gets when it (re)opens, and the count in the tray.
    state: Mutex<State>,
    /// The link's state as the page's `LinkState` reads it.
    link: Mutex<Value>,
    waiting_item: Mutex<Option<MenuItem<tauri::Wry>>>,
}

impl Shell {
    fn new() -> Self {
        Self {
            store: Arc::new(KeychainStore),
            commands: Mutex::new(None),
            state: Mutex::new(State::default()),
            link: Mutex::new(json!({ "phase": "connecting" })),
            waiting_item: Mutex::new(None),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}

/// The link's state in the page's terms (`LinkState` in `link.ts`).
fn link_state(update: &Update) -> Option<Value> {
    Some(match update {
        Update::Connecting => json!({ "phase": "connecting" }),
        Update::Connected => json!({ "phase": "connected" }),
        Update::Disconnected(d) => {
            let kind = match d.kind {
                DropKind::DaemonNotRunning => "daemon_not_running",
                DropKind::Unauthorized => "unauthorized",
                DropKind::VersionUnsupported => "version_unsupported",
                DropKind::Dropped => "dropped",
            };
            json!({
                "phase": "reconnecting",
                "kind": kind,
                "reason": d.reason,
                "retryInMs": u64::try_from(d.retry_in.as_millis()).unwrap_or(u64::MAX),
            })
        }
        Update::Frame(_) => return None,
    })
}

/// Start the connection for the paired device, if there is one.
fn start_link(app: &AppHandle) {
    let shell = app.state::<Arc<Shell>>().inner().clone();
    let device = match shell.store.load() {
        Ok(Some(device)) => device,
        _ => return,
    };
    let url = format!(
        "{}/client/v1/ws",
        device
            .gateway
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1)
    );
    let config = LinkConfig {
        url,
        auth: Auth::Device(shell.store.clone()),
        client: ClientIdentity {
            kind: ClientKind::Desktop,
            name: "aidaemon-desktop".into(),
            version: Some(env!("CARGO_PKG_VERSION").into()),
        },
        device: None,
        capabilities: vec![Capability::ViewActions],
    };
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let (updates_tx, mut updates) = mpsc::unbounded_channel();
        let commands = link::spawn(config, updates_tx);
        *lock(&shell.commands) = Some(commands.clone());
        let mut notified: HashSet<String> = HashSet::new();
        while let Some(update) = updates.recv().await {
            if let Some(state) = link_state(&update) {
                *lock(&shell.link) = state.clone();
                let _ = app.emit(LINK_EVENT, state);
            }
            match update {
                Update::Connected => {
                    let _ = commands.send((
                        link::request_id(),
                        Command::ListSessions {
                            cursor: None,
                            limit: 200,
                        },
                    ));
                }
                Update::Frame(frame) => {
                    let new_approvals: Vec<(String, String)> = {
                        let mut state = lock(&shell.state);
                        state.apply(frame.clone());
                        state
                            .approvals
                            .values()
                            .filter(|a| !notified.contains(&a.id))
                            .map(|a| (a.id.clone(), a.summary.clone()))
                            .collect()
                    };
                    for (id, summary) in new_approvals {
                        notified.insert(id);
                        notify_approval(&app, &summary);
                    }
                    let _ = app.emit(FRAME_EVENT, &frame);
                }
                _ => {}
            }
            refresh_tray(&app, &shell);
        }
    });
}

/// A native notification for a new approval, shown with the window closed.
fn notify_approval(app: &AppHandle, summary: &str) {
    let _ = app
        .notification()
        .builder()
        .title("Approval needed")
        .body(summary)
        .show();
}

/// The tray's tooltip and count follow the connection and the approvals.
fn refresh_tray(app: &AppHandle, shell: &Shell) {
    let waiting = lock(&shell.state).approvals.len();
    let phase = lock(&shell.link)["phase"]
        .as_str()
        .unwrap_or("connecting")
        .to_string();
    let connection = match phase.as_str() {
        "connected" => "Connected".to_string(),
        "connecting" => "Connecting...".to_string(),
        _ => match lock(&shell.link)["kind"].as_str() {
            Some("daemon_not_running") => "Daemon not running".to_string(),
            Some("unauthorized") => "Signed out".to_string(),
            _ => "Reconnecting...".to_string(),
        },
    };
    let count = match waiting {
        0 => "Nothing waiting".to_string(),
        1 => "1 approval waiting".to_string(),
        n => format!("{n} approvals waiting"),
    };
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_tooltip(Some(format!("AIdaemon: {connection} · {count}")));
        #[cfg(target_os = "macos")]
        let _ = tray.set_title(if waiting > 0 {
            Some(waiting.to_string())
        } else {
            None
        });
    }
    if let Some(item) = lock(&shell.waiting_item).as_ref() {
        let _ = item.set_text(format!("{connection} · {count}"));
    }
}

fn show_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

// --- Commands the page calls ---------------------------------------------------

#[tauri::command]
fn is_paired(shell: tauri::State<'_, Arc<Shell>>) -> bool {
    shell.store.load().ok().flatten().is_some()
}

/// The page's `PairRequest` (platform.ts).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PagePairRequest {
    offer_id: String,
    code: String,
    device_name: String,
    fingerprint: Option<String>,
    gateway: Option<String>,
}

#[tauri::command]
async fn pair(
    app: AppHandle,
    shell: tauri::State<'_, Arc<Shell>>,
    request: PagePairRequest,
) -> Result<(), String> {
    let gateway = request
        .gateway
        .filter(|g| !g.trim().is_empty())
        .map(|g| g.trim().trim_end_matches('/').to_string())
        .unwrap_or_else(daemon::local_gateway);
    let device = signin::pair(
        &reqwest::Client::new(),
        &PairRequest {
            gateway,
            offer_id: request.offer_id,
            code: request.code,
            device_name: request.device_name,
            device_kind: ClientKind::Desktop,
            fingerprint: request.fingerprint,
        },
        Duration::from_millis(1500),
    )
    .await
    .map_err(|e| e.to_string())?;
    shell.store.save(&device).map_err(|e| e.to_string())?;
    start_link(&app);
    Ok(())
}

/// First run on the daemon's own machine: pair with the local token, the
/// way `aidaemon tui` signs in, without a code. `false` when there is no
/// local daemon or token to pair with.
async fn pair_locally(app: &AppHandle) -> bool {
    let Some(token) = daemon::local_token() else {
        return false;
    };
    let gateway = daemon::local_gateway();
    let http = reqwest::Client::new();
    let offer: PairingOffer = match http
        .post(format!("{gateway}/client/v1/pairing/offers"))
        .bearer_auth(&token)
        .json(&OfferRequest::default())
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => match response.json().await {
            Ok(offer) => offer,
            Err(_) => return false,
        },
        _ => return false,
    };
    let device_name = format!(
        "Desktop app on {}",
        std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "this computer".into())
    );
    let request = PairRequest {
        gateway: gateway.clone(),
        offer_id: offer.offer_id.clone(),
        code: offer.code.clone(),
        device_name,
        device_kind: ClientKind::Desktop,
        fingerprint: Some(offer.fingerprint.clone()),
    };
    let pairing = tauri::async_runtime::spawn(async move {
        signin::pair(&reqwest::Client::new(), &request, Duration::from_millis(250)).await
    });
    // The owner is the one holding the local token: confirm at once.
    for _ in 0..40 {
        let decided = http
            .post(format!(
                "{gateway}/client/v1/pairing/offers/{}/decision",
                offer.offer_id
            ))
            .bearer_auth(&token)
            .json(&json!({ "approve": true }))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success());
        if decided {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let Ok(Ok(device)) = pairing.await else {
        return false;
    };
    let shell = app.state::<Arc<Shell>>();
    shell.store.save(&device).is_ok()
}

#[tauri::command]
fn unpair(shell: tauri::State<'_, Arc<Shell>>) -> Result<(), String> {
    *lock(&shell.commands) = None;
    *lock(&shell.state) = State::default();
    shell.store.clear().map_err(|e| e.to_string())
}

/// The page (re)opened: give it what the shell knows, then the live frames.
#[tauri::command]
fn link_start(app: AppHandle, shell: tauri::State<'_, Arc<Shell>>) -> Result<(), String> {
    let (seq, snapshot) = {
        let state = lock(&shell.state);
        let snapshot = Snapshot {
            sessions: state
                .order
                .iter()
                .filter_map(|id| state.sessions.get(id)?.summary.clone())
                .collect(),
            running_tasks: state.running().into_iter().cloned().collect(),
            pending_approvals: state.approvals.values().cloned().collect(),
        };
        (state.last_seq.unwrap_or(0), snapshot)
    };
    let link = lock(&shell.link).clone();
    let _ = app.emit(LINK_EVENT, link);
    let _ = app.emit(
        FRAME_EVENT,
        &ServerFrame::Snapshot {
            seq,
            state: snapshot,
        },
    );
    Ok(())
}

#[tauri::command]
fn link_send(
    shell: tauri::State<'_, Arc<Shell>>,
    request_id: String,
    command: Command,
) -> Result<(), String> {
    match lock(&shell.commands).as_ref() {
        Some(commands) => commands
            .send((request_id, command))
            .map_err(|_| "the connection has stopped".to_string()),
        None => Err("not paired".into()),
    }
}

/// Start (or restart) the local daemon through its service manager.
#[tauri::command]
fn start_daemon() -> Result<String, String> {
    daemon::start()
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .manage(Arc::new(Shell::new()))
        .invoke_handler(tauri::generate_handler![
            is_paired,
            pair,
            unpair,
            link_start,
            link_send,
            start_daemon
        ])
        .setup(|app| {
            let open = MenuItem::with_id(app, "open", "Open AIdaemon", true, None::<&str>)?;
            let waiting =
                MenuItem::with_id(app, "waiting", "Connecting...", false, None::<&str>)?;
            let start = MenuItem::with_id(
                app,
                "start",
                "Start the daemon if it is not running",
                true,
                None::<&str>,
            )?;
            let quit = MenuItem::with_id(
                app,
                "quit",
                "Quit (the daemon keeps running)",
                true,
                None::<&str>,
            )?;
            let menu = Menu::with_items(app, &[&open, &waiting, &start, &quit])?;
            *lock(&app.state::<Arc<Shell>>().waiting_item) = Some(waiting);
            let mut tray = TrayIconBuilder::with_id(TRAY_ID)
                .tooltip("AIdaemon")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_window(app),
                    "start" => {
                        let _ = daemon::start();
                    }
                    "quit" => app.exit(0),
                    _ => {}
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let paired = handle
                    .state::<Arc<Shell>>()
                    .store
                    .load()
                    .ok()
                    .flatten()
                    .is_some();
                if paired || pair_locally(&handle).await {
                    start_link(&handle);
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window hides it: the tray and notifications go on.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("the AIdaemon desktop app failed to start");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_states_read_as_the_page_expects() {
        assert_eq!(
            link_state(&Update::Connected),
            Some(json!({ "phase": "connected" }))
        );
        let down = link_state(&Update::Disconnected(aidaemon_client_core::Disconnect {
            kind: DropKind::DaemonNotRunning,
            reason: "nothing is listening".into(),
            retry_in: Duration::from_secs(2),
        }))
        .unwrap();
        assert_eq!(down["kind"], "daemon_not_running");
        assert_eq!(down["retryInMs"], 2000);
    }
}
