//! The email trigger: an IMAP IDLE watch that wakes work on new mail.

use std::sync::Arc;

use async_imap::extensions::idle::IdleResponse;
use async_imap::imap_proto::Envelope;
use async_native_tls::TlsConnector;
use futures::TryStreamExt;
use tokio::net::TcpStream;
use tracing::{error, info, warn};

use super::{EventSender, TriggerEvent, TriggerSettings};

/// Starts the configured trigger sources; each sends its events on the bus.
pub struct TriggerManager {
    config: TriggerSettings,
    sender: EventSender,
}

impl TriggerManager {
    pub fn new(config: TriggerSettings, sender: EventSender) -> Self {
        Self { config, sender }
    }

    /// Spawn all configured triggers as background tasks on the current tokio
    /// runtime.
    pub fn spawn(self: Arc<Self>) {
        if let Some(ref email_config) = self.config.email {
            let host = email_config.host.clone();
            let port = email_config.port;
            let username = email_config.username.clone();
            let password = email_config.password.clone();
            let folder = email_config.folder.clone();
            let sender = self.sender.clone();

            tokio::spawn(async move {
                loop {
                    if let Err(e) =
                        imap_idle_loop(&host, port, &username, &password, &folder, &sender).await
                    {
                        error!("IMAP IDLE error: {}. Reconnecting in 30s...", e);
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    }
                }
            });

            info!("Email trigger spawned");
        }
    }
}

async fn imap_idle_loop(
    host: &str,
    port: u16,
    username: &str,
    password: &str,
    folder: &str,
    sender: &EventSender,
) -> anyhow::Result<()> {
    let tcp = TcpStream::connect((host, port)).await?;
    let tls = TlsConnector::new();
    let tls_stream = tls.connect(host, tcp).await?;

    let client = async_imap::Client::new(tls_stream);
    let mut session = client.login(username, password).await.map_err(|e| e.0)?;

    session.select(folder).await?;
    info!("IMAP connected to {}:{}/{}", host, port, folder);

    loop {
        let mut idle = session.idle();
        idle.init().await?;

        let (idle_wait, _interrupt) = idle.wait();
        let result = idle_wait.await?;

        match result {
            IdleResponse::NewData(_data) => {
                info!("New email detected");

                // Done with IDLE, get session back
                session = idle.done().await?;

                // Fetch the latest message
                let messages: Vec<_> = session.fetch("*", "ENVELOPE").await?.try_collect().await?;
                if let Some(msg) = messages.first() {
                    let envelope: Option<&Envelope> = msg.envelope();
                    let subject: String = envelope
                        .and_then(|e: &Envelope| e.subject.as_ref())
                        .map(|s| String::from_utf8_lossy(s).to_string())
                        .unwrap_or_else(|| "(no subject)".to_string());

                    let from: String = envelope
                        .and_then(|e: &Envelope| e.from.as_deref())
                        .and_then(|addrs: &[async_imap::imap_proto::Address]| addrs.first())
                        .map(|a| {
                            let mailbox = a
                                .mailbox
                                .as_ref()
                                .map(|m| String::from_utf8_lossy(m).to_string())
                                .unwrap_or_default();
                            let host_part = a
                                .host
                                .as_ref()
                                .map(|h| String::from_utf8_lossy(h).to_string())
                                .unwrap_or_default();
                            format!("{}@{}", mailbox, host_part)
                        })
                        .unwrap_or_else(|| "unknown".to_string());

                    let event = TriggerEvent {
                        source: "email".to_string(),
                        session_id: "email_trigger".to_string(),
                        content: format!("New email from {}: {}", from, subject),
                        trusted: false,
                        mandate_signal: None,
                    };

                    if sender.send(event).is_err() {
                        warn!("No event receivers active");
                    }
                }
            }
            _ => {
                session = idle.done().await?;
            }
        }
    }
}
