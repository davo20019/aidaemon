//! Self-update system for aidaemon.
//!
//! Checks GitHub Releases for new versions and either auto-updates
//! or notifies the user for confirmation, depending on configuration.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::channels::ChannelHub;
use crate::config::{UpdateConfig, UpdateMode};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::types::ApprovalResponse;

const REPO_OWNER: &str = "davo20019";
const REPO_NAME: &str = "aidaemon";
const BIN_NAME: &str = "aidaemon";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct Updater {
    config: UpdateConfig,
    hub: Arc<ChannelHub>,
    notify_session_ids: Vec<String>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

impl Updater {
    pub fn new(
        config: UpdateConfig,
        hub: Arc<ChannelHub>,
        notify_session_ids: Vec<String>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            config,
            hub,
            notify_session_ids,
            approval_tx,
        }
    }

    /// Map the current platform to the GitHub release asset name fragment.
    fn platform_asset_identifier() -> anyhow::Result<&'static str> {
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Ok("linux-x86_64")
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            Ok("linux-aarch64")
        } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
            Ok("macos-x86_64")
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Ok("macos-aarch64")
        } else {
            Err(anyhow::anyhow!("Unsupported platform for self-update"))
        }
    }

    /// Check GitHub Releases for a newer version.
    /// Returns `Some((new_version, release_notes))` if an update is available.
    pub fn check_for_update() -> anyhow::Result<Option<(String, String)>> {
        let releases = self_update::backends::github::ReleaseList::configure()
            .repo_owner(REPO_OWNER)
            .repo_name(REPO_NAME)
            .build()?
            .fetch()?;

        if let Some(latest) = releases.first() {
            let latest_version = latest.version.trim_start_matches('v');
            if let (Ok(current), Ok(latest_parsed)) = (
                semver::Version::parse(CURRENT_VERSION),
                semver::Version::parse(latest_version),
            ) {
                if latest_parsed > current {
                    let notes = latest.body.as_deref().unwrap_or("No release notes.");
                    return Ok(Some((latest_version.to_string(), notes.to_string())));
                }
            }
        }

        Ok(None)
    }

    /// Async wrapper for version check (runs blocking GitHub API call off-thread).
    async fn check_for_update_async() -> anyhow::Result<Option<(String, String)>> {
        tokio::task::spawn_blocking(Self::check_for_update)
            .await
            .map_err(|e| anyhow::anyhow!("Update check task panicked: {}", e))?
    }

    /// Download the latest release and replace the current binary.
    fn perform_update() -> anyhow::Result<String> {
        let identifier = Self::platform_asset_identifier()?;
        let target_asset = format!("{}-{}.tar.gz", BIN_NAME, identifier);

        let status = self_update::backends::github::Update::configure()
            .repo_owner(REPO_OWNER)
            .repo_name(REPO_NAME)
            .bin_name(BIN_NAME)
            .identifier(&target_asset)
            .current_version(CURRENT_VERSION)
            .no_confirm(true)
            .show_download_progress(false)
            .build()?
            .update()?;

        Ok(status.version().to_string())
    }

    /// Async wrapper for the blocking update operation.
    async fn perform_update_async() -> anyhow::Result<String> {
        tokio::task::spawn_blocking(Self::perform_update)
            .await
            .map_err(|e| anyhow::anyhow!("Update task panicked: {}", e))?
    }

    /// Restart the daemon by exiting the process.
    /// The service manager (launchd/systemd) will restart with the new binary.
    /// Exit code 75 (EX_TEMPFAIL) triggers restart under both systemd
    /// (Restart=on-failure) and launchd (KeepAlive=true).
    fn restart_service() -> ! {
        info!("Restarting aidaemon to apply update...");
        std::thread::sleep(Duration::from_secs(1));
        std::process::exit(75)
    }

    /// Notify users that an update is available.
    async fn notify_update_available(&self, new_version: &str, release_notes: &str) {
        let truncated_notes = if release_notes.len() > 500 {
            format!("{}...", &release_notes[..500])
        } else {
            release_notes.to_string()
        };

        let action_line = match self.config.mode {
            UpdateMode::Enable => "Update will be applied automatically.",
            UpdateMode::CheckOnly => "Reply to approve the update, or it will be skipped.",
            UpdateMode::Disable => return,
        };

        let message = format!(
            "**aidaemon update available: v{} → v{}**\n\n{}\n\n{}",
            CURRENT_VERSION, new_version, truncated_notes, action_line
        );

        self.hub
            .broadcast_text(&self.notify_session_ids, &message)
            .await;
    }

    /// Notify users that an update was applied and restart is imminent.
    async fn notify_update_applied(&self, new_version: &str) {
        let message = format!(
            "aidaemon updated to v{}. Restarting now...",
            new_version
        );
        self.hub
            .broadcast_text(&self.notify_session_ids, &message)
            .await;
    }

    /// Request user approval for applying an update.
    /// Returns `true` if the user approved, `false` if denied or timed out.
    async fn request_update_approval(&self, new_version: &str) -> bool {
        let session_id = match self.notify_session_ids.first() {
            Some(id) => id.clone(),
            None => {
                warn!("No session IDs configured for update approval");
                return false;
            }
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let description = format!(
            "Update aidaemon from v{} to v{}",
            CURRENT_VERSION, new_version
        );

        let send_result = self
            .approval_tx
            .send(ApprovalRequest {
                command: description,
                session_id,
                risk_level: RiskLevel::High,
                warnings: vec![
                    "The daemon will restart after updating.".to_string(),
                    "Active conversations will be interrupted briefly.".to_string(),
                ],
                permission_mode: PermissionMode::Default,
                response_tx,
            })
            .await;

        if send_result.is_err() {
            warn!("Failed to send update approval request");
            return false;
        }

        let timeout = Duration::from_secs(self.config.confirmation_timeout_mins * 60);
        match tokio::time::timeout(timeout, response_rx).await {
            Ok(Ok(ApprovalResponse::AllowOnce))
            | Ok(Ok(ApprovalResponse::AllowSession))
            | Ok(Ok(ApprovalResponse::AllowAlways)) => {
                info!("User approved update to v{}", new_version);
                true
            }
            Ok(Ok(ApprovalResponse::Deny)) => {
                info!("User denied update to v{}", new_version);
                false
            }
            Ok(Err(_)) => {
                warn!("Update approval channel dropped");
                false
            }
            Err(_) => {
                info!(
                    "Update approval timed out after {} minutes",
                    self.config.confirmation_timeout_mins
                );
                self.hub
                    .broadcast_text(
                        &self.notify_session_ids,
                        &format!(
                            "Update to v{} skipped (no response within {} minutes).",
                            new_version, self.config.confirmation_timeout_mins
                        ),
                    )
                    .await;
                false
            }
        }
    }

    /// Perform a single update check cycle.
    async fn tick(&self) {
        info!("Checking for aidaemon updates...");

        let (new_version, release_notes) = match Self::check_for_update_async().await {
            Ok(Some(info)) => info,
            Ok(None) => {
                info!("aidaemon is up to date (v{})", CURRENT_VERSION);
                return;
            }
            Err(e) => {
                warn!("Update check failed: {}", e);
                return;
            }
        };

        info!(
            current = CURRENT_VERSION,
            available = %new_version,
            "New aidaemon version available"
        );

        self.notify_update_available(&new_version, &release_notes)
            .await;

        match self.config.mode {
            UpdateMode::Enable => {
                match Self::perform_update_async().await {
                    Ok(version) => {
                        self.notify_update_applied(&version).await;
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        Self::restart_service();
                    }
                    Err(e) => {
                        error!("Failed to apply update: {}", e);
                        self.hub
                            .broadcast_text(
                                &self.notify_session_ids,
                                &format!("Failed to apply update to v{}: {}", new_version, e),
                            )
                            .await;
                    }
                }
            }
            UpdateMode::CheckOnly => {
                if self.request_update_approval(&new_version).await {
                    match Self::perform_update_async().await {
                        Ok(version) => {
                            self.notify_update_applied(&version).await;
                            tokio::time::sleep(Duration::from_secs(2)).await;
                            Self::restart_service();
                        }
                        Err(e) => {
                            error!("Failed to apply update: {}", e);
                            self.hub
                                .broadcast_text(
                                    &self.notify_session_ids,
                                    &format!(
                                        "Failed to apply update to v{}: {}",
                                        new_version, e
                                    ),
                                )
                                .await;
                        }
                    }
                }
            }
            UpdateMode::Disable => unreachable!(),
        }
    }

    /// Spawn the background update check loop.
    pub fn spawn(self: Arc<Self>) {
        if self.config.mode == UpdateMode::Disable {
            info!("Self-update system disabled");
            return;
        }

        info!(mode = ?self.config.mode, "Self-update system started");

        tokio::spawn(async move {
            // Initial delay: wait 60 seconds after startup before first check.
            // Gives channels time to connect so notifications can be delivered.
            tokio::time::sleep(Duration::from_secs(60)).await;

            loop {
                self.tick().await;

                let sleep_duration = if let Some(hour) = self.config.check_at_utc_hour {
                    Self::duration_until_utc_hour(hour)
                } else {
                    Duration::from_secs(self.config.check_interval_hours * 3600)
                };

                tokio::time::sleep(sleep_duration).await;
            }
        });
    }

    /// Calculate how long to sleep until the next occurrence of a given UTC hour.
    fn duration_until_utc_hour(hour: u8) -> Duration {
        let now = Utc::now();
        let today_target = now
            .date_naive()
            .and_hms_opt(hour as u32, 0, 0)
            .unwrap();
        let today_target_utc =
            chrono::DateTime::<Utc>::from_naive_utc_and_offset(today_target, Utc);
        let target = if now < today_target_utc {
            today_target_utc
        } else {
            today_target_utc + chrono::Duration::days(1)
        };
        (target - now)
            .to_std()
            .unwrap_or(Duration::from_secs(3600))
    }
}
