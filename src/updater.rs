//! Self-update system for aidaemon.
//!
//! Checks GitHub Releases for new versions and either auto-updates
//! or notifies the user for confirmation, depending on configuration.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tracing::{error, info, warn};

use crate::channels::ChannelHub;
use crate::config::{UpdateConfig, UpdateMode};
use crate::tools::ApprovalBroker;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::types::ApprovalResponse;

const REPO_OWNER: &str = "davo20019";
const REPO_NAME: &str = "aidaemon";
const BIN_NAME: &str = "aidaemon";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The project's release signing key (ed25519, raw 32 bytes). release.yml
/// signs every archive with its private half (the `RELEASE_SIGNING_KEY`
/// secret); scripts/release-signing-key.pub.pem and install.sh carry the
/// same key.
pub(crate) const RELEASE_SIGNING_PUBLIC_KEY: [u8; 32] = [
    0x99, 0x97, 0x3c, 0x5c, 0xdc, 0xb7, 0xc2, 0x82, 0x87, 0x93, 0xf1, 0xf2, 0x6b, 0x24, 0x99, 0xa0,
    0x29, 0xe7, 0xc4, 0xb8, 0xd3, 0xa1, 0xfa, 0x00, 0x4b, 0x60, 0x83, 0xef, 0x89, 0x1f, 0x0e, 0x31,
];

/// A release archive is a few tens of MB; anything far past that is not one.
const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;

/// Check `signature` (raw 64 bytes, or that base64-encoded on one line) over
/// `archive` with `key`.
pub(crate) fn verify_release_signature(
    archive: &[u8],
    signature: &[u8],
    key: &[u8; 32],
) -> Result<(), &'static str> {
    use base64::Engine as _;
    let raw: Vec<u8> = if signature.len() == 64 {
        signature.to_vec()
    } else {
        let text = std::str::from_utf8(signature).map_err(|_| "the signature is malformed")?;
        base64::engine::general_purpose::STANDARD
            .decode(text.trim())
            .map_err(|_| "the signature is malformed")?
    };
    let signature =
        ed25519_dalek::Signature::from_slice(&raw).map_err(|_| "the signature is malformed")?;
    let key =
        ed25519_dalek::VerifyingKey::from_bytes(key).map_err(|_| "the release key is invalid")?;
    key.verify_strict(archive, &signature)
        .map_err(|_| "the signature does not match the release key")
}

/// A daemon running as root would replace a binary that runs every later
/// start as root; updates there go through the package manager or a
/// reinstall instead.
fn refuse_as_root() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        // SAFETY: geteuid has no preconditions and cannot fail.
        if unsafe { libc::geteuid() } == 0 {
            anyhow::bail!(
                "AIDaemon is running as root: it does not update itself there. \
                 Reinstall it, or run it as your own user (`aidaemon service install`)."
            );
        }
    }
    Ok(())
}

fn download_asset(url: &str, dest: &std::path::Path, max_bytes: u64) -> anyhow::Result<()> {
    let file = std::fs::File::create(dest)?;
    self_update::Download::from_url(url)
        .show_download_progress(false)
        .max_download_size(max_bytes)
        .timeout(Duration::from_secs(300))
        .request_header(
            self_update::http::header::ACCEPT,
            "application/octet-stream",
        )
        .download_to(file)?;
    Ok(())
}

pub struct Updater {
    config: UpdateConfig,
    hub: Arc<ChannelHub>,
    notify_session_ids: Vec<String>,
    approval_tx: ApprovalBroker,
}

impl Updater {
    pub fn new(
        config: UpdateConfig,
        hub: Arc<ChannelHub>,
        notify_session_ids: Vec<String>,
        approval_tx: ApprovalBroker,
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

        if let Some(latest) = releases.latest() {
            let latest_version = latest.version().trim_start_matches('v');
            if let (Ok(current), Ok(latest_parsed)) = (
                semver::Version::parse(CURRENT_VERSION),
                semver::Version::parse(latest_version),
            ) && latest_parsed > current
            {
                let notes = latest.body().unwrap_or("No release notes.");
                return Ok(Some((latest_version.to_string(), notes.to_string())));
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

    /// Download the latest release, check its signature and replace the
    /// current binary.
    ///
    /// The archive is replaced only when its detached signature
    /// (`<archive>.sig`, ed25519 over the archive's bytes, made by
    /// release.yml with the project's release key) verifies against
    /// [`RELEASE_SIGNING_PUBLIC_KEY`]: a compromised release asset, mirror or
    /// repository token alone cannot push a binary. A release without a
    /// signature is refused, not installed.
    fn perform_update() -> anyhow::Result<String> {
        if crate::daemon::in_signed_app() {
            anyhow::bail!(
                "This AIDaemon came with the app download: get the new version of the app \
                 (its .dmg) to update both."
            );
        }
        refuse_as_root()?;
        let identifier = Self::platform_asset_identifier()?;
        let target_asset = format!("{}-{}.tar.gz", BIN_NAME, identifier);

        let releases = self_update::backends::github::ReleaseList::configure()
            .repo_owner(REPO_OWNER)
            .repo_name(REPO_NAME)
            .build()?
            .fetch()?;
        let latest = releases
            .latest()
            .ok_or_else(|| anyhow::anyhow!("No release found"))?;
        let version = latest.version().trim_start_matches('v').to_string();
        let is_newer = matches!(
            (semver::Version::parse(CURRENT_VERSION), semver::Version::parse(&version)),
            (Ok(current), Ok(latest)) if latest > current
        );
        if !is_newer {
            anyhow::bail!("v{version} is not newer than v{CURRENT_VERSION}");
        }
        let asset_url = |name: &str| {
            latest
                .assets()
                .iter()
                .find(|asset| asset.name() == name)
                .map(|asset| asset.download_url().to_string())
        };
        let archive_url = asset_url(&target_asset)
            .ok_or_else(|| anyhow::anyhow!("v{version} has no {target_asset}"))?;
        let signature_url = asset_url(&format!("{target_asset}.sig")).ok_or_else(|| {
            anyhow::anyhow!(
                "v{version} is not signed ({target_asset}.sig is missing): not installing it"
            )
        })?;

        let work = tempfile::tempdir()?;
        let archive_path = work.path().join(&target_asset);
        download_asset(&archive_url, &archive_path, MAX_ARCHIVE_BYTES)?;
        let signature_path = work.path().join("archive.sig");
        download_asset(&signature_url, &signature_path, 1024)?;
        let archive = std::fs::read(&archive_path)?;
        let signature = std::fs::read(&signature_path)?;
        verify_release_signature(&archive, &signature, &RELEASE_SIGNING_PUBLIC_KEY)
            .map_err(|why| anyhow::anyhow!("v{version}: {why}: not installing it"))?;
        info!(version = %version, "Release signature verified");

        let unpacked = work.path().join("unpacked");
        std::fs::create_dir(&unpacked)?;
        self_update::Extract::from_source(&archive_path)
            .archive(self_update::ArchiveKind::Tar(Some(
                self_update::Compression::Gz,
            )))
            .extract_into(&unpacked)?;
        let new_binary = unpacked.join(BIN_NAME);
        let metadata = std::fs::symlink_metadata(&new_binary)
            .map_err(|_| anyhow::anyhow!("v{version}'s archive has no {BIN_NAME}"))?;
        anyhow::ensure!(
            metadata.file_type().is_file(),
            "v{version}'s {BIN_NAME} is not a regular file"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&new_binary, std::fs::Permissions::from_mode(0o755))?;
        }
        self_replace::self_replace(&new_binary)?;
        Ok(version)
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
        let truncated_notes = crate::utils::truncate_str(release_notes, 503);

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
        let message = format!("aidaemon updated to v{}. Restarting now...", new_version);
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
                expires_at_ms: None,
                card: None,
                command: description,
                session_id,
                risk_level: RiskLevel::High,
                warnings: vec![
                    "The daemon will restart after updating.".to_string(),
                    "Active conversations will be interrupted briefly.".to_string(),
                ],
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: crate::types::ApprovalKind::CommandOnce,
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
            UpdateMode::Enable => match Self::perform_update_async().await {
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
            },
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
                                    &format!("Failed to apply update to v{}: {}", new_version, e),
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
        let today_target = now.date_naive().and_hms_opt(hour as u32, 0, 0).unwrap();
        let today_target_utc =
            chrono::DateTime::<Utc>::from_naive_utc_and_offset(today_target, Utc);
        let target = if now < today_target_utc {
            today_target_utc
        } else {
            today_target_utc + chrono::Duration::days(1)
        };
        (target - now).to_std().unwrap_or(Duration::from_secs(3600))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use ed25519_dalek::Signer as _;

    #[test]
    fn only_an_archive_signed_with_the_key_passes() {
        let signing = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        let key = signing.verifying_key().to_bytes();
        let archive = b"synthetic release archive";
        let signature = signing.sign(archive).to_bytes();
        assert!(verify_release_signature(archive, &signature, &key).is_ok());
        let as_text = base64::engine::general_purpose::STANDARD.encode(signature);
        assert!(verify_release_signature(archive, as_text.as_bytes(), &key).is_ok());
        assert!(verify_release_signature(b"tampered archive", &signature, &key).is_err());
        assert!(
            verify_release_signature(archive, &signature, &RELEASE_SIGNING_PUBLIC_KEY).is_err()
        );
        assert!(verify_release_signature(archive, b"", &key).is_err());
    }

    #[test]
    fn an_openssl_signature_from_the_release_key_verifies() {
        // Made with the release key the way release.yml signs an archive:
        // `openssl pkeyutl -sign -inkey <key> -rawin -in msg -out msg.sig`.
        let message = b"aidaemon release signing check\n";
        let signature = "vN7BTePMI2L67uZ11044+XevdrjkBa3ervwoL7HioewN5z6ZN1UCa3j5fj26Bml3XbCZBOFJNm8ogyex94yaBg==";
        assert!(
            verify_release_signature(message, signature.as_bytes(), &RELEASE_SIGNING_PUBLIC_KEY)
                .is_ok()
        );
    }

    #[test]
    fn the_pinned_key_is_the_one_in_the_repository() {
        let pem = include_str!("../scripts/release-signing-key.pub.pem");
        let body: String = pem
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let der = base64::engine::general_purpose::STANDARD
            .decode(body)
            .unwrap();
        assert_eq!(der[der.len() - 32..], RELEASE_SIGNING_PUBLIC_KEY);
    }
}
