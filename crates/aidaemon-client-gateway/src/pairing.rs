//! Pairing devices and their short-lived access tokens.
//!
//! The owner makes a single-use offer (a code and a link, valid ten
//! minutes); a device redeems it with its P-256 public key; the owner
//! confirms on a chat they already use (an approval card through the hub)
//! or, for an offer made at the terminal, there. Each connection proves the
//! key by signing a challenge and gets a 15-minute access token carrying
//! the device's scopes and risk ceiling. Over a peer-to-peer transport a
//! socket's upgrade may carry the proof instead ([`Pairing::sign_in_socket`]):
//! the device signs its own time and a random value, bound to this daemon
//! and its endpoint key, and the daemon takes each such proof once. Revoking a device ends its tokens
//! and its open sockets at once.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::hub::ApprovalRequest;
use aidaemon_client_protocol::{
    ALL_SCOPES, AccessGrant, Challenge, ClientKind, DeviceInfo, Enrollment, EnrollmentState,
    ErrorCode, PairingOffer, ProtocolError, RedeemRequest, Risk, SOCKET_SIGN_IN_MAX_AGE_MS,
    SOCKET_SIGN_IN_MAX_AHEAD_MS, Scope, SocketSignIn, challenge_message, challenge_message_v2,
    is_loopback_url, sign_in_audience, socket_sign_in_message,
};
use aidaemon_store_sqlite::{DeviceRow, PairingRow, SqliteClientEventLog};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature, VerifyingKey};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::bus::now_ms;

const OFFER_TTL: Duration = Duration::from_secs(10 * 60);
const CHALLENGE_TTL: Duration = Duration::from_secs(60);
pub const ACCESS_TTL: Duration = Duration::from_secs(15 * 60);
/// Wrong codes an offer takes before it closes.
const MAX_ATTEMPTS: u32 = 5;
/// The code's letters: no 0/O, 1/I/L.
const CODE_ALPHABET: &[u8] = b"23456789ABCDEFGHJKMNPQRSTUVWXYZ";
const ACCESS_PREFIX: &str = "acs_";
/// Unanswered challenges held at once.
const MAX_OPEN_CHALLENGES: usize = 1024;
/// One device signs in a few sockets at once at most.
const MAX_OPEN_CHALLENGES_PER_DEVICE: usize = 8;
/// Socket sign-ins remembered at once (each until it could no longer be
/// fresh), so none is taken twice.
const MAX_SOCKET_SIGN_INS: usize = 4096;
/// One device's socket sign-ins within that time: a page reconnecting now
/// and then, never this many.
const MAX_SOCKET_SIGN_INS_PER_DEVICE: usize = 32;

fn ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

use crate::util::sha;

fn random_hex() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Who a request comes from.
#[derive(Debug, Clone, PartialEq)]
pub struct Principal {
    /// `None`: the owner's local token.
    pub device_id: Option<String>,
    pub device_name: Option<String>,
    pub scopes: Vec<Scope>,
    pub ceiling: Option<Risk>,
    /// Revoking the device cancels this.
    pub revoked: CancellationToken,
}

impl Principal {
    pub fn owner() -> Self {
        Self {
            device_id: None,
            device_name: None,
            scopes: ALL_SCOPES.to_vec(),
            ceiling: None,
            revoked: CancellationToken::new(),
        }
    }

    /// The owner at the machine: may pair and revoke devices.
    pub fn is_owner(&self) -> bool {
        self.device_id.is_none()
    }

    pub fn may(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }
}

struct Access {
    principal: Principal,
    expires_at_ms: u64,
}

struct PendingChallenge {
    device_id: String,
    nonce: String,
    expires_at_ms: u64,
    /// The endpoint key the device says it dials with, recorded once the
    /// signature verifies.
    endpoint: Option<String>,
    /// That endpoint as the request named it: a daemon-bound signature
    /// ([`challenge_message_v2`]) covers it.
    named_endpoint: Option<String>,
    /// When it was made: the oldest goes first when a device has too many.
    created_ms: u64,
}

/// Who the peer-to-peer listener lets past its handshake.
#[derive(Default)]
struct Admission {
    /// Endpoint keys (hex) of active devices.
    known: HashSet<String>,
    /// Any key until then: an offer is open, or a device just paired and
    /// has not signed in yet.
    open_until_ms: u64,
    /// Devices paired before endpoint keys were recorded that have not
    /// signed in since: any key, until each has (or is revoked).
    unrecorded: usize,
}

impl Admission {
    fn admits(&self, endpoint: &str, now_ms: u64) -> bool {
        self.known.contains(endpoint) || now_ms < self.open_until_ms || self.unrecorded > 0
    }
}

/// Whether `signature` (base64url, `r || s` or DER) is `device`'s key's
/// signature of `message`.
fn verify(device: &DeviceRow, message: &str, signature: &str) -> Result<(), ProtocolError> {
    let refused = || ProtocolError::new(ErrorCode::Unauthorized, "The signature does not verify");
    let key = URL_SAFE_NO_PAD
        .decode(&device.public_key)
        .ok()
        .and_then(|bytes| VerifyingKey::from_sec1_bytes(&bytes).ok())
        .ok_or_else(refused)?;
    let bytes = URL_SAFE_NO_PAD
        .decode(signature.trim())
        .map_err(|_| refused())?;
    let signature = if bytes.len() == 64 {
        Signature::from_slice(&bytes).map_err(|_| refused())?
    } else {
        Signature::from_der(&bytes).map_err(|_| refused())?
    };
    key.verify(message.as_bytes(), &signature)
        .map_err(|_| refused())
}

/// The daemon's own endpoint key in its peer-to-peer address
/// (`iroh://<key>@<relay>`; a machine format of our own).
fn own_key(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    endpoint_of(rest.split(['@', '/']).next()?)
}

/// Drop the oldest of `challenges` that `which` picks.
fn evict_oldest(
    challenges: &mut HashMap<String, PendingChallenge>,
    which: impl Fn(&PendingChallenge) -> bool,
) {
    if let Some(oldest) = challenges
        .iter()
        .filter(|(_, c)| which(c))
        .min_by_key(|(_, c)| c.created_ms)
        .map(|(id, _)| id.clone())
    {
        challenges.remove(&oldest);
    }
}

/// An endpoint key as sign-in names it: 64 lowercase hex digits.
fn endpoint_of(text: &str) -> Option<String> {
    let text = text.trim().to_ascii_lowercase();
    (text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit())).then_some(text)
}

pub struct Pairing {
    log: SqliteClientEventLog,
    /// Where confirmation cards go (the hub's approval queue).
    approvals: Option<mpsc::Sender<ApprovalRequest>>,
    /// What the offer link names as the gateway, read per offer.
    public_url: crate::PublicUrl,
    /// The daemon's peer-to-peer address (`iroh://…`), put in offer links
    /// beside `public_url` once remote access is up.
    remote_url: Mutex<Option<String>>,
    /// The web app on a site of its own, named in offers while there is a
    /// remote address.
    web_app: Mutex<Option<String>>,
    /// Offers shown on the owner's own screen ([`Self::offer_shown_here`]):
    /// redeeming one pairs at once. Kept in memory only: after a restart
    /// such an offer asks like any other.
    shown_here: Mutex<HashSet<String>>,
    fingerprint: String,
    access: Mutex<HashMap<String, Access>>,
    challenges: Mutex<HashMap<String, PendingChallenge>>,
    /// Socket sign-ins taken, by device and random value, each until it is
    /// too old to be taken anyway (its time plus the window). Kept in
    /// memory: after a restart [`Self::started_ms`] stands in for it.
    socket_sign_ins: Mutex<HashMap<(String, String), u64>>,
    /// When this gateway started: a socket sign-in made before it may have
    /// been taken by the run before, whose memory is gone.
    started_ms: u64,
    /// Per device, what revoking it cancels.
    revocations: Mutex<HashMap<String, CancellationToken>>,
    /// The message files whose links a revoked device may still hold:
    /// revoking one signs links with a new key ([`crate::files::FileStore::rotate_key`]).
    files: Mutex<Option<Arc<crate::files::FileStore>>>,
    /// The ceiling a watch gets unless the owner sets another.
    watch_ceiling: Risk,
    admission: Mutex<Admission>,
}

fn scopes_text(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .filter_map(|scope| serde_json::to_value(scope).ok())
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect::<Vec<_>>()
        .join(",")
}

fn scopes_of(text: &str) -> Vec<Scope> {
    text.split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| serde_json::from_value(serde_json::Value::String(s.to_string())).ok())
        .filter(|s| *s != Scope::Unknown)
        .collect()
}

/// What a device gets when it asks for nothing in particular: everything,
/// but watching and controlling surfaces only on a phone or a computer.
pub(crate) fn default_scopes(kind: Option<ClientKind>) -> Vec<Scope> {
    let gets_control = kind.is_some_and(|kind| {
        matches!(
            kind,
            ClientKind::Mobile | ClientKind::Desktop | ClientKind::Web | ClientKind::Tui
        )
    });
    ALL_SCOPES
        .iter()
        .copied()
        .filter(|scope| *scope != Scope::Control || gets_control)
        .collect()
}

fn kind_text(kind: ClientKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".into())
}

fn kind_of(text: &str) -> ClientKind {
    serde_json::from_value(serde_json::Value::String(text.to_string()))
        .unwrap_or(ClientKind::Unknown)
}

fn risk_of(text: &str) -> Option<Risk> {
    serde_json::from_value(serde_json::Value::String(text.to_string())).ok()
}

fn state_of(status: &str) -> EnrollmentState {
    match status {
        "open" => EnrollmentState::Open,
        "pending" => EnrollmentState::Pending,
        "approved" => EnrollmentState::Approved,
        "denied" => EnrollmentState::Denied,
        "expired" => EnrollmentState::Expired,
        _ => EnrollmentState::Unknown,
    }
}

fn not_found(what: &str) -> ProtocolError {
    ProtocolError::new(ErrorCode::NotFound, what)
}

fn internal(error: impl std::fmt::Display) -> ProtocolError {
    warn!(%error, "Pairing failed");
    ProtocolError::new(
        ErrorCode::Internal,
        "The daemon could not complete the pairing step",
    )
}

/// A pairing code: eight letters, shown as `XXXX-XXXX`.
fn new_code() -> String {
    // Two random bytes per letter, so the modulo bias is negligible, taken
    // from positions that hold none of a UUID's fixed version/variant bits.
    const RANDOM_BYTES: [usize; 8] = [0, 1, 2, 3, 4, 5, 7, 9];
    let a = uuid::Uuid::new_v4().into_bytes();
    let b = uuid::Uuid::new_v4().into_bytes();
    let letters: String = RANDOM_BYTES
        .iter()
        .map(|&i| {
            let n = usize::from(a[i]) << 8 | usize::from(b[i]);
            CODE_ALPHABET[n % CODE_ALPHABET.len()] as char
        })
        .collect();
    format!("{}-{}", &letters[..4], &letters[4..])
}

/// The code as typed: case and dashes do not matter.
fn normalize_code(code: &str) -> String {
    code.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

fn percent(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

impl Pairing {
    pub(crate) fn new(
        log: SqliteClientEventLog,
        approvals: Option<mpsc::Sender<ApprovalRequest>>,
        public_url: crate::PublicUrl,
        watch_ceiling: Risk,
        fingerprint: String,
    ) -> Self {
        // The fingerprint names this install ([`crate::token::install_fingerprint`]):
        // random, so no other daemon can give it back to a device.
        Self {
            log,
            approvals,
            public_url,
            remote_url: Mutex::default(),
            web_app: Mutex::default(),
            shown_here: Mutex::default(),
            fingerprint,
            access: Mutex::default(),
            challenges: Mutex::default(),
            socket_sign_ins: Mutex::default(),
            started_ms: now_ms(),
            revocations: Mutex::default(),
            files: Mutex::default(),
            watch_ceiling,
            admission: Mutex::default(),
        }
    }

    /// Read who the peer-to-peer listener admits from the database: the
    /// endpoint keys of active devices, offers still open, and devices
    /// paired before keys were recorded.
    pub async fn load_admission(&self) -> Result<(), ProtocolError> {
        let now = now_ms();
        let known = self.log.active_device_endpoints().await.map_err(internal)?;
        let open_until = self.log.open_offers_until(now).await.map_err(internal)?;
        let since = self
            .log
            .endpoint_admission_since(now)
            .await
            .map_err(internal)?;
        // A browser never dials a peer-to-peer address, so it is never
        // waited for.
        let unrecorded = self
            .log
            .unrecorded_devices(since, &kind_text(ClientKind::Web))
            .await
            .map_err(internal)?;
        if !unrecorded.is_empty() {
            let names: Vec<&str> = unrecorded.iter().map(|(_, name)| name.as_str()).collect();
            warn!(
                devices = %names.join(", "),
                "Remote access admits any key until these devices sign in once with an \
                 updated app (or are revoked)"
            );
        }
        let mut admission = self.admission.lock().unwrap_or_else(|p| p.into_inner());
        admission.known = known.into_iter().collect();
        admission.open_until_ms = admission.open_until_ms.max(open_until.unwrap_or(0));
        admission.unrecorded = unrecorded.len();
        Ok(())
    }

    /// Whether the peer-to-peer listener lets endpoint key `endpoint`
    /// (hex) through its handshake. The gateway's sign-in still decides
    /// every request of a key let in.
    pub fn admits(&self, endpoint: &str) -> bool {
        self.admission
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .admits(endpoint, now_ms())
    }

    /// Any key until `until_ms`: a device is pairing.
    fn open_admission_until(&self, until_ms: u64) {
        let mut admission = self.admission.lock().unwrap_or_else(|p| p.into_inner());
        admission.open_until_ms = admission.open_until_ms.max(until_ms);
    }

    /// Admit `endpoint` as `device_id`'s from now on.
    pub(crate) async fn record_endpoint(&self, device_id: &str, endpoint: &str) {
        let Some(endpoint) = endpoint_of(endpoint) else {
            return;
        };
        let known = self
            .admission
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .known
            .contains(&endpoint);
        if known {
            return;
        }
        if let Err(error) = self
            .log
            .add_device_endpoint(device_id, &endpoint, now_ms())
            .await
        {
            warn!(%error, "Could not record a device's endpoint key");
            return;
        }
        if let Err(error) = self.load_admission().await {
            warn!(error = %error.message, "Could not reload remote admission");
        }
    }

    /// Offer links from now on carry `url` (`iroh://…`) as `remote`: a
    /// device that can dial it pairs and signs in over it. `None` stops.
    pub fn set_remote_url(&self, url: Option<String>) {
        *self.remote_url.lock().unwrap_or_else(|p| p.into_inner()) = url;
    }

    /// Offers name `url` as the web app a phone with no app pairs at
    /// ([`PairingOffer::phone_link`]) while there is a remote address.
    pub fn set_web_app(&self, url: Option<String>) {
        *self.web_app.lock().unwrap_or_else(|p| p.into_inner()) =
            url.filter(|u| !u.trim().is_empty());
    }

    /// This daemon's peer-to-peer key (64 lowercase hex digits), once
    /// remote access is up.
    pub(crate) fn own_remote_key(&self) -> Option<String> {
        self.remote_url().and_then(|url| own_key(&url))
    }

    fn remote_url(&self) -> Option<String> {
        self.remote_url
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn code_hash(offer_id: &str, code: &str) -> String {
        sha(&format!("{offer_id}:{}", normalize_code(code)))
    }

    /// A single-use offer; confirmed in `confirm_in_session` when given,
    /// else by the owner's decision call.
    pub async fn offer(
        &self,
        confirm_in_session: Option<String>,
    ) -> Result<PairingOffer, ProtocolError> {
        let offer_id = format!("of-{}", &random_hex()[..20]);
        let code = new_code();
        let now = now_ms();
        let expires_at_ms = now + ms(OFFER_TTL);
        self.open_admission_until(expires_at_ms);
        self.log
            .create_offer(&PairingRow {
                offer_id: offer_id.clone(),
                code_hash: Self::code_hash(&offer_id, &code),
                origin_session: confirm_in_session,
                created_at_ms: now,
                expires_at_ms,
                attempts: 0,
                status: "open".into(),
                device_name: None,
                device_kind: None,
                public_key: None,
                scopes: None,
                device_id: None,
            })
            .await
            .map_err(internal)?;
        let public_url = self.public_url.get();
        let mut link = format!(
            "aidaemon-pair:?gateway={}&offer={offer_id}&code={code}&fp={}",
            percent(&public_url),
            self.fingerprint
        );
        let remote = self.remote_url();
        if let Some(remote) = &remote {
            link.push_str(&format!("&remote={}", percent(remote)));
        }
        // The web app can only reach the daemon by key.
        let web_app = remote.and(
            self.web_app
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
        );
        Ok(PairingOffer {
            offer_id,
            code,
            expires_at_ms,
            link,
            fingerprint: self.fingerprint.clone(),
            public_url: Some(public_url),
            web_app,
        })
    }

    /// An offer for the owner's own screen (the desktop app's QR code,
    /// `aidaemon pair`): whoever redeems it is paired at once, since only
    /// the owner at this computer can show it and the code never leaves the
    /// screen. Offers sent through a chat ask there instead.
    pub async fn offer_shown_here(&self) -> Result<PairingOffer, ProtocolError> {
        let offer = self.offer(None).await?;
        self.shown_here
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(offer.offer_id.clone());
        Ok(offer)
    }

    /// A device redeems an offer. The owner is then asked (unless the offer
    /// was shown on their screen); the device polls [`Self::enrollment`].
    pub async fn redeem(
        self: &std::sync::Arc<Self>,
        request: RedeemRequest,
    ) -> Result<Enrollment, ProtocolError> {
        let key = URL_SAFE_NO_PAD
            .decode(request.public_key.trim())
            .ok()
            .and_then(|bytes| VerifyingKey::from_sec1_bytes(&bytes).ok());
        if key.is_none() {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "public_key must be a P-256 SEC1 key, base64url",
            ));
        }
        let name: String = request.device_name.trim().chars().take(60).collect();
        if name.is_empty() {
            return Err(ProtocolError::new(ErrorCode::Invalid, "Name the device"));
        }
        let scopes = if request.scopes.is_empty() {
            default_scopes(Some(request.device_kind))
        } else {
            request.scopes.clone()
        };
        let redeemed = self
            .log
            .redeem_offer(
                &request.offer_id,
                &Self::code_hash(&request.offer_id, &request.code),
                now_ms(),
                MAX_ATTEMPTS,
                &name,
                &kind_text(request.device_kind),
                request.public_key.trim(),
                &scopes_text(&scopes),
            )
            .await
            .map_err(internal)?;
        if !redeemed {
            return Err(ProtocolError::new(
                ErrorCode::Unauthorized,
                "That code does not match an open pairing offer",
            ));
        }
        let row = self
            .log
            .offer(&request.offer_id)
            .await
            .map_err(internal)?
            .ok_or_else(|| not_found("No such offer"))?;
        let shown_here = self
            .shown_here
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&request.offer_id);
        if shown_here {
            info!(offer_id = %request.offer_id, "Paired from a code shown on the owner's screen");
            return self.decide(&request.offer_id, true).await;
        }
        if let Some(session) = row.origin_session.clone() {
            let pairing = self.clone();
            tokio::spawn(async move { pairing.confirm_in_chat(row, session).await });
        }
        self.enrollment(&request.offer_id).await
    }

    /// Ask the owner, on the chat the offer was made in, whether to pair.
    async fn confirm_in_chat(&self, row: PairingRow, session: String) {
        let Some(approvals) = self.approvals.clone() else {
            return;
        };
        let (response_tx, response_rx) = oneshot::channel();
        let name = row.device_name.clone().unwrap_or_default();
        let kind = row.device_kind.clone().unwrap_or_default();
        let scopes = scopes_of(row.scopes.as_deref().unwrap_or_default());
        let mut can = Vec::new();
        if scopes.contains(&Scope::Observe) {
            can.push("read your chats");
        }
        if scopes.contains(&Scope::Chat) {
            can.push("send messages and stop work");
        }
        if scopes.contains(&Scope::Approve) {
            can.push("answer approvals");
        }
        if scopes.contains(&Scope::Control) {
            can.push("watch and use the agent's browser");
        }
        let request = ApprovalRequest {
            command: format!(
                "Pair a new device: “{name}” ({kind}). It will be able to {}. Only allow it if \
                 you just started pairing it.",
                can.join(", ")
            ),
            session_id: session,
            risk_level: RiskLevel::High,
            warnings: Vec::new(),
            permission_mode: PermissionMode::Default,
            response_tx,
            kind: ApprovalKind::CommandOnce,
            card: None,
            expires_at_ms: Some(row.expires_at_ms),
        };
        let approve = if approvals.send(request).await.is_ok() {
            let wait = Duration::from_millis(row.expires_at_ms.saturating_sub(now_ms()));
            matches!(
                tokio::time::timeout(wait, response_rx).await,
                Ok(Ok(ApprovalResponse::AllowOnce
                    | ApprovalResponse::AllowSession
                    | ApprovalResponse::AllowAlways))
            )
        } else {
            false
        };
        if let Err(error) = self.decide(&row.offer_id, approve).await {
            warn!(?error, "Could not settle a pairing");
        }
    }

    /// Settle a pending pairing: approved pairs the device.
    pub async fn decide(&self, offer_id: &str, approve: bool) -> Result<Enrollment, ProtocolError> {
        let row = self
            .log
            .offer(offer_id)
            .await
            .map_err(internal)?
            .ok_or_else(|| not_found("No such offer"))?;
        if row.status != "pending" {
            return Err(ProtocolError::new(
                ErrorCode::Conflict,
                format!("This pairing is {}, not waiting for an answer", row.status),
            ));
        }
        if approve {
            let device_id = format!("dv-{}", &random_hex()[..20]);
            let kind = kind_of(row.device_kind.as_deref().unwrap_or_default());
            let ceiling = (kind == ClientKind::Watch).then_some(self.watch_ceiling);
            // The device exists before the offer reads approved, so a
            // client that sees approved can sign in at once.
            self.log
                .add_device(&DeviceRow {
                    device_id: device_id.clone(),
                    name: row.device_name.clone().unwrap_or_default(),
                    kind: kind_text(kind),
                    public_key: row.public_key.clone().unwrap_or_default(),
                    scopes: row.scopes.clone().unwrap_or_default(),
                    risk_ceiling: ceiling.map(|risk| {
                        serde_json::to_value(risk)
                            .ok()
                            .and_then(|v| v.as_str().map(str::to_string))
                            .unwrap_or_default()
                    }),
                    created_at_ms: now_ms(),
                    last_seen_ms: None,
                    revoked_at_ms: None,
                })
                .await
                .map_err(internal)?;
            if !self
                .log
                .decide_offer(offer_id, "approved", Some(&device_id))
                .await
                .map_err(internal)?
            {
                self.log
                    .revoke_device(&device_id, now_ms())
                    .await
                    .map_err(internal)?;
                return Err(ProtocolError::new(ErrorCode::Conflict, "Already settled"));
            }
            // The new device signs in next, from a key not known yet.
            self.open_admission_until(now_ms() + ms(OFFER_TTL));
            info!(device_id, "Paired a client device");
        } else {
            self.log
                .decide_offer(offer_id, "denied", None)
                .await
                .map_err(internal)?;
        }
        self.enrollment(offer_id).await
    }

    pub async fn enrollment(&self, offer_id: &str) -> Result<Enrollment, ProtocolError> {
        let row = self
            .log
            .offer(offer_id)
            .await
            .map_err(internal)?
            .ok_or_else(|| not_found("No such offer"))?;
        let mut state = state_of(&row.status);
        if matches!(state, EnrollmentState::Open | EnrollmentState::Pending)
            && row.expires_at_ms <= now_ms()
        {
            state = EnrollmentState::Expired;
        }
        Ok(Enrollment {
            offer_id: row.offer_id,
            state,
            device_id: row.device_id,
            device_name: row.device_name,
            device_kind: row.device_kind.as_deref().map(kind_of),
            scopes: scopes_of(row.scopes.as_deref().unwrap_or_default()),
            fingerprint: self.fingerprint.clone(),
        })
    }

    pub async fn challenge(
        &self,
        device_id: &str,
        endpoint: Option<&str>,
    ) -> Result<Challenge, ProtocolError> {
        let device = self.active_device(device_id).await?;
        let challenge_id = format!("ch-{}", &random_hex()[..20]);
        let nonce = random_hex();
        let expires_at_ms = now_ms() + ms(CHALLENGE_TTL);
        let mut challenges = self.challenges.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        challenges.retain(|_, c| c.expires_at_ms > now);
        // Per device too: one device (or anyone who learned its id) asking
        // over and over must not use up everyone's sign-ins. Its oldest
        // challenge makes way, so asking with someone else's id never
        // locks that device out: its own sign-in, the newest, still works.
        let own = challenges
            .values()
            .filter(|c| c.device_id == device.device_id)
            .count();
        if own >= MAX_OPEN_CHALLENGES_PER_DEVICE {
            evict_oldest(&mut challenges, |c| c.device_id == device.device_id);
        } else if challenges.len() >= MAX_OPEN_CHALLENGES {
            evict_oldest(&mut challenges, |_| true);
        }
        let named_endpoint = endpoint.map(str::trim).filter(|e| !e.is_empty());
        challenges.insert(
            challenge_id.clone(),
            PendingChallenge {
                device_id: device.device_id,
                nonce: nonce.clone(),
                expires_at_ms,
                endpoint: named_endpoint.and_then(endpoint_of),
                named_endpoint: named_endpoint.map(str::to_string),
                created_ms: now,
            },
        );
        Ok(Challenge {
            challenge_id,
            nonce,
            expires_at_ms,
        })
    }

    async fn active_device(&self, device_id: &str) -> Result<DeviceRow, ProtocolError> {
        match self.log.device(device_id).await.map_err(internal)? {
            Some(device) if device.revoked_at_ms.is_none() => Ok(device),
            _ => Err(ProtocolError::new(
                ErrorCode::Unauthorized,
                "This device is not paired (or was revoked)",
            )),
        }
    }

    /// Whether `audience` ([`sign_in_audience`]) names this daemon: its
    /// peer-to-peer key, an origin it is published at, or (`local`, for a
    /// request made on this computer, and only then) a loopback address: a
    /// loopback address names whatever listens there on the device's own
    /// computer, not this daemon.
    fn is_own_audience(&self, audience: &str, local: bool) -> bool {
        if let Some(key) = self.remote_url().and_then(|url| own_key(&url))
            && audience == format!("iroh:{key}")
        {
            return true;
        }
        if audience.starts_with("http") && is_loopback_url(audience) {
            return local;
        }
        let mut origins = self.public_url.endpoints.current();
        origins.push(self.public_url.fallback.clone());
        origins
            .iter()
            .filter_map(|origin| sign_in_audience(origin))
            .any(|origin| origin == audience)
    }

    /// Trade a signed challenge for an access token. `audience`: the daemon
    /// the signature names ([`challenge_message_v2`]); `None` is the older
    /// [`challenge_message`], bound to no daemon, taken only when `local`
    /// (the request came on this computer, not through a tunnel or a
    /// peer-to-peer transport), so a daemon that relays this one's
    /// challenge to a device never gets a signature it can use here.
    pub async fn open_session(
        &self,
        challenge_id: &str,
        signature: &str,
        audience: Option<&str>,
        local: bool,
    ) -> Result<AccessGrant, ProtocolError> {
        let challenge = self
            .challenges
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(challenge_id)
            .filter(|c| c.expires_at_ms > now_ms())
            .ok_or_else(|| {
                ProtocolError::new(ErrorCode::Unauthorized, "Unknown or expired challenge")
            })?;
        let device = self.active_device(&challenge.device_id).await?;
        let message = match audience.map(str::trim) {
            Some(audience) => {
                if !self.is_own_audience(audience, local) {
                    return Err(ProtocolError::new(
                        ErrorCode::Unauthorized,
                        "This sign-in was signed for another daemon",
                    ));
                }
                challenge_message_v2(
                    challenge_id,
                    &challenge.nonce,
                    audience,
                    challenge.named_endpoint.as_deref(),
                )
            }
            None if local => challenge_message(challenge_id, &challenge.nonce),
            None => {
                // Which device, so its owner can find it and update its app.
                tracing::warn!(
                    device = %device.device_id,
                    name = %device.name,
                    "A device signs in remotely with an app too old to bind its sign-in"
                );
                return Err(ProtocolError::new(
                    ErrorCode::Unauthorized,
                    "Update the app: signing in from another computer needs a sign-in \
                     bound to this daemon",
                ));
            }
        };
        verify(&device, &message, signature)?;
        let grant = self.grant(&device).await;
        if let Some(endpoint) = &challenge.endpoint {
            self.record_endpoint(&device.device_id, endpoint).await;
        }
        Ok(grant)
    }

    /// Sign a device in from the proof its socket upgrade carried, over a
    /// peer-to-peer transport whose handshake proved `endpoint` (hex): the
    /// same token a signed challenge gets, in one round trip. Refused
    /// unless the device is active and `endpoint` is one it signed in from
    /// before (so a first sign-in takes a challenge); the proof's time is
    /// within the window and not before this gateway started; it signs
    /// [`socket_sign_in_message`] for this daemon's own key and `endpoint`;
    /// and its random value was not taken before.
    pub async fn sign_in_socket(
        &self,
        proof: &SocketSignIn,
        endpoint: &str,
    ) -> Result<AccessGrant, ProtocolError> {
        let refused = |why: &str| ProtocolError::new(ErrorCode::Unauthorized, why);
        let endpoint =
            endpoint_of(endpoint).ok_or_else(|| refused("This transport names no endpoint key"))?;
        // Its own key, as its peer-to-peer address names it: without one
        // (remote access not up yet) there is nothing to bind the proof to.
        let daemon = self
            .remote_url()
            .and_then(|url| own_key(&url))
            .ok_or_else(|| refused("This daemon has no peer-to-peer address"))?;
        let nonce = proof.nonce.trim();
        if !(32..=64).contains(&nonce.len()) || !nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(refused("The sign-in's random value is malformed"));
        }
        let now = now_ms();
        if proof.issued_at_ms.saturating_add(SOCKET_SIGN_IN_MAX_AGE_MS) < now
            || proof.issued_at_ms > now.saturating_add(SOCKET_SIGN_IN_MAX_AHEAD_MS)
            || proof.issued_at_ms < self.started_ms
        {
            return Err(refused(
                "The sign-in is not fresh; sign in with a challenge",
            ));
        }
        let device = self.active_device(&proof.device_id).await?;
        if !self
            .log
            .device_has_endpoint(&device.device_id, &endpoint)
            .await
            .map_err(internal)?
        {
            return Err(refused(
                "This endpoint key is not this device's yet; sign in with a challenge",
            ));
        }
        verify(
            &device,
            &socket_sign_in_message(
                &daemon,
                &endpoint,
                &device.device_id,
                proof.issued_at_ms,
                nonce,
            ),
            &proof.signature,
        )?;
        // Taken once: checked and recorded under one lock, after the
        // signature, so nobody without the device's key fills this memory.
        {
            let mut taken = self
                .socket_sign_ins
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            taken.retain(|_, until| *until > now);
            let slot = (device.device_id.clone(), nonce.to_ascii_lowercase());
            if taken.contains_key(&slot) {
                return Err(refused("This sign-in was used already"));
            }
            let own = taken.keys().filter(|(d, _)| *d == device.device_id).count();
            if own >= MAX_SOCKET_SIGN_INS_PER_DEVICE || taken.len() >= MAX_SOCKET_SIGN_INS {
                return Err(ProtocolError::new(
                    ErrorCode::RateLimited,
                    "Too many sign-ins at once; try again in a minute",
                ));
            }
            // Past this it is refused as stale anyway.
            let until = proof
                .issued_at_ms
                .saturating_add(SOCKET_SIGN_IN_MAX_AGE_MS)
                .max(now)
                .saturating_add(1);
            taken.insert(slot, until);
        }
        Ok(self.grant(&device).await)
    }

    /// A new access token for `device`, with its scopes and ceiling.
    async fn grant(&self, device: &DeviceRow) -> AccessGrant {
        let scopes = scopes_of(&device.scopes);
        let ceiling = device.risk_ceiling.as_deref().and_then(risk_of);
        let revoked = self
            .revocations
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(device.device_id.clone())
            .or_default()
            .clone();
        let token = format!("{ACCESS_PREFIX}{}", random_hex());
        let expires_at_ms = now_ms() + ms(ACCESS_TTL);
        let principal = Principal {
            device_id: Some(device.device_id.clone()),
            device_name: Some(device.name.clone()),
            scopes: scopes.clone(),
            ceiling,
            revoked,
        };
        {
            let mut access = self.access.lock().unwrap_or_else(|p| p.into_inner());
            let now = now_ms();
            access.retain(|_, a| a.expires_at_ms > now);
            access.insert(
                sha(&token),
                Access {
                    principal,
                    expires_at_ms,
                },
            );
        }
        let _ = self.log.touch_device(&device.device_id, now_ms()).await;
        AccessGrant {
            access_token: token,
            expires_at_ms,
            device_id: device.device_id.clone(),
            scopes,
            risk_ceiling: ceiling,
        }
    }

    /// The device behind an access token, while it is valid.
    pub fn principal(&self, token: &str) -> Option<Principal> {
        if !token.starts_with(ACCESS_PREFIX) {
            return None;
        }
        let access = self.access.lock().unwrap_or_else(|p| p.into_inner());
        let found = access.get(&sha(token))?;
        (found.expires_at_ms > now_ms() && !found.principal.revoked.is_cancelled())
            .then(|| found.principal.clone())
    }

    pub async fn devices(&self) -> Result<Vec<DeviceInfo>, ProtocolError> {
        Ok(self
            .log
            .devices()
            .await
            .map_err(internal)?
            .into_iter()
            .map(|device| DeviceInfo {
                scopes: scopes_of(&device.scopes),
                kind: kind_of(&device.kind),
                risk_ceiling: device.risk_ceiling.as_deref().and_then(risk_of),
                revoked: device.revoked_at_ms.is_some(),
                device_id: device.device_id,
                name: device.name,
                created_at_ms: device.created_at_ms,
                last_seen_ms: device.last_seen_ms,
            })
            .collect())
    }

    /// Change what a device may do. Its live access tokens change with it;
    /// taking a scope away does not wait for them to expire.
    pub async fn set_scopes(
        &self,
        device_id: &str,
        scopes: &[Scope],
    ) -> Result<bool, ProtocolError> {
        let scopes: Vec<Scope> = scopes
            .iter()
            .copied()
            .filter(|scope| *scope != Scope::Unknown)
            .collect();
        let changed = self
            .log
            .set_device_scopes(device_id, &scopes_text(&scopes))
            .await
            .map_err(internal)?;
        if changed {
            let mut narrowed = false;
            for access in self
                .access
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .values_mut()
                .filter(|a| a.principal.device_id.as_deref() == Some(device_id))
            {
                narrowed |= access.principal.scopes.iter().any(|s| !scopes.contains(s));
                access.principal.scopes = scopes.clone();
            }
            // A scope taken away: sockets the device already has open (the
            // event stream, a live view it watches or controls) were let in
            // under the old scopes and check them only then, so they close
            // now. The device signs in again with what it still has.
            if narrowed {
                if let Some(token) = self
                    .revocations
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(device_id)
                {
                    token.cancel();
                }
                self.access
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .retain(|_, a| a.principal.device_id.as_deref() != Some(device_id));
            }
        }
        Ok(changed)
    }

    /// Whether `principal` may use `scope` now: a device's scopes as they
    /// are after any change since its socket opened.
    pub fn may_now(&self, principal: &Principal, scope: Scope) -> bool {
        let Some(device_id) = principal.device_id.as_deref() else {
            return principal.may(scope);
        };
        if principal.revoked.is_cancelled() {
            return false;
        }
        let access = self.access.lock().unwrap_or_else(|p| p.into_inner());
        access
            .values()
            .find(|a| a.principal.device_id.as_deref() == Some(device_id))
            .map_or_else(|| principal.may(scope), |a| a.principal.may(scope))
    }

    /// Whether `principal` may use a shell on this computer now: it may
    /// control and approve, with no risk ceiling. A device that may allow
    /// any command the agent asks to run may type one itself; one that may
    /// not, may not.
    pub fn may_shell_now(&self, principal: &Principal) -> bool {
        let may =
            |p: &Principal| p.may(Scope::Control) && p.may(Scope::Approve) && p.ceiling.is_none();
        let Some(device_id) = principal.device_id.as_deref() else {
            return may(principal);
        };
        if principal.revoked.is_cancelled() {
            return false;
        }
        let access = self.access.lock().unwrap_or_else(|p| p.into_inner());
        access
            .values()
            .find(|a| a.principal.device_id.as_deref() == Some(device_id))
            .map_or_else(|| may(principal), |a| may(&a.principal))
    }

    /// The message files whose links revoking a device ends.
    pub fn set_files(&self, files: Arc<crate::files::FileStore>) {
        *self.files.lock().unwrap_or_else(|p| p.into_inner()) = Some(files);
    }

    /// Revoke a device: its tokens stop working, its sockets close now and
    /// the file links it may hold stop working.
    pub async fn revoke(&self, device_id: &str) -> Result<bool, ProtocolError> {
        let revoked = self
            .log
            .revoke_device(device_id, now_ms())
            .await
            .map_err(internal)?;
        if revoked {
            let files = self.files.lock().unwrap_or_else(|p| p.into_inner()).clone();
            if let Some(files) = files {
                // A failure leaves links working until they expire
                // (LINK_LIFETIME), not the device paired.
                if let Err(error) = files.rotate_key() {
                    warn!(%error, "Could not end the revoked device's file links");
                }
            }
        }
        if let Some(token) = self
            .revocations
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(device_id)
        {
            token.cancel();
        }
        self.access
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|_, a| a.principal.device_id.as_deref() != Some(device_id));
        if let Err(error) = self.load_admission().await {
            warn!(error = %error.message, "Could not reload remote admission");
        }
        Ok(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn pairing() -> Pairing {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let log = SqliteClientEventLog::open(pool).await.unwrap();
        Pairing::new(
            log,
            None,
            crate::PublicUrl {
                endpoints: aidaemon_core::EndpointSource::fixed(Vec::new()),
                fallback: "http://127.0.0.1:8788".into(),
            },
            Risk::Low,
            "0123456789abcdef".into(),
        )
    }

    fn device(id: &str, kind: &str, created_at_ms: u64) -> DeviceRow {
        DeviceRow {
            device_id: id.into(),
            name: format!("Synthetic {id}"),
            kind: kind.into(),
            public_key: "unused".into(),
            scopes: "observe".into(),
            risk_ceiling: None,
            created_at_ms,
            last_seen_ms: None,
            revoked_at_ms: None,
        }
    }

    fn close_window(pairing: &Pairing) {
        pairing
            .admission
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .open_until_ms = 0;
    }

    const PHONE: &str = "11111111111111111111111111111111111111111111111111111111111111aa";
    const STRANGER: &str = "22222222222222222222222222222222222222222222222222222222222222bb";

    mod socket_sign_in {
        use super::*;
        use p256::ecdsa::SigningKey;
        use p256::ecdsa::signature::Signer as _;
        use p256::elliptic_curve::Generate as _;

        const DAEMON: &str = "33333333333333333333333333333333333333333333333333333333333333cc";
        const OTHER_DAEMON: &str =
            "44444444444444444444444444444444444444444444444444444444444444dd";

        struct Setup {
            pairing: Pairing,
            key: SigningKey,
        }

        /// A paired phone that signed in from PHONE once, at a daemon whose
        /// peer-to-peer address names DAEMON.
        async fn setup() -> Setup {
            let pairing = pairing().await;
            pairing.set_remote_url(Some(format!("iroh://{DAEMON}@relay.example.test")));
            let key = SigningKey::generate_from_rng(&mut rand::rng());
            let mut row = device("dv-phone", "web", now_ms());
            row.public_key =
                URL_SAFE_NO_PAD.encode(key.verifying_key().to_sec1_point(false).as_bytes());
            pairing.log.add_device(&row).await.unwrap();
            pairing.record_endpoint("dv-phone", PHONE).await;
            Setup { pairing, key }
        }

        fn nonce() -> String {
            random_hex()[..32].to_string()
        }

        fn signed(key: &SigningKey, message: &str) -> String {
            let signature: Signature = key.sign(message.as_bytes());
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        }

        /// A proof as the phone makes it: for `daemon`, from `endpoint`.
        fn proof(
            key: &SigningKey,
            daemon: &str,
            endpoint: &str,
            at: u64,
            nonce: &str,
        ) -> SocketSignIn {
            SocketSignIn {
                device_id: "dv-phone".into(),
                issued_at_ms: at,
                nonce: nonce.into(),
                signature: signed(
                    key,
                    &socket_sign_in_message(daemon, endpoint, "dv-phone", at, nonce),
                ),
            }
        }

        fn refused(result: Result<AccessGrant, ProtocolError>) -> ErrorCode {
            result.expect_err("refused").code
        }

        #[tokio::test]
        async fn a_fresh_proof_signs_in_once_and_its_token_works() {
            let s = setup().await;
            let good = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
            let grant = s.pairing.sign_in_socket(&good, PHONE).await.unwrap();
            assert_eq!(grant.device_id, "dv-phone");
            assert!(s.pairing.principal(&grant.access_token).is_some());
            // The same proof again (a replay): refused.
            assert_eq!(
                refused(s.pairing.sign_in_socket(&good, PHONE).await),
                ErrorCode::Unauthorized
            );
            // The same random value in upper case is the same value.
            let mut shouted = good.clone();
            shouted.nonce = good.nonce.to_uppercase();
            shouted.signature = signed(
                &s.key,
                &socket_sign_in_message(
                    DAEMON,
                    PHONE,
                    "dv-phone",
                    good.issued_at_ms,
                    &shouted.nonce,
                ),
            );
            assert_eq!(
                refused(s.pairing.sign_in_socket(&shouted, PHONE).await),
                ErrorCode::Unauthorized
            );
            // A new one is fine.
            let next = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
            assert!(s.pairing.sign_in_socket(&next, PHONE).await.is_ok());
        }

        #[tokio::test]
        async fn a_proof_is_good_only_at_this_daemon_and_from_its_own_endpoint() {
            let s = setup().await;
            // Made for another daemon.
            let elsewhere = proof(&s.key, OTHER_DAEMON, PHONE, now_ms(), &nonce());
            assert_eq!(
                refused(s.pairing.sign_in_socket(&elsewhere, PHONE).await),
                ErrorCode::Unauthorized
            );
            // Made for PHONE, sent from another endpoint (even a known one).
            s.pairing.record_endpoint("dv-phone", STRANGER).await;
            let lifted = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
            assert_eq!(
                refused(s.pairing.sign_in_socket(&lifted, STRANGER).await),
                ErrorCode::Unauthorized
            );
            // And it was not used up by those refusals.
            assert!(s.pairing.sign_in_socket(&lifted, PHONE).await.is_ok());
        }

        #[tokio::test]
        async fn an_endpoint_the_device_never_signed_in_from_takes_a_challenge() {
            let s = setup().await;
            let new_endpoint = STRANGER;
            let first = proof(&s.key, DAEMON, new_endpoint, now_ms(), &nonce());
            assert_eq!(
                refused(s.pairing.sign_in_socket(&first, new_endpoint).await),
                ErrorCode::Unauthorized
            );
        }

        #[tokio::test]
        async fn a_stale_early_or_pre_start_proof_is_refused() {
            let s = setup().await;
            let now = now_ms();
            for at in [
                now - SOCKET_SIGN_IN_MAX_AGE_MS - 1_000,
                now + SOCKET_SIGN_IN_MAX_AHEAD_MS + 1_000,
                // Within the window, but before this gateway started: the
                // run before may have taken it.
                s.pairing.started_ms - 1,
            ] {
                let old = proof(&s.key, DAEMON, PHONE, at, &nonce());
                assert_eq!(
                    refused(s.pairing.sign_in_socket(&old, PHONE).await),
                    ErrorCode::Unauthorized,
                    "issued at {at} (now {now})"
                );
            }
        }

        /// A challenge sign-in names the daemon it was signed for: this
        /// daemon's own key passes; a signature for another daemon (one that
        /// relayed this daemon's challenge) does not, and an unbound one
        /// passes only from this computer.
        #[tokio::test]
        async fn a_challenge_signature_counts_only_at_the_daemon_it_names() {
            let s = setup().await;
            let sign = |challenge: &Challenge, audience: &str| {
                signed(
                    &s.key,
                    &challenge_message_v2(
                        &challenge.challenge_id,
                        &challenge.nonce,
                        audience,
                        Some(PHONE),
                    ),
                )
            };
            let own = format!("iroh:{DAEMON}");
            let c = s.pairing.challenge("dv-phone", Some(PHONE)).await.unwrap();
            assert!(
                s.pairing
                    .open_session(&c.challenge_id, &sign(&c, &own), Some(&own), false)
                    .await
                    .is_ok()
            );
            let other = format!("iroh:{OTHER_DAEMON}");
            let c = s.pairing.challenge("dv-phone", Some(PHONE)).await.unwrap();
            assert_eq!(
                refused(
                    s.pairing
                        .open_session(&c.challenge_id, &sign(&c, &other), Some(&other), false)
                        .await
                ),
                ErrorCode::Unauthorized
            );
            // Claiming this daemon over a signature for another fails too.
            let c = s.pairing.challenge("dv-phone", Some(PHONE)).await.unwrap();
            assert_eq!(
                refused(
                    s.pairing
                        .open_session(&c.challenge_id, &sign(&c, &other), Some(&own), false)
                        .await
                ),
                ErrorCode::Unauthorized
            );
            // A loopback address is this daemon only for a request made here.
            let local = "http://127.0.0.1:8788";
            let c = s.pairing.challenge("dv-phone", Some(PHONE)).await.unwrap();
            assert_eq!(
                refused(
                    s.pairing
                        .open_session(&c.challenge_id, &sign(&c, local), Some(local), false)
                        .await
                ),
                ErrorCode::Unauthorized
            );
            let c = s.pairing.challenge("dv-phone", Some(PHONE)).await.unwrap();
            assert!(
                s.pairing
                    .open_session(&c.challenge_id, &sign(&c, local), Some(local), true)
                    .await
                    .is_ok()
            );
            // The unbound message of older clients: from this computer only.
            let old = |c: &Challenge| signed(&s.key, &challenge_message(&c.challenge_id, &c.nonce));
            let c = s.pairing.challenge("dv-phone", None).await.unwrap();
            assert_eq!(
                refused(
                    s.pairing
                        .open_session(&c.challenge_id, &old(&c), None, false)
                        .await
                ),
                ErrorCode::Unauthorized
            );
            let c = s.pairing.challenge("dv-phone", None).await.unwrap();
            assert!(
                s.pairing
                    .open_session(&c.challenge_id, &old(&c), None, true)
                    .await
                    .is_ok()
            );
        }

        /// Asking for challenges in a device's name over and over never locks
        /// it out: the oldest makes way, the newest still signs in.
        #[tokio::test]
        async fn many_challenges_for_one_device_never_lock_it_out() {
            let s = setup().await;
            let first = s.pairing.challenge("dv-phone", None).await.unwrap();
            for _ in 0..(MAX_OPEN_CHALLENGES_PER_DEVICE * 3) {
                s.pairing.challenge("dv-phone", None).await.unwrap();
            }
            let last = s.pairing.challenge("dv-phone", None).await.unwrap();
            let own = format!("iroh:{DAEMON}");
            let sign = |c: &Challenge| {
                signed(
                    &s.key,
                    &challenge_message_v2(&c.challenge_id, &c.nonce, &own, None),
                )
            };
            assert!(
                s.pairing
                    .open_session(&last.challenge_id, &sign(&last), Some(&own), false)
                    .await
                    .is_ok()
            );
            assert_eq!(
                refused(
                    s.pairing
                        .open_session(&first.challenge_id, &sign(&first), Some(&own), false)
                        .await
                ),
                ErrorCode::Unauthorized
            );
        }

        #[tokio::test]
        async fn only_the_devices_key_signing_this_message_passes() {
            let s = setup().await;
            let now = now_ms();
            let n = nonce();
            // Another key.
            let stranger = SigningKey::generate_from_rng(&mut rand::rng());
            let forged = proof(&stranger, DAEMON, PHONE, now, &n);
            assert_eq!(
                refused(s.pairing.sign_in_socket(&forged, PHONE).await),
                ErrorCode::Unauthorized
            );
            // The device's key over a challenge message, not this one.
            let mut crossed = proof(&s.key, DAEMON, PHONE, now, &n);
            crossed.signature = signed(&s.key, &challenge_message("ch-1", &n));
            assert_eq!(
                refused(s.pairing.sign_in_socket(&crossed, PHONE).await),
                ErrorCode::Unauthorized
            );
            // A good signature with a field changed after signing.
            let mut moved = proof(&s.key, DAEMON, PHONE, now, &n);
            moved.issued_at_ms += 1;
            assert_eq!(
                refused(s.pairing.sign_in_socket(&moved, PHONE).await),
                ErrorCode::Unauthorized
            );
            // A malformed random value.
            for bad in ["short", &"z".repeat(32), &"a".repeat(65)] {
                let odd = proof(&s.key, DAEMON, PHONE, now, bad);
                assert_eq!(
                    refused(s.pairing.sign_in_socket(&odd, PHONE).await),
                    ErrorCode::Unauthorized
                );
            }
        }

        #[tokio::test]
        async fn a_revoked_device_or_a_daemon_without_an_address_refuses() {
            let s = setup().await;
            let good = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
            s.pairing.set_remote_url(None);
            assert_eq!(
                refused(s.pairing.sign_in_socket(&good, PHONE).await),
                ErrorCode::Unauthorized
            );
            s.pairing
                .set_remote_url(Some(format!("iroh://{DAEMON}@relay.example.test")));
            s.pairing
                .log
                .revoke_device("dv-phone", now_ms())
                .await
                .unwrap();
            assert_eq!(
                refused(s.pairing.sign_in_socket(&good, PHONE).await),
                ErrorCode::Unauthorized
            );
        }

        #[tokio::test]
        async fn one_device_cannot_fill_the_memory_of_taken_proofs() {
            let s = setup().await;
            for _ in 0..MAX_SOCKET_SIGN_INS_PER_DEVICE {
                let p = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
                s.pairing.sign_in_socket(&p, PHONE).await.unwrap();
            }
            let more = proof(&s.key, DAEMON, PHONE, now_ms(), &nonce());
            assert_eq!(
                refused(s.pairing.sign_in_socket(&more, PHONE).await),
                ErrorCode::RateLimited
            );
        }
    }

    #[tokio::test]
    async fn the_transport_admits_paired_endpoints_and_anyone_only_while_pairing() {
        let pairing = pairing().await;
        pairing.load_admission().await.unwrap();
        assert!(!pairing.admits(STRANGER), "nothing paired, nothing open");

        // An offer opens the door until it expires.
        let offer = pairing.offer(None).await.unwrap();
        assert!(pairing.admits(STRANGER));
        let open_until = pairing.admission.lock().unwrap().open_until_ms;
        assert_eq!(open_until, offer.expires_at_ms);
        close_window(&pairing);
        assert!(!pairing.admits(STRANGER));

        // A device's recorded endpoint is admitted; others are not.
        pairing
            .log
            .add_device(&device("dv-phone", "mobile", now_ms()))
            .await
            .unwrap();
        pairing
            .record_endpoint("dv-phone", &PHONE.to_uppercase())
            .await;
        // Reading the database again keeps the still-open offer open.
        assert!(pairing.admits(STRANGER));
        close_window(&pairing);
        assert!(pairing.admits(PHONE));
        assert!(!pairing.admits(STRANGER));
        // Not hex, or the wrong length: never recorded.
        pairing.record_endpoint("dv-phone", "not-an-endpoint").await;
        assert_eq!(pairing.admission.lock().unwrap().known.len(), 1);

        // Revoking the device closes its endpoint too.
        pairing.revoke("dv-phone").await.unwrap();
        close_window(&pairing);
        assert!(!pairing.admits(PHONE));
    }

    #[tokio::test]
    async fn revoking_a_device_ends_the_file_links_it_may_hold() {
        let pairing = pairing().await;
        let dir = tempfile::tempdir().unwrap();
        let files = Arc::new(
            crate::files::FileStore::open(dir.path().join("files"), dir.path().join("inbox"))
                .unwrap(),
        );
        pairing.set_files(files.clone());
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"synthetic").unwrap();
        let link = |files: &crate::files::FileStore| {
            let url = files
                .register(&path, "note.txt", "text/plain", 9)
                .unwrap()
                .url
                .unwrap();
            let mut it = url.trim_start_matches("/client/v1/files/").split('/');
            (
                it.next().unwrap().to_string(),
                it.next().unwrap().to_string(),
            )
        };
        let (id, held) = link(&files);
        pairing
            .log
            .add_device(&device("dv-phone", "mobile", now_ms()))
            .await
            .unwrap();

        // No such device: nothing revoked, links keep working.
        assert!(!pairing.revoke("dv-nobody").await.unwrap());
        assert!(files.signed(&id, &held).is_some());

        // The phone revoked: the link it held stops working; a device still
        // paired reads the file again at a new link.
        assert!(pairing.revoke("dv-phone").await.unwrap());
        assert!(files.signed(&id, &held).is_none());
        let (_, fresh) = link(&files);
        assert!(files.signed(&id, &fresh).is_some());
    }

    #[tokio::test]
    async fn devices_paired_before_endpoints_were_recorded_keep_the_door_open_until_seen() {
        let pairing = pairing().await;
        pairing
            .log
            .add_device(&device("dv-old-phone", "mobile", 1))
            .await
            .unwrap();
        // A browser never dials peer to peer: it is not waited for.
        pairing
            .log
            .add_device(&device("dv-old-browser", "web", 1))
            .await
            .unwrap();
        pairing.load_admission().await.unwrap();
        assert!(
            pairing.admits(STRANGER),
            "the old phone has not signed in yet"
        );
        pairing.record_endpoint("dv-old-phone", PHONE).await;
        assert!(pairing.admits(PHONE));
        assert!(!pairing.admits(STRANGER));

        // A device paired after recording began is never waited for.
        pairing
            .log
            .add_device(&device("dv-new", "mobile", now_ms() + 1))
            .await
            .unwrap();
        pairing.load_admission().await.unwrap();
        assert!(!pairing.admits(STRANGER));
    }

    #[test]
    fn codes_read_back_whatever_their_case_or_dashes() {
        let code = new_code();
        assert_eq!(code.len(), 9);
        assert_eq!(
            Pairing::code_hash("of-1", &code),
            Pairing::code_hash("of-1", &code.to_lowercase().replace('-', " "))
        );
        assert!(!code.contains('0') && !code.contains('O') && !code.contains('I'));
    }

    #[test]
    fn scopes_round_trip_as_text() {
        assert_eq!(scopes_of(&scopes_text(&ALL_SCOPES)), ALL_SCOPES.to_vec());
        assert_eq!(scopes_of("observe,bogus"), vec![Scope::Observe]);
    }
}
