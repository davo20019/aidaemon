//! Pairing devices and their short-lived access tokens.
//!
//! The owner makes a single-use offer (a code and a link, valid ten
//! minutes); a device redeems it with its P-256 public key; the owner
//! confirms on a chat they already use (an approval card through the hub)
//! or, for an offer made at the terminal, there. Each connection proves the
//! key by signing a challenge and gets a 15-minute access token carrying
//! the device's scopes and risk ceiling. Revoking a device ends its tokens
//! and its open sockets at once.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use aidaemon_channels::hub::ApprovalRequest;
use aidaemon_client_protocol::{
    challenge_message, AccessGrant, Challenge, ClientKind, DeviceInfo, Enrollment, EnrollmentState,
    ErrorCode, PairingOffer, ProtocolError, RedeemRequest, Risk, Scope, ALL_SCOPES,
};
use aidaemon_store_sqlite::{DeviceRow, PairingRow, SqliteClientEventLog};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
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

fn ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

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
}

pub struct Pairing {
    log: SqliteClientEventLog,
    /// Where confirmation cards go (the hub's approval queue).
    approvals: Option<mpsc::Sender<ApprovalRequest>>,
    /// What the offer link names as the gateway, read per offer.
    public_url: crate::PublicUrl,
    fingerprint: String,
    access: Mutex<HashMap<String, Access>>,
    challenges: Mutex<HashMap<String, PendingChallenge>>,
    /// Per device, what revoking it cancels.
    revocations: Mutex<HashMap<String, CancellationToken>>,
    /// The ceiling a watch gets unless the owner sets another.
    watch_ceiling: Risk,
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
    let bytes = uuid::Uuid::new_v4().into_bytes();
    let letters: String = bytes[..8]
        .iter()
        .map(|b| CODE_ALPHABET[usize::from(*b) % CODE_ALPHABET.len()] as char)
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
    ) -> Self {
        // The fingerprint identifies this daemon's database, which holds
        // the devices it paired. It is keyed on the configured address,
        // never a live tunnel origin, so it does not change when one does.
        let fingerprint =
            sha(&format!("aidaemon-client-gateway:{}", public_url.fallback))[..16].to_string();
        Self {
            log,
            approvals,
            public_url,
            fingerprint,
            access: Mutex::default(),
            challenges: Mutex::default(),
            revocations: Mutex::default(),
            watch_ceiling,
        }
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
        let link = format!(
            "aidaemon-pair:?gateway={}&offer={offer_id}&code={code}&fp={}",
            percent(&public_url),
            self.fingerprint
        );
        Ok(PairingOffer {
            offer_id,
            code,
            expires_at_ms,
            link,
            fingerprint: self.fingerprint.clone(),
            public_url: Some(public_url),
        })
    }

    /// A device redeems an offer. The owner is then asked; the device polls
    /// [`Self::enrollment`].
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
            ALL_SCOPES.to_vec()
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
            if !self
                .log
                .decide_offer(offer_id, "approved", Some(&device_id))
                .await
                .map_err(internal)?
            {
                return Err(ProtocolError::new(ErrorCode::Conflict, "Already settled"));
            }
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

    pub async fn challenge(&self, device_id: &str) -> Result<Challenge, ProtocolError> {
        let device = self.active_device(device_id).await?;
        let challenge_id = format!("ch-{}", &random_hex()[..20]);
        let nonce = random_hex();
        let expires_at_ms = now_ms() + ms(CHALLENGE_TTL);
        let mut challenges = self.challenges.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        challenges.retain(|_, c| c.expires_at_ms > now);
        if challenges.len() >= MAX_OPEN_CHALLENGES {
            return Err(ProtocolError::new(
                ErrorCode::RateLimited,
                "Too many sign-ins at once; try again in a minute",
            ));
        }
        challenges.insert(
            challenge_id.clone(),
            PendingChallenge {
                device_id: device.device_id,
                nonce: nonce.clone(),
                expires_at_ms,
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

    /// Trade a signed challenge for an access token.
    pub async fn open_session(
        &self,
        challenge_id: &str,
        signature: &str,
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
        let refused =
            || ProtocolError::new(ErrorCode::Unauthorized, "The signature does not verify");
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
        key.verify(
            challenge_message(challenge_id, &challenge.nonce).as_bytes(),
            &signature,
        )
        .map_err(|_| refused())?;

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
        Ok(AccessGrant {
            access_token: token,
            expires_at_ms,
            device_id: device.device_id,
            scopes,
            risk_ceiling: ceiling,
        })
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

    /// Revoke a device: its tokens stop working and its sockets close now.
    pub async fn revoke(&self, device_id: &str) -> Result<bool, ProtocolError> {
        let revoked = self
            .log
            .revoke_device(device_id, now_ms())
            .await
            .map_err(internal)?;
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
        Ok(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
