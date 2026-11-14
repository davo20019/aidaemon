//! Device-key sign-in, the Rust twin of the web client's: pair once (redeem
//! an offer with a fresh P-256 public key, wait for the owner to confirm),
//! then on each connection ask for a challenge, sign
//! [`challenge_message`] and trade the signature for a 15-minute
//! [`AccessGrant`], renewed before it runs out.

use std::time::Duration;

use aidaemon_client_protocol::{
    AccessGrant, Challenge, ChallengeRequest, ClientKind, Enrollment, EnrollmentState,
    RedeemRequest, ServerFrame, SessionRequest, challenge_message,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};
use p256::elliptic_curve::Generate;

use crate::keystore::StoredDevice;

/// A grant this close to its end is renewed before it is used.
pub const RENEW_BEFORE: Duration = Duration::from_secs(60);

/// A device's private key.
#[derive(Clone)]
pub struct DeviceKey(SigningKey);

impl std::fmt::Debug for DeviceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceKey(<redacted>)")
    }
}

impl DeviceKey {
    pub fn generate() -> Self {
        Self(SigningKey::generate_from_rng(&mut rand::rng()))
    }

    /// The key a stored identity holds.
    pub fn from_stored(device: &StoredDevice) -> anyhow::Result<Self> {
        let bytes = URL_SAFE_NO_PAD.decode(device.secret_key.trim())?;
        Ok(Self(SigningKey::from_slice(&bytes).map_err(|_| {
            anyhow::anyhow!("the stored device key is not a P-256 key")
        })?))
    }

    /// The private scalar, base64url, as a [`StoredDevice`] keeps it.
    pub fn secret_b64(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.0.to_bytes())
    }

    /// The public key as the gateway takes it: SEC1 uncompressed, base64url.
    pub fn public_key_b64(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.0.verifying_key().to_sec1_point(false).as_bytes())
    }

    /// The signature (`r || s`, base64url) the gateway checks for a
    /// challenge.
    pub fn sign_challenge(&self, challenge: &Challenge) -> String {
        let signature: Signature = self
            .0
            .sign(challenge_message(&challenge.challenge_id, &challenge.nonce).as_bytes());
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    }
}

/// Why signing in did not work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignInError {
    /// Nothing answered at the gateway's address (the daemon is not
    /// running, or the address is wrong).
    Unreachable(String),
    /// The gateway refused the device (unknown, revoked, a bad signature):
    /// pairing again is the way back.
    Refused(String),
    /// Anything else (a server error, a malformed reply); worth retrying.
    Failed(String),
}

impl std::fmt::Display for SignInError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(why) => write!(f, "the daemon did not answer ({why})"),
            Self::Refused(why) => write!(f, "the daemon refused this device: {why}"),
            Self::Failed(why) => write!(f, "sign-in failed: {why}"),
        }
    }
}

impl std::error::Error for SignInError {}

/// A reply as the sign-in code reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpReply {
    pub status: u16,
    pub body: String,
}

/// The few HTTP calls sign-in and pairing make, so any client can bring its
/// own network stack (reqwest natively, a UI framework's `fetch` in a
/// browser). Futures need not be `Send`.
pub trait Http {
    /// POST a JSON body. A failure to reach the gateway at all is
    /// [`SignInError::Unreachable`].
    fn post_json(
        &self,
        url: &str,
        body: String,
    ) -> impl std::future::Future<Output = Result<HttpReply, SignInError>>;
    fn get(&self, url: &str) -> impl std::future::Future<Output = Result<HttpReply, SignInError>>;
    /// Wait `duration` (between pairing polls).
    fn sleep(&self, duration: Duration) -> impl std::future::Future<Output = ()>;
}

/// POST `body` to `base + path` and read a `T`, or the gateway's error.
async fn post<B: serde::Serialize, T: serde::de::DeserializeOwned>(
    http: &impl Http,
    base: &str,
    path: &str,
    body: &B,
) -> Result<T, SignInError> {
    let body = serde_json::to_string(body).map_err(|e| SignInError::Failed(e.to_string()))?;
    read(
        http.post_json(&format!("{}{path}", base.trim_end_matches('/')), body)
            .await?,
    )
}

/// A gateway reply as a `T`, or as the error it names.
pub fn read<T: serde::de::DeserializeOwned>(reply: HttpReply) -> Result<T, SignInError> {
    if (200..300).contains(&reply.status) {
        return serde_json::from_str(&reply.body).map_err(|e| SignInError::Failed(e.to_string()));
    }
    let message = match serde_json::from_str::<ServerFrame>(&reply.body) {
        Ok(ServerFrame::Error { error, .. }) => error.message,
        _ => format!("HTTP {}", reply.status),
    };
    if (400..500).contains(&reply.status) {
        Err(SignInError::Refused(message))
    } else {
        Err(SignInError::Failed(message))
    }
}

/// Sign `device` in over `http`: a challenge, its signature, a grant.
pub async fn sign_in_with(
    http: &impl Http,
    device: &StoredDevice,
) -> Result<AccessGrant, SignInError> {
    let key = DeviceKey::from_stored(device).map_err(|e| SignInError::Refused(e.to_string()))?;
    let challenge: Challenge = post(
        http,
        &device.gateway,
        "/client/v1/sessions/challenge",
        &ChallengeRequest {
            device_id: device.device_id.clone(),
        },
    )
    .await?;
    post(
        http,
        &device.gateway,
        "/client/v1/sessions",
        &SessionRequest {
            signature: key.sign_challenge(&challenge),
            challenge_id: challenge.challenge_id,
        },
    )
    .await
}

/// What a pairing needs: the offer from the owner's `/pair_device` or
/// `aidaemon pair`, and how the device should be known.
#[derive(Debug, Clone)]
pub struct PairRequest {
    /// The gateway's HTTP base.
    pub gateway: String,
    pub offer_id: String,
    pub code: String,
    pub device_name: String,
    pub device_kind: ClientKind,
    /// The daemon's fingerprint from the pairing link; checked against the
    /// redeem reply when given.
    pub fingerprint: Option<String>,
}

/// Pair over `http`: redeem the offer with a new key, then wait (polling
/// every `poll`) until the owner approves. Returns the identity to keep.
pub async fn pair_with(
    http: &impl Http,
    request: &PairRequest,
    poll: Duration,
) -> anyhow::Result<StoredDevice> {
    let key = DeviceKey::generate();
    let mut enrollment: Enrollment = post(
        http,
        &request.gateway,
        "/client/v1/enrollments/redeem",
        &RedeemRequest {
            offer_id: request.offer_id.clone(),
            code: request.code.clone(),
            device_name: request.device_name.clone(),
            device_kind: request.device_kind,
            public_key: key.public_key_b64(),
            scopes: Vec::new(),
        },
    )
    .await?;
    if let Some(expected) = &request.fingerprint
        && &enrollment.fingerprint != expected
    {
        anyhow::bail!("this is not the daemon the pairing code came from");
    }
    while matches!(
        enrollment.state,
        EnrollmentState::Open | EnrollmentState::Pending
    ) {
        http.sleep(poll).await;
        let url = format!(
            "{}/client/v1/enrollments/{}",
            request.gateway.trim_end_matches('/'),
            request.offer_id
        );
        enrollment = read(http.get(&url).await?)?;
    }
    match (enrollment.state, enrollment.device_id) {
        (EnrollmentState::Approved, Some(device_id)) => Ok(StoredDevice {
            gateway: request.gateway.trim_end_matches('/').to_string(),
            device_id,
            secret_key: key.secret_b64(),
        }),
        (state, _) => anyhow::bail!("the pairing ended {state:?}"),
    }
}

/// A grant kept between connections and renewed when it nears its end.
#[derive(Debug, Default)]
pub struct GrantCache {
    grant: Option<AccessGrant>,
}

impl GrantCache {
    /// A token good for at least [`RENEW_BEFORE`], signing in again over
    /// `http` when the one held is older.
    pub async fn token_with(
        &mut self,
        http: &impl Http,
        device: &StoredDevice,
        now_ms: u64,
    ) -> Result<String, SignInError> {
        let margin = u64::try_from(RENEW_BEFORE.as_millis()).unwrap_or(60_000);
        if let Some(grant) = &self.grant
            && grant.expires_at_ms > now_ms.saturating_add(margin)
        {
            return Ok(grant.access_token.clone());
        }
        let grant = sign_in_with(http, device).await?;
        let token = grant.access_token.clone();
        self.grant = Some(grant);
        Ok(token)
    }

    /// Forget the grant (the gateway refused it).
    pub fn drop_grant(&mut self) {
        self.grant = None;
    }
}

#[cfg(feature = "native")]
pub use native::*;

/// reqwest and tokio: what the terminal UI and the Tauri shell use.
#[cfg(feature = "native")]
mod native {
    use super::*;

    impl Http for reqwest::Client {
        async fn post_json(&self, url: &str, body: String) -> Result<HttpReply, SignInError> {
            let response = self
                .post(url)
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .map_err(classify)?;
            reply(response).await
        }

        async fn get(&self, url: &str) -> Result<HttpReply, SignInError> {
            reply(self.get(url).send().await.map_err(classify)?).await
        }

        async fn sleep(&self, duration: Duration) {
            tokio::time::sleep(duration).await;
        }
    }

    fn classify(error: reqwest::Error) -> SignInError {
        if error.is_connect() || error.is_timeout() {
            SignInError::Unreachable(error.to_string())
        } else {
            SignInError::Failed(error.to_string())
        }
    }

    async fn reply(response: reqwest::Response) -> Result<HttpReply, SignInError> {
        let status = response.status().as_u16();
        let body = response.text().await.map_err(classify)?;
        Ok(HttpReply { status, body })
    }

    /// Sign `device` in: a challenge, its signature, a grant.
    pub async fn sign_in(
        http: &reqwest::Client,
        device: &StoredDevice,
    ) -> Result<AccessGrant, SignInError> {
        sign_in_with(http, device).await
    }

    /// Pair: redeem the offer with a new key, then wait (polling every
    /// `poll`) until the owner approves. Returns the identity to keep.
    pub async fn pair(
        http: &reqwest::Client,
        request: &PairRequest,
        poll: Duration,
    ) -> anyhow::Result<StoredDevice> {
        pair_with(http, request, poll).await
    }

    impl GrantCache {
        /// [`GrantCache::token_with`] over reqwest.
        pub async fn token(
            &mut self,
            http: &reqwest::Client,
            device: &StoredDevice,
            now_ms: u64,
        ) -> Result<String, SignInError> {
            self.token_with(http, device, now_ms).await
        }
    }
}

#[cfg(feature = "native")]
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_survives_the_store_and_signs_what_the_gateway_checks() {
        let key = DeviceKey::generate();
        let stored = StoredDevice {
            gateway: "http://127.0.0.1:1".into(),
            device_id: "dev-synthetic-1".into(),
            secret_key: key.secret_b64(),
        };
        let back = DeviceKey::from_stored(&stored).unwrap();
        assert_eq!(back.public_key_b64(), key.public_key_b64());
        let challenge = Challenge {
            challenge_id: "c1".into(),
            nonce: "n1".into(),
            expires_at_ms: 1,
        };
        let signature = URL_SAFE_NO_PAD
            .decode(back.sign_challenge(&challenge))
            .unwrap();
        assert_eq!(signature.len(), 64);
        let public = URL_SAFE_NO_PAD.decode(key.public_key_b64()).unwrap();
        let verifying = p256::ecdsa::VerifyingKey::from_sec1_bytes(&public).unwrap();
        use p256::ecdsa::signature::Verifier;
        verifying
            .verify(
                challenge_message("c1", "n1").as_bytes(),
                &Signature::from_slice(&signature).unwrap(),
            )
            .unwrap();
        assert!(!format!("{key:?}").contains(&key.secret_b64()));
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn a_gateway_that_is_not_there_is_unreachable() {
        let stored = StoredDevice {
            gateway: "http://127.0.0.1:9".into(),
            device_id: "dev-synthetic-1".into(),
            secret_key: DeviceKey::generate().secret_b64(),
        };
        let error = sign_in(&reqwest::Client::new(), &stored).await.unwrap_err();
        assert!(matches!(error, SignInError::Unreachable(_)), "{error:?}");
    }
}
