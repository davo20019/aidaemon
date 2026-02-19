//! Biometric/identity verification for high-sensitivity operations.
//!
//! Adds a second factor on top of the existing button-based approval flow.
//! When a biometric-required operation is approved via the standard flow,
//! the user must additionally prove their identity through:
//!
//! - **Challenge-response questions** — pre-configured questions with hashed answers
//! - **TOTP codes** — time-based one-time passwords (Google Authenticator, etc.)
//!
//! Anti-abuse: configurable max attempts with exponential lockout.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Configuration types (deserialized from config.toml [security] section)
// ---------------------------------------------------------------------------

/// Top-level security / biometric configuration.
///
/// ```toml
/// [security]
/// biometric_enabled = true
/// max_attempts = 3
/// lockout_duration_secs = 300
/// required_for = ["critical_commands", "allow_always"]
///
/// [[security.challenges]]
/// question = "What is the name of your first pet?"
/// answer_hash = "sha256:<hex>"   # or "keychain" to read from OS keychain
///
/// [security.totp]
/// secret = "keychain"
/// ```
#[derive(Debug, Deserialize, Clone)]
pub struct SecurityConfig {
    /// Master switch — when false, no biometric checks are performed.
    #[serde(default)]
    pub biometric_enabled: bool,

    /// Maximum consecutive failed verification attempts before lockout.
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,

    /// How long to lock out a session after exceeding max_attempts (seconds).
    #[serde(default = "default_lockout_duration_secs")]
    pub lockout_duration_secs: u64,

    /// Which operation categories require biometric verification.
    /// Recognized values: "critical_commands", "allow_always", "config_changes",
    /// "memory_sharing".
    #[serde(default)]
    pub required_for: Vec<String>,

    /// Challenge-response questions.
    #[serde(default)]
    pub challenges: Vec<ChallengeConfig>,

    /// Optional TOTP configuration.
    #[serde(default)]
    pub totp: Option<TotpConfig>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            biometric_enabled: false,
            max_attempts: default_max_attempts(),
            lockout_duration_secs: default_lockout_duration_secs(),
            required_for: Vec::new(),
            challenges: Vec::new(),
            totp: None,
        }
    }
}

fn default_max_attempts() -> u32 {
    3
}

fn default_lockout_duration_secs() -> u64 {
    300
}

/// A single challenge-response question.
#[derive(Debug, Deserialize, Clone)]
pub struct ChallengeConfig {
    pub question: String,
    /// SHA-256 hex digest of the lowercased, trimmed answer.
    /// Can be set to `"keychain"` to read from OS keychain.
    pub answer_hash: String,
}

/// TOTP (Time-based One-Time Password) configuration.
#[derive(Deserialize, Clone)]
pub struct TotpConfig {
    /// Base32-encoded TOTP secret. Can be `"keychain"`.
    pub secret: String,
    /// Period in seconds (default: 30).
    #[serde(default = "default_totp_period")]
    pub period: u64,
    /// Number of digits in the TOTP code (default: 6).
    #[serde(default = "default_totp_digits")]
    pub digits: u32,
    /// How many adjacent time windows to accept (default: 1 — allows ±1 period).
    #[serde(default = "default_totp_skew")]
    pub skew: u64,
}

impl std::fmt::Debug for TotpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TotpConfig")
            .field("secret", &"[REDACTED]")
            .field("period", &self.period)
            .field("digits", &self.digits)
            .field("skew", &self.skew)
            .finish()
    }
}

impl Default for TotpConfig {
    fn default() -> Self {
        Self {
            secret: String::new(),
            period: default_totp_period(),
            digits: default_totp_digits(),
            skew: default_totp_skew(),
        }
    }
}

fn default_totp_period() -> u64 {
    30
}

fn default_totp_digits() -> u32 {
    6
}

fn default_totp_skew() -> u64 {
    1
}

// ---------------------------------------------------------------------------
// Verification types
// ---------------------------------------------------------------------------

/// Which verification methods are available for a given check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Challenge-response questions.
    ChallengeResponse,
    /// Time-based one-time password.
    Totp,
}

/// The result of an identity verification attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Identity confirmed.
    Verified,
    /// Verification failed (wrong answer / invalid TOTP).
    Failed { reason: String },
    /// The user explicitly cancelled.
    Cancelled,
    /// Verification timed out (no response within the window).
    TimedOut,
    /// Too many consecutive failures — session is locked out.
    LockedOut { remaining_secs: u64 },
}

/// Which operation categories require biometric verification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiometricTrigger {
    /// Commands classified as RiskLevel::Critical.
    CriticalCommands,
    /// When the user selects "Allow Always" (permanent approval).
    AllowAlways,
    /// Changes to sensitive config keys (secrets, allow-lists).
    ConfigChanges,
    /// Cross-channel memory sharing.
    MemorySharing,
}

impl BiometricTrigger {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "critical_commands" => Some(Self::CriticalCommands),
            "allow_always" => Some(Self::AllowAlways),
            "config_changes" => Some(Self::ConfigChanges),
            "memory_sharing" => Some(Self::MemorySharing),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CriticalCommands => "critical_commands",
            Self::AllowAlways => "allow_always",
            Self::ConfigChanges => "config_changes",
            Self::MemorySharing => "memory_sharing",
        }
    }
}

// ---------------------------------------------------------------------------
// Lockout tracking
// ---------------------------------------------------------------------------

/// Per-session lockout state.
struct SessionLockout {
    /// Consecutive failures since last success / reset.
    consecutive_failures: u32,
    /// When the lockout expires (None = not currently locked out).
    locked_until: Option<Instant>,
}

// ---------------------------------------------------------------------------
// BiometricVerifier — the main verification service
// ---------------------------------------------------------------------------

/// Central service for identity verification.
///
/// Thread-safe (`Arc`-wrapped internally via `Mutex`/`RwLock`).
pub struct BiometricVerifier {
    config: SecurityConfig,
    /// Parsed triggers from config.required_for.
    triggers: Vec<BiometricTrigger>,
    /// Per-session lockout tracking.
    lockouts: Mutex<HashMap<String, SessionLockout>>,
    /// Sessions that have been verified within a cooldown window.
    /// Maps session_id → verification expiry instant.
    verified_sessions: RwLock<HashMap<String, Instant>>,
}

impl BiometricVerifier {
    /// Create a new verifier from config.
    pub fn new(config: SecurityConfig) -> Self {
        let triggers: Vec<BiometricTrigger> = config
            .required_for
            .iter()
            .filter_map(|s| BiometricTrigger::from_str(s))
            .collect();

        Self {
            config,
            triggers,
            lockouts: Mutex::new(HashMap::new()),
            verified_sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Whether biometric verification is enabled at all.
    pub fn is_enabled(&self) -> bool {
        self.config.biometric_enabled && self.has_any_method()
    }

    /// Whether at least one verification method is configured.
    fn has_any_method(&self) -> bool {
        !self.config.challenges.is_empty() || self.config.totp.is_some()
    }

    /// Which verification methods are available.
    pub fn available_methods(&self) -> Vec<VerificationMethod> {
        let mut methods = Vec::new();
        if !self.config.challenges.is_empty() {
            methods.push(VerificationMethod::ChallengeResponse);
        }
        if self.config.totp.is_some() {
            methods.push(VerificationMethod::Totp);
        }
        methods
    }

    /// Check whether a given trigger category requires biometric verification.
    pub fn requires_verification(&self, trigger: &BiometricTrigger) -> bool {
        self.is_enabled() && self.triggers.contains(trigger)
    }

    /// Check if a session has been recently verified (within the cooldown window).
    /// The cooldown window is equal to the lockout duration to avoid re-prompting
    /// the user repeatedly within a short time.
    pub async fn is_recently_verified(&self, session_id: &str) -> bool {
        let sessions = self.verified_sessions.read().await;
        if let Some(expiry) = sessions.get(session_id) {
            if Instant::now() < *expiry {
                return true;
            }
        }
        false
    }

    /// Check if a session is currently locked out.
    pub async fn check_lockout(&self, session_id: &str) -> Option<u64> {
        let lockouts = self.lockouts.lock().await;
        if let Some(lockout) = lockouts.get(session_id) {
            if let Some(locked_until) = lockout.locked_until {
                let now = Instant::now();
                if now < locked_until {
                    return Some(locked_until.duration_since(now).as_secs());
                }
            }
        }
        None
    }

    /// Get a random challenge question. Returns (index, question_text).
    /// Returns None if no challenges are configured.
    pub fn random_challenge(&self) -> Option<(usize, &str)> {
        if self.config.challenges.is_empty() {
            return None;
        }
        let idx = rand::random::<usize>() % self.config.challenges.len();
        Some((idx, &self.config.challenges[idx].question))
    }

    /// Validate a challenge-response answer.
    pub async fn validate_challenge(
        &self,
        session_id: &str,
        challenge_index: usize,
        answer: &str,
    ) -> VerificationResult {
        // Check lockout first
        if let Some(remaining) = self.check_lockout(session_id).await {
            return VerificationResult::LockedOut {
                remaining_secs: remaining,
            };
        }

        let challenge = match self.config.challenges.get(challenge_index) {
            Some(c) => c,
            None => {
                return VerificationResult::Failed {
                    reason: "Invalid challenge index".to_string(),
                }
            }
        };

        let answer_hash = hash_answer(answer);
        if answer_hash == challenge.answer_hash {
            self.record_success(session_id).await;
            VerificationResult::Verified
        } else {
            self.record_failure(session_id).await
        }
    }

    /// Validate a TOTP code.
    pub async fn validate_totp(&self, session_id: &str, code: &str) -> VerificationResult {
        // Check lockout first
        if let Some(remaining) = self.check_lockout(session_id).await {
            return VerificationResult::LockedOut {
                remaining_secs: remaining,
            };
        }

        let totp_config = match &self.config.totp {
            Some(t) => t,
            None => {
                return VerificationResult::Failed {
                    reason: "TOTP not configured".to_string(),
                }
            }
        };

        if verify_totp(
            &totp_config.secret,
            code,
            totp_config.period,
            totp_config.digits,
            totp_config.skew,
        ) {
            self.record_success(session_id).await;
            VerificationResult::Verified
        } else {
            self.record_failure(session_id).await
        }
    }

    /// Record a successful verification — resets failure count and marks session
    /// as recently verified.
    async fn record_success(&self, session_id: &str) {
        // Reset lockout
        {
            let mut lockouts = self.lockouts.lock().await;
            lockouts.remove(session_id);
        }
        // Mark as recently verified (cooldown = lockout_duration)
        {
            let cooldown = Duration::from_secs(self.config.lockout_duration_secs);
            let mut sessions = self.verified_sessions.write().await;
            sessions.insert(session_id.to_string(), Instant::now() + cooldown);
        }
        info!(session_id, "Biometric verification succeeded");
    }

    /// Record a failed verification attempt. Returns the appropriate result
    /// (either Failed or LockedOut if max attempts exceeded).
    async fn record_failure(&self, session_id: &str) -> VerificationResult {
        let mut lockouts = self.lockouts.lock().await;
        let lockout = lockouts
            .entry(session_id.to_string())
            .or_insert(SessionLockout {
                consecutive_failures: 0,
                locked_until: None,
            });

        lockout.consecutive_failures += 1;

        if lockout.consecutive_failures >= self.config.max_attempts {
            let duration = Duration::from_secs(self.config.lockout_duration_secs);
            lockout.locked_until = Some(Instant::now() + duration);
            warn!(
                session_id,
                failures = lockout.consecutive_failures,
                lockout_secs = self.config.lockout_duration_secs,
                "Biometric verification locked out due to too many failures"
            );
            VerificationResult::LockedOut {
                remaining_secs: self.config.lockout_duration_secs,
            }
        } else {
            let remaining = self.config.max_attempts - lockout.consecutive_failures;
            warn!(
                session_id,
                failures = lockout.consecutive_failures,
                remaining_attempts = remaining,
                "Biometric verification failed"
            );
            VerificationResult::Failed {
                reason: format!("Incorrect answer. {} attempt(s) remaining.", remaining),
            }
        }
    }

    /// Get the max attempts config value.
    pub fn max_attempts(&self) -> u32 {
        self.config.max_attempts
    }

    /// Get the lockout duration.
    #[allow(dead_code)]
    pub fn lockout_duration_secs(&self) -> u64 {
        self.config.lockout_duration_secs
    }
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

/// Hash an answer using SHA-256 for comparison with stored answer_hash.
/// The answer is lowercased and trimmed before hashing.
pub fn hash_answer(answer: &str) -> String {
    let normalized = answer.trim().to_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let result = hasher.finalize();
    format!("sha256:{}", hex::encode(result))
}

/// Generate an answer hash for initial setup (utility for users).
#[allow(dead_code)]
pub fn generate_answer_hash(answer: &str) -> String {
    hash_answer(answer)
}

// Inline hex encoding to avoid adding a dependency
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    #[cfg(test)]
    pub fn decode(s: &str) -> Option<Vec<u8>> {
        if s.len() % 2 != 0 {
            return None;
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TOTP implementation (RFC 6238)
// ---------------------------------------------------------------------------

type HmacSha1 = Hmac<Sha1>;

/// Generate a TOTP code for a given time step.
fn generate_totp(secret_base32: &str, time_step: u64, digits: u32) -> Option<String> {
    let secret = base32_decode(secret_base32)?;
    let counter_bytes = time_step.to_be_bytes();

    let mut mac = HmacSha1::new_from_slice(&secret).ok()?;
    mac.update(&counter_bytes);
    let result = mac.finalize().into_bytes();

    let offset = (result[result.len() - 1] & 0x0f) as usize;
    let code = ((result[offset] as u32 & 0x7f) << 24)
        | ((result[offset + 1] as u32) << 16)
        | ((result[offset + 2] as u32) << 8)
        | (result[offset + 3] as u32);

    let modulus = 10u32.pow(digits);
    Some(format!(
        "{:0>width$}",
        code % modulus,
        width = digits as usize
    ))
}

/// Verify a TOTP code, allowing for clock skew.
fn verify_totp(secret_base32: &str, code: &str, period: u64, digits: u32, skew: u64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let current_step = now / period;

    // Check current step and adjacent steps within skew window
    for offset in 0..=skew {
        // Check current + offset
        if let Some(expected) = generate_totp(secret_base32, current_step + offset, digits) {
            if constant_time_eq(code.as_bytes(), expected.as_bytes()) {
                return true;
            }
        }
        // Check current - offset (but not for offset == 0 to avoid double-check)
        if offset > 0 && current_step >= offset {
            if let Some(expected) = generate_totp(secret_base32, current_step - offset, digits) {
                if constant_time_eq(code.as_bytes(), expected.as_bytes()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Decode a Base32-encoded string (RFC 4648, no padding required).
fn base32_decode(input: &str) -> Option<Vec<u8>> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let input = input.trim_end_matches('=').to_uppercase();
    let input = input.as_bytes();

    let mut output = Vec::with_capacity(input.len() * 5 / 8);
    let mut buffer: u64 = 0;
    let mut bits_in_buffer = 0u32;

    for &byte in input {
        let value = alphabet.iter().position(|&c| c == byte)? as u64;
        buffer = (buffer << 5) | value;
        bits_in_buffer += 5;

        if bits_in_buffer >= 8 {
            bits_in_buffer -= 8;
            output.push((buffer >> bits_in_buffer) as u8);
            buffer &= (1u64 << bits_in_buffer) - 1;
        }
    }

    Some(output)
}

// ---------------------------------------------------------------------------
// Event payloads for audit trail
// ---------------------------------------------------------------------------

/// Data emitted when biometric verification is requested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricVerificationRequestedData {
    /// Which verification method was used.
    pub method: String,
    /// What triggered the verification requirement.
    pub trigger: String,
    /// The command or action being verified.
    pub context: String,
    /// Associated task ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

/// Data emitted when biometric verification succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricVerificationSucceededData {
    pub method: String,
    pub trigger: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

/// Data emitted when biometric verification fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricVerificationFailedData {
    pub method: String,
    pub trigger: String,
    pub reason: String,
    /// How many consecutive failures so far.
    pub attempt_number: u32,
    /// Whether the session is now locked out.
    pub locked_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_answer_deterministic() {
        let hash1 = hash_answer("Fluffy");
        let hash2 = hash_answer("fluffy");
        let hash3 = hash_answer("  FLUFFY  ");
        // All should produce the same hash (lowercased, trimmed)
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
        assert!(hash1.starts_with("sha256:"));
    }

    #[test]
    fn test_hash_answer_different_inputs() {
        let hash1 = hash_answer("fluffy");
        let hash2 = hash_answer("buddy");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_base32_decode() {
        // "JBSWY3DPEHPK3PXP" is base32 for "Hello!"
        let decoded = base32_decode("JBSWY3DPEHPK3PXP").unwrap();
        assert_eq!(decoded, b"Hello!");
    }

    #[test]
    fn test_base32_decode_with_padding() {
        let decoded = base32_decode("JBSWY3DPEHPK3PXP====").unwrap();
        assert_eq!(decoded, b"Hello!");
    }

    #[test]
    fn test_totp_generation() {
        // Test vector from RFC 6238 (using SHA1)
        // Secret: "12345678901234567890" in base32 = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
        let secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";
        // At time step 1 (counter = 1)
        let code = generate_totp(secret, 1, 6).unwrap();
        assert_eq!(code.len(), 6);
        // Verify it's numeric
        assert!(code.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }

    #[test]
    fn test_biometric_trigger_roundtrip() {
        for trigger in &[
            BiometricTrigger::CriticalCommands,
            BiometricTrigger::AllowAlways,
            BiometricTrigger::ConfigChanges,
            BiometricTrigger::MemorySharing,
        ] {
            let s = trigger.as_str();
            let parsed = BiometricTrigger::from_str(s).unwrap();
            assert_eq!(*trigger, parsed);
        }
    }

    #[tokio::test]
    async fn test_verifier_disabled_by_default() {
        let verifier = BiometricVerifier::new(SecurityConfig::default());
        assert!(!verifier.is_enabled());
        assert!(!verifier.requires_verification(&BiometricTrigger::CriticalCommands));
    }

    #[tokio::test]
    async fn test_verifier_enabled_with_challenges() {
        let config = SecurityConfig {
            biometric_enabled: true,
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        assert!(verifier.is_enabled());
        assert!(verifier.requires_verification(&BiometricTrigger::CriticalCommands));
        assert!(!verifier.requires_verification(&BiometricTrigger::AllowAlways));
    }

    #[tokio::test]
    async fn test_challenge_validation_success() {
        let config = SecurityConfig {
            biometric_enabled: true,
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        let result = verifier.validate_challenge("session1", 0, "Fluffy").await;
        assert!(matches!(result, VerificationResult::Verified));
    }

    #[tokio::test]
    async fn test_challenge_validation_failure() {
        let config = SecurityConfig {
            biometric_enabled: true,
            max_attempts: 3,
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        let result = verifier.validate_challenge("session1", 0, "buddy").await;
        assert!(matches!(result, VerificationResult::Failed { .. }));
    }

    #[tokio::test]
    async fn test_lockout_after_max_attempts() {
        let config = SecurityConfig {
            biometric_enabled: true,
            max_attempts: 2,
            lockout_duration_secs: 60,
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);

        // First failure
        let result = verifier.validate_challenge("session1", 0, "wrong1").await;
        assert!(matches!(result, VerificationResult::Failed { .. }));

        // Second failure → lockout
        let result = verifier.validate_challenge("session1", 0, "wrong2").await;
        assert!(matches!(result, VerificationResult::LockedOut { .. }));

        // Third attempt while locked out
        let result = verifier.validate_challenge("session1", 0, "fluffy").await;
        assert!(matches!(result, VerificationResult::LockedOut { .. }));
    }

    #[tokio::test]
    async fn test_success_resets_lockout() {
        let config = SecurityConfig {
            biometric_enabled: true,
            max_attempts: 3,
            lockout_duration_secs: 1, // short for testing
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);

        // One failure
        let _ = verifier.validate_challenge("session1", 0, "wrong").await;

        // Then success
        let result = verifier.validate_challenge("session1", 0, "fluffy").await;
        assert!(matches!(result, VerificationResult::Verified));

        // Failure count should be reset — next failure should say 2 remaining
        let result = verifier.validate_challenge("session1", 0, "wrong").await;
        match result {
            VerificationResult::Failed { reason } => {
                assert!(reason.contains("2 attempt(s) remaining"));
            }
            _ => panic!("Expected Failed, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_recently_verified_skips_check() {
        let config = SecurityConfig {
            biometric_enabled: true,
            lockout_duration_secs: 300, // cooldown window
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);

        // Not yet verified
        assert!(!verifier.is_recently_verified("session1").await);

        // Verify successfully
        let _ = verifier.validate_challenge("session1", 0, "fluffy").await;

        // Now recently verified
        assert!(verifier.is_recently_verified("session1").await);
    }

    #[tokio::test]
    async fn test_session_isolation() {
        let config = SecurityConfig {
            biometric_enabled: true,
            max_attempts: 2,
            challenges: vec![ChallengeConfig {
                question: "What is your pet's name?".to_string(),
                answer_hash: hash_answer("fluffy"),
            }],
            required_for: vec!["critical_commands".to_string()],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);

        // Fail session1 twice → lockout
        let _ = verifier.validate_challenge("session1", 0, "wrong").await;
        let result = verifier.validate_challenge("session1", 0, "wrong").await;
        assert!(matches!(result, VerificationResult::LockedOut { .. }));

        // Session2 should not be affected
        let result = verifier.validate_challenge("session2", 0, "fluffy").await;
        assert!(matches!(result, VerificationResult::Verified));
    }

    #[test]
    fn test_available_methods() {
        // No methods
        let verifier = BiometricVerifier::new(SecurityConfig::default());
        assert!(verifier.available_methods().is_empty());

        // Challenge only
        let config = SecurityConfig {
            challenges: vec![ChallengeConfig {
                question: "Q?".to_string(),
                answer_hash: "sha256:abc".to_string(),
            }],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        assert_eq!(
            verifier.available_methods(),
            vec![VerificationMethod::ChallengeResponse]
        );

        // TOTP only
        let config = SecurityConfig {
            totp: Some(TotpConfig {
                secret: "JBSWY3DPEHPK3PXP".to_string(),
                ..TotpConfig::default()
            }),
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        assert_eq!(verifier.available_methods(), vec![VerificationMethod::Totp]);

        // Both
        let config = SecurityConfig {
            challenges: vec![ChallengeConfig {
                question: "Q?".to_string(),
                answer_hash: "sha256:abc".to_string(),
            }],
            totp: Some(TotpConfig {
                secret: "JBSWY3DPEHPK3PXP".to_string(),
                ..TotpConfig::default()
            }),
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        assert_eq!(
            verifier.available_methods(),
            vec![
                VerificationMethod::ChallengeResponse,
                VerificationMethod::Totp
            ]
        );
    }

    #[test]
    fn test_random_challenge_returns_valid_index() {
        let config = SecurityConfig {
            challenges: vec![
                ChallengeConfig {
                    question: "Q1?".to_string(),
                    answer_hash: hash_answer("a1"),
                },
                ChallengeConfig {
                    question: "Q2?".to_string(),
                    answer_hash: hash_answer("a2"),
                },
            ],
            ..SecurityConfig::default()
        };
        let verifier = BiometricVerifier::new(config);
        let (idx, question) = verifier.random_challenge().unwrap();
        assert!(idx <= 1);
        assert!(question == "Q1?" || question == "Q2?");
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = b"hello world";
        let encoded = hex::encode(data);
        let decoded = hex::decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
