//! Claiming a bot that has no owner yet. A bot connected without an
//! allowlist used to make whoever messaged it first its owner, so anyone who
//! found its username before the owner did owned the daemon. Now the first
//! direct message must carry a one-time code that only the surface that
//! connected the bot shows (the chat that ran `/connect`, the setup screen,
//! or the daemon's log for a bot from the config file).

use std::sync::Mutex;

/// The key a dynamic bot's `extra_config` keeps its claim code under.
pub const CLAIM_CODE_KEY: &str = "owner_claim_code";

/// The command word a claim may start with (`/claim <code>`); the bare code
/// is accepted too.
const CLAIM_COMMAND: &str = "/claim";

/// Wrong codes before the code is replaced (and announced again), so a
/// stranger cannot keep guessing at one code.
const MAX_WRONG_ATTEMPTS: u32 = 5;

/// A fresh claim code: 12 characters from a random UUID (48 bits), upper
/// case, in groups of four for reading aloud.
pub fn new_claim_code() -> String {
    let raw = uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .to_ascii_uppercase();
    let raw = &raw[..12];
    format!("{}-{}-{}", &raw[..4], &raw[4..8], &raw[8..12])
}

/// The claim code a dynamic bot's `extra_config` JSON stores, if any.
pub fn claim_code_from_extra(extra_config: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(extra_config)
        .ok()?
        .get(CLAIM_CODE_KEY)?
        .as_str()
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(str::to_string)
}

/// `extra_config` with `code` stored under [`CLAIM_CODE_KEY`].
pub fn extra_with_claim_code(extra_config: &str, code: &str) -> String {
    let mut value = serde_json::from_str::<serde_json::Value>(extra_config)
        .ok()
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    value[CLAIM_CODE_KEY] = serde_json::Value::String(code.to_string());
    value.to_string()
}

/// What one direct message to an unowned bot did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimAttempt {
    /// It carried the code: the sender is the owner now.
    Claimed,
    /// It did not; the bot stays unowned.
    Refused,
}

/// The one-time code that claims one unowned bot.
pub struct OwnerClaim {
    platform: &'static str,
    state: Mutex<ClaimState>,
}

struct ClaimState {
    code: String,
    wrong: u32,
}

impl OwnerClaim {
    /// A claim with `code` (a dynamic bot's stored code), or a fresh one.
    pub fn new(platform: &'static str, code: Option<String>) -> Self {
        Self {
            platform,
            state: Mutex::new(ClaimState {
                code: code.unwrap_or_else(new_claim_code),
                wrong: 0,
            }),
        }
    }

    /// The current code.
    pub fn code(&self) -> String {
        self.lock().code.clone()
    }

    /// Replaces the code (a connect flow that showed the owner its own).
    pub fn set_code(&self, code: &str) {
        let mut state = self.lock();
        state.code = code.trim().to_string();
        state.wrong = 0;
    }

    /// Writes the code to the daemon's log, where the owner of a bot from
    /// the config file reads it.
    pub fn announce(&self) {
        tracing::warn!(
            platform = self.platform,
            "This {} bot has no owner yet. To claim it, send it a direct message: {} {}",
            self.platform,
            CLAIM_COMMAND,
            self.code()
        );
    }

    /// Judges one direct message to the unowned bot. After
    /// five wrong codes a new code replaces the old one
    /// and is announced in the log.
    pub fn attempt(&self, text: &str) -> ClaimAttempt {
        let presented = presented_code(text);
        let mut state = self.lock();
        if presented.is_some_and(|presented| same_code(presented, &state.code)) {
            return ClaimAttempt::Claimed;
        }
        state.wrong += 1;
        if state.wrong >= MAX_WRONG_ATTEMPTS {
            state.code = new_claim_code();
            state.wrong = 0;
            drop(state);
            self.announce();
        }
        ClaimAttempt::Refused
    }

    /// The reply to a direct message that did not claim the bot. It never
    /// shows the code.
    pub fn refusal(&self) -> String {
        format!(
            "This bot has no owner yet. If you connected it, send the claim code \
             shown where you connected it: {CLAIM_COMMAND} <code>"
        )
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ClaimState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The code a message presents: `/claim <code>` (an exact command word) or
/// the message alone.
fn presented_code(text: &str) -> Option<&str> {
    let text = text.trim();
    let code = match text.strip_prefix(CLAIM_COMMAND) {
        Some(rest) if rest.is_empty() || rest.starts_with(char::is_whitespace) => rest.trim(),
        Some(_) => return None,
        None => text,
    };
    (!code.is_empty()).then_some(code)
}

/// Case-insensitive compare in time that does not depend on where the two
/// first differ.
fn same_code(presented: &str, code: &str) -> bool {
    let (a, b) = (presented.as_bytes(), code.as_bytes());
    let mut diff = a.len() ^ b.len();
    for (i, byte) in b.iter().enumerate() {
        let other = a.get(i).copied().unwrap_or(0).to_ascii_uppercase();
        diff |= usize::from(other ^ byte.to_ascii_uppercase());
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_code_claims() {
        let claim = OwnerClaim::new("telegram", Some("ABCD-EF12-3456".into()));
        assert_eq!(claim.attempt("hello"), ClaimAttempt::Refused);
        assert_eq!(
            claim.attempt("/claim abcd-ef12-3456"),
            ClaimAttempt::Claimed
        );
        assert_eq!(claim.attempt("  ABCD-EF12-3456 "), ClaimAttempt::Claimed);
        assert_eq!(claim.attempt("/claimABCD-EF12-3456"), ClaimAttempt::Refused);
        assert_eq!(claim.attempt("ABCD-EF12-345"), ClaimAttempt::Refused);
        assert!(!claim.refusal().contains("ABCD"));
    }

    #[test]
    fn wrong_guesses_replace_the_code() {
        let claim = OwnerClaim::new("slack", Some("ABCD-EF12-3456".into()));
        for _ in 0..MAX_WRONG_ATTEMPTS {
            assert_eq!(
                claim.attempt("/claim 0000-0000-0000"),
                ClaimAttempt::Refused
            );
        }
        assert_ne!(claim.code(), "ABCD-EF12-3456");
        assert_eq!(
            claim.attempt("/claim ABCD-EF12-3456"),
            ClaimAttempt::Refused
        );
        let fresh = claim.code();
        assert_eq!(claim.attempt(&fresh), ClaimAttempt::Claimed);
    }

    #[test]
    fn codes_round_trip_through_extra_config() {
        let code = new_claim_code();
        assert_eq!(code.len(), 14);
        let extra = extra_with_claim_code(r#"{"username":"synthetic_bot"}"#, &code);
        assert_eq!(claim_code_from_extra(&extra), Some(code));
        assert!(extra.contains("synthetic_bot"));
        assert_eq!(claim_code_from_extra("{}"), None);
        assert_eq!(claim_code_from_extra("not json"), None);
    }
}
