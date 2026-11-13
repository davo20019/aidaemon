//! Secret redaction at the kernel boundary.
//!
//! Every value registered here ([`register_secret`], which
//! `aidaemon_secrets` calls for each secret it hands out) and everything a
//! hook's [`Redact`] recognizes (the policies crate's `Redactor`: credential
//! shapes and literals) is replaced with `[REDACTED:...]` in what leaves the
//! kernel:
//!
//! - every model request (the whole transcript, after `before_model` hooks);
//! - every event, before sinks and the store see it;
//! - receipts: arguments, details and reason;
//! - the interruption put to the owner (call arguments, reason, request);
//! - the transcript and receipts in the saved and returned [`crate::RunState`],
//!   and the run's final text.
//!
//! What it does not cover: the exact arguments of a call that has not run
//! yet (a paused call in `RunState::pending`, a batch in flight) stay as the
//! model sent them so the call can run; a secret shorter than
//! [`MIN_SECRET_CHARS`] (refused, since redacting it would garble ordinary
//! text); encodings of a secret (hex, base64) or a secret split across
//! streamed text deltas.

use std::borrow::Cow;
use std::sync::{OnceLock, RwLock};

/// Secrets shorter than this are refused: redacting every `abc` would
/// destroy output without protecting anything.
pub const MIN_SECRET_CHARS: usize = 6;

/// A secret too short to redact safely.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "a secret of {chars} characters is too short to redact (the minimum is {MIN_SECRET_CHARS}); \
     it will appear as is wherever it is used"
)]
pub struct SecretTooShort {
    pub chars: usize,
}

fn registry() -> &'static RwLock<Vec<String>> {
    static REGISTRY: OnceLock<RwLock<Vec<String>>> = OnceLock::new();
    REGISTRY.get_or_init(RwLock::default)
}

/// Redact `value` from everything every run in this process sends out.
/// Errors (and registers nothing) for a value under [`MIN_SECRET_CHARS`].
pub fn register_secret(value: &str) -> Result<(), SecretTooShort> {
    let chars = value.chars().count();
    if chars < MIN_SECRET_CHARS {
        return Err(SecretTooShort { chars });
    }
    let mut secrets = registry()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !secrets.iter().any(|known| known == value) {
        secrets.push(value.to_string());
        // Longest first, so a secret that contains another is fully hidden.
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
    }
    Ok(())
}

/// `text` with every registered secret replaced by `[REDACTED:secret]`.
pub fn redact_registered(text: &str) -> Cow<'_, str> {
    let secrets = registry()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut out = Cow::Borrowed(text);
    for secret in secrets.iter() {
        if out.contains(secret.as_str()) {
            out = Cow::Owned(out.replace(secret.as_str(), "[REDACTED:secret]"));
        }
    }
    out
}

/// Something that hides secrets in text. A hook offers one through
/// [`crate::Hook::redactor`], and the kernel applies it at the boundary.
pub trait Redact: Send + Sync {
    fn redact(&self, text: &str) -> String;
}

/// The kernel's one redaction pass: the registry, then every hook's
/// [`Redact`].
pub(crate) struct Scrubber<'a> {
    pub hooks: &'a [std::sync::Arc<dyn crate::Hook>],
}

impl Scrubber<'_> {
    /// `text` redacted, or `None` when nothing changed.
    fn changed(&self, text: &str) -> Option<String> {
        let mut out = match redact_registered(text) {
            Cow::Borrowed(_) => None,
            Cow::Owned(owned) => Some(owned),
        };
        for hook in self.hooks {
            if let Some(redactor) = hook.redactor() {
                let current = out.as_deref().unwrap_or(text);
                let redacted = redactor.redact(current);
                if redacted != current {
                    out = Some(redacted);
                }
            }
        }
        out
    }

    pub fn text(&self, text: &mut String) {
        if let Some(redacted) = self.changed(text) {
            *text = redacted;
        }
    }

    /// Redact every string in `value`. Returns whether anything changed.
    pub fn value(&self, value: &mut serde_json::Value) -> bool {
        use serde_json::Value;
        match value {
            Value::String(text) => match self.changed(text) {
                Some(redacted) => {
                    *text = redacted;
                    true
                }
                None => false,
            },
            Value::Array(items) => items
                .iter_mut()
                .fold(false, |changed, item| self.value(item) | changed),
            Value::Object(map) => map
                .values_mut()
                .fold(false, |changed, item| self.value(item) | changed),
            _ => false,
        }
    }

    /// A message's text and its tool calls' arguments (not its images).
    pub fn message(&self, message: &mut crate::Message) {
        self.text(&mut message.content);
        for call in &mut message.tool_calls {
            self.value(&mut call.arguments);
        }
        if let Some(reasoning) = &mut message.reasoning {
            self.text(reasoning);
        }
    }

    pub fn receipt(&self, receipt: &mut crate::Receipt) {
        self.value(&mut receipt.arguments);
        self.value(&mut receipt.details);
        if let Some(reason) = &mut receipt.reason {
            self.text(reason);
        }
    }

    /// Any serializable value, through its JSON form.
    pub fn json<T>(&self, item: &mut T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let Ok(mut value) = serde_json::to_value(&*item) else {
            return;
        };
        if self.value(&mut value)
            && let Ok(redacted) = serde_json::from_value(value) {
                *item = redacted;
            }
    }
}
