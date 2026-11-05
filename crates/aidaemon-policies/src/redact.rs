//! Secret redaction as an `after_tool` hook.

use std::sync::OnceLock;

use aidaemon_core::{Hook, HookContext, Receipt, ToolOutput};
use async_trait::async_trait;
use regex::Regex;

/// Literal secrets shorter than this are ignored: redacting every `abc`
/// would destroy output without protecting anything.
pub(crate) const MIN_LITERAL_LEN: usize = 6;

/// Replaces secrets in tool output before the model sees it.
///
/// Built-in patterns cover common credential *shapes*: OpenAI/Anthropic
/// style `sk-...` keys, GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`,
/// `ghr_`, `github_pat_`), AWS access key ids (`AKIA`/`ASIA`), Slack tokens
/// (`xox?-`), Google API keys (`AIza...`), `Bearer` tokens, PEM private
/// key blocks and database or broker URLs that carry a password. Add exact
/// values with [`Redactor::secret`] (for example the API keys from your own
/// config).
///
/// Only `ToolOutput::content` changes. Receipts are final before
/// `after_tool` runs, and this hook never edits them; keep secrets out of
/// receipt `details` at the tool level.
#[derive(Debug, Clone)]
pub struct Redactor {
    patterns: Vec<(&'static str, Regex)>,
    literals: Vec<String>,
}

impl Default for Redactor {
    fn default() -> Self {
        Self::new()
    }
}

/// The built-in secret shapes, as `(kind, pattern)`. [`Redactor`] hides
/// them in tool output and [`crate::EgressGuard`] refuses outbound
/// calls that carry them; both use this one list.
pub fn secret_patterns() -> &'static [(&'static str, Regex)] {
    static PATTERNS: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let raw: &[(&'static str, &str)] = &[
            (
                "private_key",
                r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            ),
            ("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{8,}=*"),
            ("api_key", r"\bsk-[A-Za-z0-9_\-]{20,}"),
            ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
            ("github_token", r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
            ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            ("slack_token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
            ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}"),
            // A database or broker URL that carries a password.
            (
                "connection_string",
                r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|rediss?|amqps?)://[^\s:/@]*:[^\s/@]+@[^\s]+",
            ),
        ];
        raw.iter()
            .map(|(kind, re)| (*kind, Regex::new(re).expect("built-in pattern compiles")))
            .collect()
    })
}

impl Redactor {
    pub fn new() -> Self {
        Self {
            patterns: secret_patterns().to_vec(),
            literals: Vec::new(),
        }
    }

    /// The `(kind, pattern)` sources in use, for tests.
    #[cfg(test)]
    pub(crate) fn pattern_sources(&self) -> Vec<(&'static str, &str)> {
        self.patterns
            .iter()
            .map(|(kind, re)| (*kind, re.as_str()))
            .collect()
    }

    /// Also redact this exact value wherever it appears.
    pub fn secret(mut self, value: impl Into<String>) -> Self {
        let value = value.into();
        if value.chars().count() >= MIN_LITERAL_LEN {
            self.literals.push(value);
            // Longest first, so a secret that contains another is fully hidden.
            self.literals.sort_by_key(|s| std::cmp::Reverse(s.len()));
        }
        self
    }

    /// Add a custom pattern with a label used in the placeholder.
    pub fn pattern(mut self, kind: &'static str, regex: Regex) -> Self {
        self.patterns.push((kind, regex));
        self
    }

    /// Redact `text`.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for literal in &self.literals {
            if out.contains(literal.as_str()) {
                out = out.replace(literal.as_str(), "[REDACTED:secret]");
            }
        }
        for (kind, re) in &self.patterns {
            if re.is_match(&out) {
                let placeholder = if *kind == "bearer_token" {
                    "Bearer [REDACTED:bearer_token]".to_string()
                } else {
                    format!("[REDACTED:{kind}]")
                };
                out = re.replace_all(&out, placeholder.as_str()).into_owned();
            }
        }
        out
    }
}

#[async_trait]
impl Hook for Redactor {
    async fn after_tool(
        &self,
        _ctx: &HookContext<'_>,
        _receipt: &Receipt,
        output: &mut ToolOutput,
    ) {
        output.content = self.redact(&output.content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_shapes_are_redacted() {
        let r = Redactor::new();
        let input = "OPENAI=sk-proj-abcdefghijklmnopqrstuvwx\n\
                     gh=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n\
                     aws=AKIAABCDEFGHIJKLMNOP\n\
                     Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig\n\
                     -----BEGIN RSA PRIVATE KEY-----\nMIIEow\nabc\n-----END RSA PRIVATE KEY-----\n\
                     plain text stays";
        let out = r.redact(input);
        for secret in [
            "sk-proj-abcdefghijklmnopqrstuvwx",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAABCDEFGHIJKLMNOP",
            "eyJhbGciOiJIUzI1NiJ9",
            "MIIEow",
        ] {
            assert!(!out.contains(secret), "{secret} leaked: {out}");
        }
        assert!(out.contains("[REDACTED:api_key]"));
        assert!(out.contains("[REDACTED:github_token]"));
        assert!(out.contains("[REDACTED:aws_access_key]"));
        assert!(out.contains("Bearer [REDACTED:bearer_token]"));
        assert!(out.contains("[REDACTED:private_key]"));
        assert!(out.contains("plain text stays"));
    }

    #[test]
    fn literal_secrets_are_redacted_and_short_ones_ignored() {
        let r = Redactor::new()
            .secret("hunter2-synthetic-password")
            .secret("abc");
        let out = r.redact("pw=hunter2-synthetic-password; abc stays");
        assert_eq!(out, "pw=[REDACTED:secret]; abc stays");
    }

    #[test]
    fn near_misses_are_left_alone() {
        let r = Redactor::new();
        let text = "task-sk-1 and the word bearer alone, AKIA123";
        assert_eq!(r.redact(text), text);
    }

    #[test]
    fn connection_strings_with_credentials_are_redacted() {
        let r = Redactor::new();
        let out = r.redact(
            "db=postgres://admin:synthetic-pass@db.example.invalid:5432/app and \
             cache=redis://:synthetic-pass@cache.example.invalid:6379",
        );
        assert!(!out.contains("synthetic-pass"), "{out}");
        assert!(out.contains("[REDACTED:connection_string]"), "{out}");
        // Without a password there is no credential to leak.
        let plain = "postgres://db.example.invalid:5432/app";
        assert_eq!(r.redact(plain), plain);
    }
}
