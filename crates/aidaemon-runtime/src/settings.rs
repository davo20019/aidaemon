//! Settings the runtime takes from its host, in place of the daemon's
//! config: the optional decision backend and context window management.
//! The daemon's config parses them under the same keys.

use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;

fn default_true() -> bool {
    true
}

/// `[decisions]`: an optional backend that answers small closed questions
/// (how a message relates to earlier ones, which recalled facts matter,
/// which model tier a request needs) with a confidence. Off by default;
/// with it off, or on any failure, the daemon behaves as without it.
#[derive(Deserialize, Clone, Default)]
pub struct DecisionsConfig {
    #[serde(default)]
    pub backend: DecisionBackendKind,
    /// For `backend = "provider"`: the model to ask; empty means the
    /// configured cheap (fast-tier) model. For `backend = "gemini"`: the
    /// decision model; empty means `gemini-decision-1`.
    #[serde(default)]
    pub model: String,
    /// For `backend = "gemini"`: the API key, or `"keychain"` to read
    /// `decisions_api_key` from the OS credential store. Empty means the key
    /// of the first `google_genai` provider in `[provider]`, if any.
    #[serde(default)]
    pub api_key: String,
    /// Per decision: `off`, `shadow` (ask and record, never act) or `on`.
    #[serde(default)]
    pub mode: DecisionModes,
}

impl fmt::Debug for DecisionsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionsConfig")
            .field("backend", &self.backend)
            .field("model", &self.model)
            .field(
                "api_key",
                &if self.api_key.is_empty() {
                    ""
                } else {
                    "[redacted]"
                },
            )
            .field("mode", &self.mode)
            .finish()
    }
}

/// Which decision backend answers.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum DecisionBackendKind {
    #[default]
    Off,
    /// The configured chat provider, asked for JSON. Its confidence is
    /// self-reported, so decisions that need an honest confidence stay off.
    Provider,
    /// Google's Decisions API (calibrated confidence).
    Gemini,
}

/// Whether a decision runs, and whether it may act.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum DecisionMode {
    #[default]
    Off,
    Shadow,
    On,
}

/// The mode of each decision. Unset decisions are shadowed when a backend
/// is configured, except model routing and the command second opinion,
/// which stay off until asked for.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub struct DecisionModes {
    #[serde(default = "decision_mode_shadow")]
    pub turn_digest: DecisionMode,
    #[serde(default = "decision_mode_shadow")]
    pub resume: DecisionMode,
    #[serde(default = "decision_mode_shadow")]
    pub recall_filter: DecisionMode,
    #[serde(default)]
    pub model_route: DecisionMode,
    #[serde(default)]
    pub command_risk_second: DecisionMode,
}

fn decision_mode_shadow() -> DecisionMode {
    DecisionMode::Shadow
}

impl Default for DecisionModes {
    fn default() -> Self {
        Self {
            turn_digest: DecisionMode::Shadow,
            resume: DecisionMode::Shadow,
            recall_filter: DecisionMode::Shadow,
            model_route: DecisionMode::Off,
            command_risk_second: DecisionMode::Off,
        }
    }
}

/// Context window management configuration.
/// Controls conversation history trimming, summarization, and progressive fact extraction.
#[derive(Debug, Deserialize, Clone)]
pub struct ContextWindowConfig {
    /// Enable context window management (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Context budget for a model whose family has no known window (default:
    /// 128000). Known hosted families use their own window; set
    /// `model_budgets` for a local or unlisted model.
    #[serde(default = "default_context_budget")]
    pub default_budget: usize,
    /// Per-model token budgets (model name → budget). Overrides default_budget.
    #[serde(default)]
    pub model_budgets: HashMap<String, usize>,
    /// Maximum characters for tool results before truncation (default: 16000).
    #[serde(default = "default_tool_result_chars")]
    pub max_tool_result_chars: usize,
    /// Per-model tool result character caps (model name → max chars).
    /// Overrides `max_tool_result_chars` for the named model. Models listed
    /// in `model_budgets` but not here get a cap scaled proportionally to
    /// their budget.
    #[serde(default)]
    pub model_tool_result_chars: HashMap<String, usize>,
    /// Legacy message-window setting retained for configuration compatibility.
    /// Runtime compaction now protects recent raw context by token budget.
    #[allow(dead_code)]
    #[serde(default = "default_summary_window")]
    pub summary_window: usize,
    /// Enable progressive fact extraction after each interaction (default: true).
    #[serde(default = "default_true")]
    pub progressive_facts: bool,
    /// Legacy message-count threshold retained for configuration compatibility.
    /// Runtime compaction uses `summarize_token_threshold` instead.
    #[allow(dead_code)]
    #[serde(default = "default_summarize_threshold")]
    pub summarize_threshold: usize,
    /// Cap on the estimated unsummarized conversation tokens that trigger
    /// compaction. Unset, compaction waits until the history fills 60% of the
    /// turn model's context window.
    #[serde(default = "default_summarize_token_threshold")]
    pub summarize_token_threshold: usize,
    /// Cap on the estimated tokens of recent whole turns kept verbatim after
    /// compaction. Unset, a quarter of the compaction threshold.
    #[serde(default = "default_summary_recent_tokens")]
    pub summary_recent_tokens: usize,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_budget: default_context_budget(),
            model_budgets: HashMap::new(),
            max_tool_result_chars: default_tool_result_chars(),
            model_tool_result_chars: HashMap::new(),
            summary_window: default_summary_window(),
            progressive_facts: true,
            summarize_threshold: default_summarize_threshold(),
            summarize_token_threshold: default_summarize_token_threshold(),
            summary_recent_tokens: default_summary_recent_tokens(),
        }
    }
}

impl ContextWindowConfig {
    /// Effective tool-result character cap for a specific model.
    ///
    /// Resolution order: explicit `model_tool_result_chars` entry → cap
    /// scaled by the model's `model_budgets` ratio against `default_budget`
    /// (floored so tiny budgets stay usable) → global `max_tool_result_chars`.
    pub fn tool_result_chars_for(&self, model: &str) -> usize {
        const MIN_TOOL_RESULT_CHARS: usize = 1_000;
        if let Some(&chars) = self.model_tool_result_chars.get(model) {
            return chars;
        }
        if let Some(&budget) = self.model_budgets.get(model) {
            if let Some(scaled) = self
                .max_tool_result_chars
                .saturating_mul(budget)
                .checked_div(self.default_budget)
            {
                return scaled.max(MIN_TOOL_RESULT_CHARS);
            }
        }
        self.max_tool_result_chars
    }

    /// The model's total context budget: an explicit `model_budgets` entry,
    /// else the model family's known window (or `default_budget`), lowered
    /// to what the provider reported if it rejected a request as too long.
    pub fn context_budget_for(&self, model: &str) -> usize {
        if let Some(budget) = self.model_budgets.get(model) {
            return *budget;
        }
        let assumed = known_context_budget(model).unwrap_or(self.default_budget);
        // A window the provider itself reported (it rejected a request as
        // too long) overrides the assumption when smaller.
        learned_context_budget(model).map_or(assumed, |learned| learned.min(assumed))
    }

    /// Token-pressure high-water mark for the model whose window the history
    /// fills (the turn's model, not the summarizer's): 60% of that window, or
    /// the configured cap when lower. Compaction runs after a reply, so a
    /// turn never waits for it below `emergency_compaction_threshold_for`.
    pub fn summarize_token_threshold_for(&self, model: &str) -> usize {
        self.summarize_token_threshold
            .min(aidaemon_policies::compaction_threshold(
                self.context_budget_for(model),
            ))
            .max(512)
    }

    /// History size at which a turn waits for compaction before its first
    /// call: 80% of the model's window, so only a history the background
    /// pass could not keep up with blocks a reply.
    pub fn emergency_compaction_threshold_for(&self, model: &str) -> usize {
        aidaemon_policies::wait_threshold(self.context_budget_for(model))
            .max(self.summarize_token_threshold_for(model))
    }

    /// Raw recent-context overlap, also clamped so a summary can make progress.
    pub fn summary_recent_tokens_for(&self, model: &str) -> usize {
        let threshold = self.summarize_token_threshold_for(model);
        let recent = if self.summary_recent_tokens == default_summary_recent_tokens() {
            aidaemon_policies::verbatim_tokens(threshold)
        } else {
            self.summary_recent_tokens.min(threshold / 2)
        };
        recent.max(256)
    }
}

/// Context budgets for hosted model families, by the model name's prefix
/// (a provider route such as `openrouter/` or `models/` is ignored). Each is
/// about 85% of the family's input window: the daemon's token count is an
/// estimate, and reasoning output shares the window on some APIs. GPT models
/// use the window the ChatGPT/Codex backend serves (272k), not the larger
/// public API one.
const KNOWN_CONTEXT_WINDOWS: &[(&str, usize)] = &[
    ("gpt-6", 272_000),
    ("gpt-5", 272_000),
    ("gpt-4.1", 1_000_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("claude-", 200_000),
    ("gemini-", 1_000_000),
    ("grok-4", 256_000),
];

/// Budgets learned this process from providers rejecting a request as too
/// long, by model.
fn learned_context_budgets() -> &'static std::sync::Mutex<HashMap<String, usize>> {
    static LEARNED: std::sync::OnceLock<std::sync::Mutex<HashMap<String, usize>>> =
        std::sync::OnceLock::new();
    LEARNED.get_or_init(Default::default)
}

fn learned_context_budget(model: &str) -> Option<usize> {
    learned_context_budgets()
        .lock()
        .ok()
        .and_then(|learned| learned.get(model).copied())
}

/// Record that `model` rejected a request of about `attempted_tokens` as
/// too long. The budget becomes 85% of the window the error stated, or 75%
/// of the attempt when it stated none; it only ever shrinks.
pub fn record_context_overflow(
    model: &str,
    stated_window: Option<usize>,
    attempted_tokens: usize,
) -> usize {
    let budget = stated_window
        .map(|window| window / 100 * 85)
        .unwrap_or(attempted_tokens / 4 * 3)
        .max(4_000);
    let mut learned = learned_context_budgets()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = learned.entry(model.to_string()).or_insert(budget);
    *entry = (*entry).min(budget);
    *entry
}

/// The budget for a model family listed in [`KNOWN_CONTEXT_WINDOWS`].
pub fn known_context_budget(model: &str) -> Option<usize> {
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    KNOWN_CONTEXT_WINDOWS
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix))
        .map(|(_, window)| window / 100 * 85)
}

fn default_context_budget() -> usize {
    128_000
}
fn default_tool_result_chars() -> usize {
    16_000
}
fn default_summary_window() -> usize {
    6
}
fn default_summarize_threshold() -> usize {
    12
}
// No cap: both follow the model's context window.
fn default_summarize_token_threshold() -> usize {
    usize::MAX
}
fn default_summary_recent_tokens() -> usize {
    usize::MAX
}
