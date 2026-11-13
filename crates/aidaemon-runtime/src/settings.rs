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
        if let Some(&budget) = self.model_budgets.get(model)
            && let Some(scaled) = self
                .max_tool_result_chars
                .saturating_mul(budget)
                .checked_div(self.default_budget)
        {
            return scaled.max(MIN_TOOL_RESULT_CHARS);
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

// ---- Received files: where they go, and what a model may be shown ----

#[derive(Debug, Deserialize, Clone)]
pub struct FilesConfig {
    #[serde(default = "default_files_enabled")]
    pub enabled: bool,
    #[serde(default = "default_inbox_dir")]
    pub inbox_dir: String,
    #[serde(default = "default_outbox_dirs")]
    pub outbox_dirs: Vec<String>,
    #[serde(default = "default_max_file_size_mb")]
    pub max_file_size_mb: u64,
    #[serde(default = "default_retention_hours")]
    pub retention_hours: u64,
    #[serde(default = "default_vision_enabled")]
    pub vision_enabled: bool,
    #[serde(default = "default_max_vision_image_mb")]
    pub max_vision_image_mb: u64,
    #[serde(default = "default_vision_mime_types")]
    pub vision_mime_types: Vec<String>,
    #[serde(default = "default_vision_model_patterns")]
    pub vision_model_patterns: Vec<String>,
    #[serde(default = "default_audio_enabled")]
    pub audio_enabled: bool,
    #[serde(default = "default_max_audio_mb")]
    pub max_audio_mb: u64,
    #[serde(default = "default_audio_mime_types")]
    pub audio_mime_types: Vec<String>,
    #[serde(default = "default_audio_model_patterns")]
    pub audio_model_patterns: Vec<String>,
    #[serde(default)]
    pub stt: SttFilesConfig,
}

/// Local Whisper STT settings under `[files.stt]`.
#[derive(Debug, Deserialize, Clone)]
pub struct SttFilesConfig {
    #[serde(default = "default_stt_enabled")]
    pub enabled: bool,
    #[serde(default = "default_stt_cli_path")]
    pub cli_path: String,
    #[serde(default = "default_stt_model_path")]
    pub model_path: String,
    #[serde(default = "default_stt_ffmpeg_path")]
    pub ffmpeg_path: String,
    #[serde(default = "default_stt_language")]
    pub language: String,
    #[serde(default = "default_stt_max_audio_mb")]
    pub max_audio_mb: u64,
    #[serde(default = "default_stt_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_stt_mime_types")]
    pub mime_types: Vec<String>,
}

/// Vision/image encoding settings derived from `[files]` config.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub enabled: bool,
    pub max_image_bytes: u64,
    pub mime_types: Vec<String>,
    pub model_patterns: Vec<String>,
}

impl VisionConfig {
    pub fn from_files(files: &FilesConfig) -> Self {
        Self {
            enabled: files.vision_enabled,
            max_image_bytes: files.max_vision_image_mb.saturating_mul(1_048_576),
            mime_types: files.vision_mime_types.clone(),
            model_patterns: files.vision_model_patterns.clone(),
        }
    }

    pub fn model_supports_vision(&self, model: &str) -> bool {
        if !self.enabled {
            return false;
        }
        let model_lower = model.to_ascii_lowercase();
        self.model_patterns
            .iter()
            .any(|pattern| model_lower.contains(&pattern.to_ascii_lowercase()))
    }

    pub fn mime_allowed(&self, mime: &str) -> bool {
        self.mime_types.iter().any(|allowed| allowed == mime)
    }
}

/// Native audio input settings derived from `[files]` config.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub enabled: bool,
    pub max_audio_bytes: u64,
    pub mime_types: Vec<String>,
    pub model_patterns: Vec<String>,
}

impl AudioConfig {
    pub fn from_files(files: &FilesConfig) -> Self {
        Self {
            enabled: files.audio_enabled,
            max_audio_bytes: files.max_audio_mb.saturating_mul(1_048_576),
            mime_types: files.audio_mime_types.clone(),
            model_patterns: files.audio_model_patterns.clone(),
        }
    }

    pub fn mime_allowed(&self, mime: &str) -> bool {
        self.mime_types.iter().any(|allowed| allowed == mime)
    }

    pub fn model_supports_audio(&self, model: &str) -> bool {
        let model_lower = model.to_ascii_lowercase();
        self.model_patterns
            .iter()
            .any(|pattern| model_lower.contains(&pattern.to_ascii_lowercase()))
    }
}

/// Resolved Whisper STT settings for runtime use.
#[derive(Debug, Clone)]
pub struct SttConfig {
    pub enabled: bool,
    pub cli_path: std::path::PathBuf,
    pub model_path: std::path::PathBuf,
    pub ffmpeg_path: std::path::PathBuf,
    pub language: String,
    pub max_audio_bytes: u64,
    pub timeout_secs: u64,
    pub mime_types: Vec<String>,
}

impl SttConfig {
    pub fn from_files(files: &FilesConfig) -> Self {
        Self {
            enabled: files.stt.enabled,
            cli_path: std::path::PathBuf::from(
                shellexpand::tilde(&files.stt.cli_path).into_owned(),
            ),
            model_path: std::path::PathBuf::from(
                shellexpand::tilde(&files.stt.model_path).into_owned(),
            ),
            ffmpeg_path: std::path::PathBuf::from(
                shellexpand::tilde(&files.stt.ffmpeg_path).into_owned(),
            ),
            language: files.stt.language.clone(),
            max_audio_bytes: files.stt.max_audio_mb.saturating_mul(1_048_576),
            timeout_secs: files.stt.timeout_secs,
            mime_types: files.stt.mime_types.clone(),
        }
    }

    pub fn mime_allowed(&self, mime: &str) -> bool {
        self.mime_types.iter().any(|allowed| allowed == mime)
    }
}

impl Default for SttFilesConfig {
    fn default() -> Self {
        Self {
            enabled: default_stt_enabled(),
            cli_path: default_stt_cli_path(),
            model_path: default_stt_model_path(),
            ffmpeg_path: default_stt_ffmpeg_path(),
            language: default_stt_language(),
            max_audio_mb: default_stt_max_audio_mb(),
            timeout_secs: default_stt_timeout_secs(),
            mime_types: default_stt_mime_types(),
        }
    }
}

impl Default for FilesConfig {
    fn default() -> Self {
        Self {
            enabled: default_files_enabled(),
            inbox_dir: default_inbox_dir(),
            outbox_dirs: default_outbox_dirs(),
            max_file_size_mb: default_max_file_size_mb(),
            retention_hours: default_retention_hours(),
            vision_enabled: default_vision_enabled(),
            max_vision_image_mb: default_max_vision_image_mb(),
            vision_mime_types: default_vision_mime_types(),
            vision_model_patterns: default_vision_model_patterns(),
            audio_enabled: default_audio_enabled(),
            max_audio_mb: default_max_audio_mb(),
            audio_mime_types: default_audio_mime_types(),
            audio_model_patterns: default_audio_model_patterns(),
            stt: SttFilesConfig::default(),
        }
    }
}

fn default_files_enabled() -> bool {
    true
}
fn default_inbox_dir() -> String {
    "~/.aidaemon/files/inbox".to_string()
}
fn default_outbox_dirs() -> Vec<String> {
    vec!["~".to_string()]
}
fn default_max_file_size_mb() -> u64 {
    10
}
fn default_retention_hours() -> u64 {
    24
}
fn default_vision_enabled() -> bool {
    true
}
fn default_max_vision_image_mb() -> u64 {
    4
}
fn default_vision_mime_types() -> Vec<String> {
    vec![
        "image/jpeg".to_string(),
        "image/png".to_string(),
        "image/gif".to_string(),
        "image/webp".to_string(),
    ]
}
fn default_vision_model_patterns() -> Vec<String> {
    vec![
        "gpt-4o".to_string(),
        "gpt-4".to_string(),
        "gpt-5".to_string(),
        "gpt-6".to_string(),
        "o1".to_string(),
        "o3".to_string(),
        "o4".to_string(),
        "gemini".to_string(),
        "claude-3".to_string(),
        "claude-sonnet".to_string(),
        "claude-opus".to_string(),
        "claude-haiku".to_string(),
        "fable".to_string(),
        "gemma".to_string(),
        "llava".to_string(),
        "qwen-vl".to_string(),
        "qwen2-vl".to_string(),
        "qwen2.5-vl".to_string(),
        "qwen3-vl".to_string(),
        "internvl".to_string(),
        "vision".to_string(),
        "pixtral".to_string(),
        "mistral-large".to_string(),
    ]
}
fn default_audio_enabled() -> bool {
    true
}
fn default_max_audio_mb() -> u64 {
    10
}
fn default_audio_mime_types() -> Vec<String> {
    vec![
        "audio/ogg".to_string(),
        "audio/mpeg".to_string(),
        "audio/mp3".to_string(),
        "audio/wav".to_string(),
        "audio/flac".to_string(),
        "audio/aac".to_string(),
    ]
}
fn default_audio_model_patterns() -> Vec<String> {
    vec![
        "audio".to_string(),
        "gemini-2".to_string(),
        "gemini-3".to_string(),
    ]
}
fn default_stt_enabled() -> bool {
    false
}
fn default_stt_cli_path() -> String {
    "/opt/homebrew/bin/whisper-cli".to_string()
}
fn default_stt_model_path() -> String {
    "~/models/whisper/ggml-medium.en.bin".to_string()
}
fn default_stt_ffmpeg_path() -> String {
    "ffmpeg".to_string()
}
fn default_stt_language() -> String {
    "en".to_string()
}
fn default_stt_max_audio_mb() -> u64 {
    25
}
fn default_stt_timeout_secs() -> u64 {
    120
}
fn default_stt_mime_types() -> Vec<String> {
    vec![
        "audio/ogg".to_string(),
        "audio/mpeg".to_string(),
        "audio/mp3".to_string(),
        "audio/wav".to_string(),
        "audio/flac".to_string(),
        "audio/aac".to_string(),
        "audio/webm".to_string(),
    ]
}

// ---- Turn policy rollout and enforcement (`[policy]`) ----

#[derive(Debug, Deserialize, Clone)]
pub struct WriteConsistencyConfig {
    #[serde(default = "default_write_consistency_max_abs_global_delta")]
    pub max_abs_global_delta: u64,
    #[serde(default = "default_write_consistency_max_session_mismatch_count")]
    pub max_session_mismatch_count: u64,
    #[serde(default = "default_write_consistency_max_stale_task_starts")]
    pub max_stale_task_starts: u64,
    #[serde(default = "default_write_consistency_max_missing_message_id_events")]
    pub max_missing_message_id_events: u64,
}

impl Default for WriteConsistencyConfig {
    fn default() -> Self {
        Self {
            max_abs_global_delta: default_write_consistency_max_abs_global_delta(),
            max_session_mismatch_count: default_write_consistency_max_session_mismatch_count(),
            max_stale_task_starts: default_write_consistency_max_stale_task_starts(),
            max_missing_message_id_events: default_write_consistency_max_missing_message_id_events(
            ),
        }
    }
}

impl WriteConsistencyConfig {
    pub fn thresholds(&self) -> aidaemon_events::WriteConsistencyThresholds {
        aidaemon_events::WriteConsistencyThresholds {
            max_abs_global_delta: self.max_abs_global_delta,
            max_session_mismatch_count: self.max_session_mismatch_count,
            max_stale_task_starts: self.max_stale_task_starts,
            max_missing_message_id_events: self.max_missing_message_id_events,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct PolicyConfig {
    #[serde(default = "default_policy_shadow_mode")]
    pub policy_shadow_mode: bool,
    /// Enforce policy-derived context budget caps. When false, message trimming
    /// only respects the model provider budget.
    #[serde(default = "default_policy_enforce")]
    pub policy_enforce: bool,
    #[serde(default = "default_tool_filter_enforce")]
    pub tool_filter_enforce: bool,
    #[serde(default = "default_uncertainty_clarify_enforce")]
    pub uncertainty_clarify_enforce: bool,
    #[serde(default = "default_context_refresh_enforce")]
    pub context_refresh_enforce: bool,
    #[serde(default = "default_learning_evidence_gate_enforce")]
    pub learning_evidence_gate_enforce: bool,
    #[serde(default = "default_autotune_shadow")]
    pub autotune_shadow: bool,
    #[serde(default = "default_autotune_enforce")]
    pub autotune_enforce: bool,
    #[serde(default = "default_uncertainty_threshold")]
    pub uncertainty_clarify_threshold: f32,
    /// Model trust tier override: "auto" (classify by model id), "guided"
    /// (always enforce full supervision scaffolding), or "autonomous"
    /// (supervision gates become telemetry-only). Default: "auto".
    #[serde(default = "default_trust_tier")]
    pub trust_tier: String,
    /// Opt-in auxiliary supervision calls inside the tool loop for Guided-tier
    /// models: the pre-execution plan, the pre-execution critique, and the
    /// per-step re-planner. Each is an extra model call that can bounce the
    /// proposed tool call into another full loop iteration. Live autonomy
    /// traces showed these calls cost more failed finalizations (checklist
    /// tools displacing the required operation, validation budgets consumed by
    /// re-plans) than they prevented, so they are off by default. Anti-
    /// fabrication guards, approval gates, and hard caps are unaffected.
    #[serde(default = "default_pre_execution_supervision")]
    pub pre_execution_supervision: bool,
    #[serde(default)]
    pub write_consistency: WriteConsistencyConfig,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            policy_shadow_mode: default_policy_shadow_mode(),
            policy_enforce: default_policy_enforce(),
            tool_filter_enforce: default_tool_filter_enforce(),
            uncertainty_clarify_enforce: default_uncertainty_clarify_enforce(),
            context_refresh_enforce: default_context_refresh_enforce(),
            learning_evidence_gate_enforce: default_learning_evidence_gate_enforce(),
            autotune_shadow: default_autotune_shadow(),
            autotune_enforce: default_autotune_enforce(),
            uncertainty_clarify_threshold: default_uncertainty_threshold(),
            trust_tier: default_trust_tier(),
            pre_execution_supervision: default_pre_execution_supervision(),
            write_consistency: WriteConsistencyConfig::default(),
        }
    }
}

fn default_policy_shadow_mode() -> bool {
    true
}
fn default_policy_enforce() -> bool {
    true
}
fn default_tool_filter_enforce() -> bool {
    true
}
fn default_uncertainty_clarify_enforce() -> bool {
    false
}
fn default_context_refresh_enforce() -> bool {
    true
}
fn default_learning_evidence_gate_enforce() -> bool {
    true
}
fn default_autotune_shadow() -> bool {
    true
}
fn default_autotune_enforce() -> bool {
    true
}
fn default_uncertainty_threshold() -> f32 {
    0.55
}
fn default_trust_tier() -> String {
    "auto".to_string()
}
fn default_pre_execution_supervision() -> bool {
    false
}
fn default_write_consistency_max_abs_global_delta() -> u64 {
    3
}
fn default_write_consistency_max_session_mismatch_count() -> u64 {
    0
}
fn default_write_consistency_max_stale_task_starts() -> u64 {
    0
}
fn default_write_consistency_max_missing_message_id_events() -> u64 {
    0
}

/// `[computer_use]`: whether the desktop tool is on, and its limits.
#[derive(Debug, Deserialize, Clone)]
pub struct ComputerUseConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cu_screenshot_max_width")]
    pub screenshot_max_width: u32,
    #[serde(default = "default_cu_screenshot_max_height")]
    pub screenshot_max_height: u32,
    #[serde(default = "default_cu_screenshot_max_bytes")]
    pub screenshot_max_bytes: u64,
    #[serde(default = "default_cu_ax_max_depth")]
    pub ax_max_depth: u32,
    #[serde(default = "default_cu_ax_max_nodes")]
    pub ax_max_nodes: u32,
    #[serde(default = "default_cu_action_timeout_secs")]
    pub action_timeout_secs: u64,
    #[serde(default = "default_cu_max_mutating_actions")]
    pub max_mutating_actions: u32,
    #[serde(default = "default_cu_max_consecutive_observations")]
    pub max_consecutive_observations: u32,
    #[serde(default = "default_cu_max_wall_clock_secs")]
    pub max_wall_clock_secs: u64,
    #[serde(default)]
    pub always_allowed_apps: Vec<String>,
    #[serde(default)]
    pub mirror_screenshots_to_channel: bool,
}

fn default_cu_screenshot_max_width() -> u32 {
    1280
}
fn default_cu_screenshot_max_height() -> u32 {
    800
}
fn default_cu_screenshot_max_bytes() -> u64 {
    2_000_000
}
fn default_cu_ax_max_depth() -> u32 {
    12
}
fn default_cu_ax_max_nodes() -> u32 {
    500
}
fn default_cu_action_timeout_secs() -> u64 {
    10
}
fn default_cu_max_mutating_actions() -> u32 {
    40
}
fn default_cu_max_consecutive_observations() -> u32 {
    3
}
fn default_cu_max_wall_clock_secs() -> u64 {
    600
}

impl Default for ComputerUseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            screenshot_max_width: default_cu_screenshot_max_width(),
            screenshot_max_height: default_cu_screenshot_max_height(),
            screenshot_max_bytes: default_cu_screenshot_max_bytes(),
            ax_max_depth: default_cu_ax_max_depth(),
            ax_max_nodes: default_cu_ax_max_nodes(),
            action_timeout_secs: default_cu_action_timeout_secs(),
            max_mutating_actions: default_cu_max_mutating_actions(),
            max_consecutive_observations: default_cu_max_consecutive_observations(),
            max_wall_clock_secs: default_cu_max_wall_clock_secs(),
            always_allowed_apps: Vec::new(),
            mirror_screenshots_to_channel: false,
        }
    }
}

impl ComputerUseConfig {
    pub fn action_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.action_timeout_secs.clamp(1, 120))
    }
}

/// `[path_aliases]`: alias roots for "projects/..." paths.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PathAliasConfig {
    /// Optional alias roots for "projects/..." paths.
    /// Example:
    /// [path_aliases]
    /// projects = ["~/projects"]
    #[serde(default)]
    pub projects: Vec<String>,
}

/// Iteration limit configuration for agent loops.
/// Controls how iteration limits are enforced (or not enforced).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum IterationLimitConfig {
    /// No iteration limit - run until natural completion, token budget, or timeout.
    /// This is the recommended default for complex tasks.
    #[default]
    Unlimited,
    /// Soft limit with warnings but no forced termination.
    /// Agent will be warned at `warn_at` iterations and continue until natural completion.
    Soft { threshold: usize, warn_at: usize },
    /// Legacy hard limit behavior (backward compatibility).
    /// Agent will be forced to stop at `cap` iterations.
    Hard { initial: usize, cap: usize },
}
