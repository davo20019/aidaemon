//! The turn runtime's own state: what one agent (and each child it spawns)
//! keeps for its turns, apart from the host's stores, tools and channels.
//!
//! The settings come from the host's configuration; the caches live as long
//! as the agent and are lost on restart (one re-prefill per restart).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::policy::trust_tier::{classify_model_trust, ModelTrustTier, TrustTierSetting};
use crate::prompt::core::CachedCore;
use crate::settings::{AudioConfig, ContextWindowConfig, PolicyConfig, SttConfig, VisionConfig};
use crate::turn::turn_render::RenderOptions;
use crate::turn::turn_render_cache::CachedRender;

/// Per-session core-prompt cache: `session_id` -> last rendered
/// [`CachedCore`]. On a hit the cached bytes are reused verbatim; on a miss
/// the changed component is logged and the entry replaced.
pub type CorePromptCache = Arc<RwLock<HashMap<String, CachedCore>>>;

/// Per-session, per-turn render cache: `session_id` -> (`turn_id` ->
/// [`CachedRender`]). Rendered turn bytes are reused verbatim while the
/// content fingerprint, renderer version and mode match.
pub type TurnRenderCache = Arc<RwLock<HashMap<String, HashMap<String, CachedRender>>>>;

/// Per-session anchor: `session_id` -> the `turn_seq` the archived region
/// carried into each payload starts at. Eviction advances it.
pub type TurnAnchorMemory = Arc<RwLock<HashMap<String, i64>>>;

/// Per-session conversation turn ids: set at the start of each turn so every
/// message written during it carries the same `turn_id`, which message
/// building reads to find the current task's boundary.
pub type CurrentTurnIds = Arc<RwLock<HashMap<String, String>>>;

/// The turn runtime's settings and caches for one agent.
pub struct TurnRuntime {
    /// Context window management.
    pub context_window_config: ContextWindowConfig,
    /// Policy rollout and enforcement.
    pub policy_config: PolicyConfig,
    /// What a model may be shown of received images.
    pub vision_config: VisionConfig,
    /// What a model may be shown of received audio.
    pub audio_config: AudioConfig,
    /// The speech-to-text fallback for audio a model cannot hear.
    pub stt_config: SttConfig,
    /// Record structured decision points in the event store.
    pub record_decision_points: bool,
    /// See [`CorePromptCache`].
    pub core_prompts: CorePromptCache,
    /// See [`TurnRenderCache`].
    pub turn_renders: TurnRenderCache,
    /// See [`TurnAnchorMemory`].
    pub turn_anchors: TurnAnchorMemory,
    /// See [`CurrentTurnIds`].
    pub current_turn_ids: CurrentTurnIds,
}

impl TurnRuntime {
    /// A runtime with these settings and empty caches.
    pub fn new(
        context_window_config: ContextWindowConfig,
        policy_config: PolicyConfig,
        vision_config: VisionConfig,
        audio_config: AudioConfig,
        stt_config: SttConfig,
        record_decision_points: bool,
    ) -> Self {
        Self {
            context_window_config,
            policy_config,
            vision_config,
            audio_config,
            stt_config,
            record_decision_points,
            core_prompts: Arc::default(),
            turn_renders: Arc::default(),
            turn_anchors: Arc::default(),
            current_turn_ids: Arc::default(),
        }
    }

    /// How turns render for `model`.
    pub fn render_options(&self, model: &str) -> RenderOptions {
        RenderOptions {
            vision: self.vision_config.clone(),
            audio: self.audio_config.clone(),
            stt: self.stt_config.clone(),
            model: model.to_string(),
        }
    }

    /// Trust tier for the model handling the current call, combining the
    /// `[policy] trust_tier` override with the model-id heuristic.
    pub fn trust_tier_for_model(&self, model: &str) -> ModelTrustTier {
        classify_model_trust(
            model,
            TrustTierSetting::parse(&self.policy_config.trust_tier),
        )
    }
}
