//! Plain data types shared by stores, scorers and writers.

use aidaemon_core::ChannelVisibility;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, Result};
use crate::records::FactPrivacy;
use crate::sqlite::{memory_visible, ExactChannelIds, FACT_RESTATEMENT_MIN_SIMILARITY};

/// Longest fact value accepted, in characters.
pub const MAX_FACT_VALUE_CHARS: usize = 4_000;

/// Where a fact came from. Every field is optional so manual imports work,
/// but writers in this crate fill in what they know.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The agent run that produced the fact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// The session (conversation) it was stated in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The snippet of conversation that supports the fact. Facts with an
    /// excerpt score slightly higher (the provenance signal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
}

/// A durable piece of knowledge, keyed by `(scope, category, key)`.
///
/// At most one row per key is active; replaced values stay as superseded
/// history and are never recalled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    /// Whose memory this is: a user, session or agent id (see
    /// [`crate::Memory::scope_for`]).
    pub scope: String,
    pub category: String,
    pub key: String,
    pub value: String,
    /// 0.0-1.0, how sure the writer was.
    pub confidence: f32,
    pub source: Provenance,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub recall_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
}

impl Fact {
    pub fn is_active(&self) -> bool {
        self.superseded_at.is_none()
    }

    /// Text used for embeddings and lexical search: category, readable key
    /// and value.
    pub fn search_text(&self) -> String {
        fact_search_text(&self.category, &self.key, &self.value)
    }
}

/// `"{category} {key words}: {value}"`, shared by every store so lexical and
/// semantic signals see the same text.
pub fn fact_search_text(category: &str, key: &str, value: &str) -> String {
    format!("{} {}: {}", category, key.replace(['_', '-'], " "), value)
}

/// A fact to write. Normalized and validated by [`NewFact::normalized`]
/// before it reaches a store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewFact {
    pub scope: String,
    pub category: String,
    pub key: String,
    pub value: String,
    pub confidence: f32,
    #[serde(default)]
    pub source: Provenance,
}

impl NewFact {
    pub fn new(
        scope: impl Into<String>,
        category: impl Into<String>,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            scope: scope.into(),
            category: category.into(),
            key: key.into(),
            value: value.into(),
            confidence: 1.0,
            source: Provenance::default(),
        }
    }

    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn source(mut self, source: Provenance) -> Self {
        self.source = source;
        self
    }

    /// Trim and canonicalize the identifying fields (lowercase category,
    /// snake_case key), clamp confidence, and reject empty or oversized input.
    pub fn normalized(mut self) -> Result<Self> {
        self.scope = self.scope.trim().to_string();
        self.category = normalize_category(&self.category);
        self.key = normalize_key(&self.key);
        self.value = self.value.trim().to_string();
        if self.scope.is_empty() {
            return Err(MemoryError::Invalid("scope is empty".into()));
        }
        if self.category.is_empty() || self.key.is_empty() || self.value.is_empty() {
            return Err(MemoryError::Invalid(
                "category, key and value must be non-empty".into(),
            ));
        }
        if self.value.chars().count() > MAX_FACT_VALUE_CHARS {
            return Err(MemoryError::Invalid(format!(
                "value longer than {MAX_FACT_VALUE_CHARS} characters"
            )));
        }
        self.confidence = if self.confidence.is_finite() {
            self.confidence.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.source.excerpt = self
            .source
            .excerpt
            .map(|e| truncate_chars(e.trim(), 500))
            .filter(|e| !e.is_empty());
        Ok(self)
    }
}

/// Lowercase and trim.
pub fn normalize_category(category: &str) -> String {
    category.trim().to_lowercase()
}

/// Lowercase snake_case: runs of whitespace, `-` and `.` become `_`.
pub fn normalize_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.trim().to_lowercase().chars() {
        if c.is_whitespace() || c == '-' || c == '.' {
            if !out.ends_with('_') {
                out.push('_');
            }
        } else {
            out.push(c);
        }
    }
    out.trim_matches('_').to_string()
}

/// True when two values state the same thing modulo case and whitespace.
/// Used to tell a restatement (reinforce) from a correction (supersede).
pub fn same_value(a: &str, b: &str) -> bool {
    let canon = |s: &str| {
        s.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    canon(a) == canon(b)
}

/// What an upsert did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum UpsertOutcome {
    /// No active fact had this key.
    Inserted { id: String },
    /// The same value was already active; `updated_at` and confidence were
    /// refreshed.
    Reinforced { id: String },
    /// A different value was active; it was marked superseded and kept as
    /// history.
    Superseded {
        id: String,
        previous_id: String,
        previous_value: String,
    },
}

impl UpsertOutcome {
    /// Id of the now-active fact.
    pub fn id(&self) -> &str {
        match self {
            Self::Inserted { id } | Self::Reinforced { id } | Self::Superseded { id, .. } => id,
        }
    }
}

/// A summary of a finished run or session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub scope: String,
    pub session_id: String,
    pub summary: String,
    #[serde(default)]
    pub topics: Vec<String>,
    /// 0.1-1.0, see [`crate::episode_importance`].
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub recall_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
}

impl Episode {
    /// Text used for embeddings and lexical search.
    pub fn search_text(&self) -> String {
        episode_search_text(&self.summary, &self.topics)
    }
}

pub fn episode_search_text(summary: &str, topics: &[String]) -> String {
    if topics.is_empty() {
        summary.to_string()
    } else {
        format!("{summary} Topics: {}", topics.join(", "))
    }
}

/// An episode to write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewEpisode {
    pub scope: String,
    pub session_id: String,
    pub summary: String,
    #[serde(default)]
    pub topics: Vec<String>,
    pub importance: f32,
}

/// A stored vector plus the model that produced it. Vectors from a
/// different model are ignored at recall time instead of being compared.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredEmbedding {
    pub model: String,
    pub vector: Vec<f32>,
}

/// When a new fact counts as a near duplicate of an active one under another
/// key. A threshold above 1.0 never matches; see [`MergeConfig::disabled`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeConfig {
    /// Minimum cosine similarity when both sides have vectors from the same
    /// model. Defaults to the daemon's restatement threshold.
    pub semantic_threshold: f32,
    /// Minimum share of words in common when they do not.
    pub text_threshold: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            semantic_threshold: FACT_RESTATEMENT_MIN_SIMILARITY,
            text_threshold: 0.9,
        }
    }
}

impl MergeConfig {
    /// Never merge across keys.
    pub fn disabled() -> Self {
        Self {
            semantic_threshold: 2.0,
            text_threshold: 2.0,
        }
    }
}

/// A fact as a store hands it back, with its embedding if it has one.
#[derive(Debug, Clone, PartialEq)]
pub struct FactRecord {
    pub fact: Fact,
    pub embedding: Option<StoredEmbedding>,
}

/// An episode as a store hands it back.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeRecord {
    pub episode: Episode,
    pub embedding: Option<StoredEmbedding>,
}

/// Who may see a stored fact or episode. The default is the narrowest level,
/// [`FactPrivacy::Private`], so a record nobody classified stays out of
/// shared channels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactAccess {
    pub privacy: FactPrivacy,
    /// The originating channel; only meaningful with [`FactPrivacy::Channel`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
}

impl Default for FactAccess {
    fn default() -> Self {
        Self::private()
    }
}

impl FactAccess {
    /// Direct messages with the owner only.
    pub fn private() -> Self {
        Self {
            privacy: FactPrivacy::Private,
            channel_id: None,
        }
    }

    /// Shareable outside direct messages.
    pub fn global() -> Self {
        Self {
            privacy: FactPrivacy::Global,
            channel_id: None,
        }
    }

    /// Only in `channel_id` (and the owner's direct messages).
    pub fn channel(channel_id: impl Into<String>) -> Self {
        Self {
            privacy: FactPrivacy::Channel,
            channel_id: Some(channel_id.into()),
        }
    }

    /// Read from stored text. Anything unrecognized, and a channel level with
    /// no channel, is private: never widen on data this build cannot read.
    pub fn from_stored(privacy: &str, channel_id: Option<String>) -> Self {
        match (privacy, channel_id) {
            ("global", _) => Self::global(),
            ("channel", Some(channel)) => Self::channel(channel),
            _ => Self::private(),
        }
    }

    /// The access that satisfies both sides. Used when two writes land on
    /// one record, so restating something can never widen who sees it.
    pub fn narrowest(&self, other: &Self) -> Self {
        match (self.privacy, other.privacy) {
            (FactPrivacy::Private, _) | (_, FactPrivacy::Private) => Self::private(),
            (FactPrivacy::Global, _) => other.clone(),
            (_, FactPrivacy::Global) => self.clone(),
            (FactPrivacy::Channel, FactPrivacy::Channel) => {
                if self.channel_id == other.channel_id {
                    self.clone()
                } else {
                    Self::private()
                }
            }
        }
    }
}

/// Who a recall is for, as the host knows it. Uses the same visibility rule
/// as the daemon's own store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallAudience {
    pub visibility: ChannelVisibility,
    pub channel_id: Option<String>,
    pub requester_is_owner: bool,
}

impl RecallAudience {
    pub fn new(
        visibility: ChannelVisibility,
        channel_id: Option<String>,
        requester_is_owner: bool,
    ) -> Self {
        Self {
            visibility,
            channel_id,
            requester_is_owner,
        }
    }

    /// The owner in a direct conversation: sees everything in the scope.
    pub fn owner() -> Self {
        Self::new(ChannelVisibility::Private, None, true)
    }

    /// From a host's stored visibility text. An unrecognized value is the
    /// most restrictive visibility, unlike
    /// [`ChannelVisibility::from_str_lossy`], which defaults to `Internal`.
    pub fn from_stored(
        visibility: &str,
        channel_id: Option<String>,
        requester_is_owner: bool,
    ) -> Self {
        let visibility = match visibility {
            "private" => ChannelVisibility::Private,
            "private_group" => ChannelVisibility::PrivateGroup,
            "public" => ChannelVisibility::Public,
            "internal" => ChannelVisibility::Internal,
            _ => ChannelVisibility::PublicExternal,
        };
        Self::new(visibility, channel_id, requester_is_owner)
    }

    /// Whether this audience may see a record with `access`. `None` means
    /// the store could not say, which is treated as private.
    pub fn may_see(&self, access: Option<&FactAccess>) -> bool {
        let private = FactAccess::private();
        let access = access.unwrap_or(&private);
        memory_visible(
            &ExactChannelIds,
            access.privacy,
            access.channel_id.as_deref(),
            self.channel_id.as_deref(),
            self.visibility,
            self.requester_is_owner,
        )
    }

    /// Sees every record whatever its access, so no lookup is needed.
    pub fn sees_everything(&self) -> bool {
        self.may_see(None)
    }

    /// The access of something learned in this audience's conversation:
    /// a shared channel's record stays in that channel, anything else is
    /// private. Sharing wider is a deliberate call to
    /// [`crate::Memory::remember_with`].
    pub fn access_for_writes(&self) -> FactAccess {
        match (self.visibility, &self.channel_id) {
            (ChannelVisibility::PrivateGroup | ChannelVisibility::Public, Some(channel_id)) => {
                FactAccess::channel(channel_id.clone())
            }
            _ => FactAccess::private(),
        }
    }
}

/// The independently bounded retrieval signals behind a score (port of
/// the daemon's `HybridSignals`). All 0.0-1.0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Signals {
    pub semantic: f32,
    pub lexical: f32,
    pub graph: f32,
    pub freshness: f32,
    pub confidence: f32,
    pub provenance: f32,
}

/// A scorer's verdict on one candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub value: f32,
    pub signals: Signals,
}

/// A recalled fact with its score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredFact {
    pub fact: Fact,
    pub score: f32,
    pub signals: Signals,
}

/// A recalled episode with its score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredEpisode {
    pub episode: Episode,
    pub score: f32,
    pub signals: Signals,
}

/// Counts reported by [`crate::MemoryStore::stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryStats {
    pub active_facts: u64,
    pub superseded_facts: u64,
    pub episodes: u64,
    pub scopes: u64,
}

/// What one [`crate::Memory::decay`] pass changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecayReport {
    /// Active facts whose recall count was lowered.
    pub facts_decayed: u64,
    pub episodes_decayed: u64,
    /// Superseded fact versions deleted after the retention window.
    pub superseded_deleted: u64,
    /// Never-recalled episodes deleted after the retention window.
    pub episodes_deleted: u64,
}

/// Decay and retention windows (ports of the daemon's `decay_memories` and
/// `RetentionConfig` defaults). `None` disables a step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetentionConfig {
    /// Facts not recalled for this long lose a tenth of their recall count
    /// (at least one) per pass. Default 14 days.
    pub fact_recall_decay_after: Option<chrono::Duration>,
    /// Episodes not recalled for this long lose one recall per pass.
    /// Default 30 days.
    pub episode_recall_decay_after: Option<chrono::Duration>,
    /// Superseded fact versions older than this are deleted. Default 90 days.
    pub superseded_retention: Option<chrono::Duration>,
    /// Episodes never recalled and older than this are deleted. Default 365
    /// days.
    pub episode_retention: Option<chrono::Duration>,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            fact_recall_decay_after: Some(chrono::Duration::days(14)),
            episode_recall_decay_after: Some(chrono::Duration::days(30)),
            superseded_retention: Some(chrono::Duration::days(90)),
            episode_retention: Some(chrono::Duration::days(365)),
        }
    }
}

/// Fixed-width UTC timestamp so text comparison equals time comparison.
pub(crate) fn ts(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Micros, true)
}

pub(crate) fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

/// Truncate to at most `max` characters on a char boundary.
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((at, _)) => s[..at].to_string(),
        None => s.to_string(),
    }
}

pub(crate) fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_categories_are_canonical() {
        assert_eq!(normalize_key("  Favorite Color "), "favorite_color");
        assert_eq!(normalize_key("dog-name"), "dog_name");
        assert_eq!(normalize_category(" Preference "), "preference");
        assert!(same_value("Dark  Mode", "dark mode"));
        assert!(!same_value("dark mode", "light mode"));
    }

    #[test]
    fn stored_access_fails_closed() {
        assert_eq!(
            FactAccess::from_stored("global", None),
            FactAccess::global()
        );
        assert_eq!(
            FactAccess::from_stored("channel", Some("c1".into())),
            FactAccess::channel("c1")
        );
        for unreadable in ["", "GLOBAL", "public", "everyone"] {
            assert_eq!(
                FactAccess::from_stored(unreadable, None),
                FactAccess::private(),
                "{unreadable}"
            );
        }
        // A channel level that names no channel cannot be scoped.
        assert_eq!(
            FactAccess::from_stored("channel", None),
            FactAccess::private()
        );
    }

    #[test]
    fn merged_access_never_widens() {
        let (private, global) = (FactAccess::private(), FactAccess::global());
        let (one, two) = (FactAccess::channel("c1"), FactAccess::channel("c2"));
        assert_eq!(global.narrowest(&global), global);
        assert_eq!(global.narrowest(&one), one);
        assert_eq!(one.narrowest(&global), one);
        assert_eq!(one.narrowest(&one), one);
        assert_eq!(one.narrowest(&two), private);
        assert_eq!(private.narrowest(&global), private);
        assert_eq!(global.narrowest(&private), private);
    }

    #[test]
    fn audience_visibility_matches_the_daemon_rule() {
        use ChannelVisibility as V;
        let group = RecallAudience::new(V::PrivateGroup, Some("c1".into()), true);
        assert!(!group.may_see(Some(&FactAccess::private())));
        assert!(group.may_see(Some(&FactAccess::global())));
        assert!(group.may_see(Some(&FactAccess::channel("c1"))));
        assert!(!group.may_see(Some(&FactAccess::channel("c2"))));
        assert!(!group.may_see(None), "an unknown level is private");

        let public = RecallAudience::new(V::Public, Some("c1".into()), true);
        assert!(!public.may_see(Some(&FactAccess::private())));
        assert!(!public.may_see(Some(&FactAccess::global())));

        let external = RecallAudience::new(V::PublicExternal, Some("c1".into()), true);
        assert!(!external.may_see(Some(&FactAccess::global())));
        assert!(!external.may_see(Some(&FactAccess::channel("c1"))));

        let guest_dm = RecallAudience::new(V::Private, None, false);
        assert!(!guest_dm.may_see(Some(&FactAccess::private())));
        assert!(guest_dm.may_see(Some(&FactAccess::global())));

        assert!(RecallAudience::owner().sees_everything());
        assert!(RecallAudience::new(V::Internal, None, true).sees_everything());
        assert!(!group.sees_everything());
    }

    #[test]
    fn unknown_visibility_text_is_the_most_restrictive() {
        for unknown in ["", "PUBLIC", "dm", "internal "] {
            let audience = RecallAudience::from_stored(unknown, None, true);
            assert_eq!(audience.visibility, ChannelVisibility::PublicExternal);
            assert!(!audience.may_see(Some(&FactAccess::global())));
        }
        assert_eq!(
            RecallAudience::from_stored("private_group", None, true).visibility,
            ChannelVisibility::PrivateGroup
        );
    }

    #[test]
    fn writes_stay_in_the_conversation_they_came_from() {
        use ChannelVisibility as V;
        let in_channel = |v| RecallAudience::new(v, Some("c1".into()), true).access_for_writes();
        assert_eq!(in_channel(V::PrivateGroup), FactAccess::channel("c1"));
        assert_eq!(in_channel(V::Public), FactAccess::channel("c1"));
        assert_eq!(in_channel(V::Private), FactAccess::private());
        assert_eq!(in_channel(V::Internal), FactAccess::private());
        assert_eq!(in_channel(V::PublicExternal), FactAccess::private());
        assert_eq!(
            RecallAudience::new(V::Public, None, true).access_for_writes(),
            FactAccess::private()
        );
    }

    #[test]
    fn invalid_facts_are_rejected() {
        assert!(NewFact::new("s", "user", "", "x").normalized().is_err());
        assert!(NewFact::new("", "user", "k", "x").normalized().is_err());
        let f = NewFact::new("s", "user", "k", "x")
            .confidence(7.0)
            .normalized()
            .unwrap();
        assert_eq!(f.confidence, 1.0);
    }
}
