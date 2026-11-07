//! Episodes: a model summarizes a finished stretch of conversation into a
//! recallable episode (summary, topics, tone, outcome, importance).

use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tracing::info;

use crate::history::TurnRole;
use crate::scoring::episode_importance;
use crate::stores::EpisodeStore;

/// One message of a session, as an episode transcript reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptEntry {
    pub role: TurnRole,
    pub content: Option<String>,
    /// For an assistant message: the tools it called, by name.
    pub tool_calls: Vec<String>,
    /// For a tool result: the tool's name.
    pub tool_name: Option<String>,
}

/// The transcript an analysis model reads: the user's words in full, the
/// assistant's actions and (shortened) text, and tool outputs compressed to
/// a line each.
pub fn clean_transcript(entries: &[TranscriptEntry]) -> String {
    let mut transcript = String::new();
    for entry in entries {
        match entry.role {
            TurnRole::User => {
                if let Some(content) = &entry.content {
                    transcript.push_str(&format!("User: {content}\n\n"));
                }
            }
            TurnRole::Assistant => {
                for name in &entry.tool_calls {
                    transcript.push_str(&format!("[Action: {name}]\n"));
                }
                if let Some(content) = entry.content.as_deref().filter(|c| !c.trim().is_empty()) {
                    let shortened = crate::text::truncate_chars(content, 500);
                    transcript.push_str(&format!("Assistant: {shortened}\n\n"));
                }
            }
            TurnRole::Tool => {
                if let (Some(content), Some(name)) = (&entry.content, &entry.tool_name) {
                    let summary = if content.len() > 200 {
                        format!("[{name} output: {} chars]", content.len())
                    } else {
                        format!(
                            "[{name}: {}]",
                            content.chars().take(100).collect::<String>()
                        )
                    };
                    transcript.push_str(&format!("{summary}\n"));
                }
            }
        }
    }
    transcript
}

const ANALYSIS_PROMPT: &str = r#"You are a memory system analyzing a conversation. Extract:
1. A brief summary (1-2 sentences)
2. Main topics discussed (list of keywords)
3. Emotional tone (one of: neutral, productive, frustrated, exploratory, celebratory)
4. Outcome (one of: resolved, ongoing, abandoned, learning)
5. Possible enduring personal goals the user explicitly stated for themselves
6. Whether the session ran into errors (had_errors) and whether the user made decisions or commitments (made_decisions)

For goals_mentioned, return an empty list for ordinary one-off requests, implementation
steps, delegated work, quoted examples, prohibitions, review checklists, or specialist/system
instructions. Only include a goal when the user's own words describe an intention that should
remain relevant beyond this conversation.

Respond with ONLY valid JSON:
{
  "summary": "...",
  "topics": ["topic1", "topic2"],
  "emotional_tone": "...",
  "outcome": "...",
  "goals_mentioned": ["goal1", "goal2"],
  "emotional_intensity": 0.5,
  "had_errors": false,
  "made_decisions": false
}

emotional_intensity is 0.0-1.0 scale (0=calm, 1=highly emotional)"#;

/// Summarizes sessions into episodes with a model.
#[derive(Clone)]
pub struct SessionAnalyst {
    provider: Arc<dyn Provider>,
    model: String,
}

/// A session episode for the store to record.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionEpisode {
    pub session_id: String,
    pub summary: String,
    pub topics: Vec<String>,
    pub emotional_tone: String,
    pub outcome: String,
    pub importance: f32,
    pub message_count: usize,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub channel_id: Option<String>,
}

impl SessionAnalyst {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// Ask the model about a transcript. An unparseable reply is a neutral
    /// default analysis; provider failures and empty replies are errors.
    pub async fn analyze(&self, transcript: &str) -> anyhow::Result<SessionAnalysis> {
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![
                Message::system(ANALYSIS_PROMPT),
                Message::user(format!("Analyze this conversation:\n\n{transcript}")),
            ],
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        if response.text.trim().is_empty() {
            anyhow::bail!("Empty response from LLM");
        }
        Ok(parse_session_analysis(&response.text))
    }

    /// Analyze a session and record it as an episode. Importance comes from
    /// the session's size and the analysis's typed signals (errors,
    /// decisions, goals, intensity), never from matching words. Returns the
    /// episode id and the analysis, whose goals the host may act on.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_episode<S>(
        &self,
        store: &S,
        session_id: &str,
        entries: &[TranscriptEntry],
        message_count: usize,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        channel_id: Option<String>,
    ) -> anyhow::Result<(i64, SessionAnalysis)>
    where
        S: EpisodeStore + ?Sized,
    {
        anyhow::ensure!(message_count > 0, "No messages to create episode from");
        let analysis = self.analyze(&clean_transcript(entries)).await?;
        let importance = episode_importance(
            message_count,
            analysis.had_errors,
            analysis.made_decisions,
            !analysis.goals_mentioned.is_empty(),
            analysis.emotional_intensity,
        );
        let episode = SessionEpisode {
            session_id: session_id.to_string(),
            summary: analysis.summary.clone(),
            topics: analysis.topics.clone(),
            emotional_tone: analysis.emotional_tone.clone(),
            outcome: analysis.outcome.clone(),
            importance,
            message_count,
            start_time,
            end_time,
            channel_id,
        };
        let episode_id = store.record_session_episode(&episode).await?;
        info!(
            session_id,
            episode_id,
            topics = ?analysis.topics,
            emotional_tone = ?analysis.emotional_tone,
            outcome = ?analysis.outcome,
            goals = analysis.goals_mentioned.len(),
            "Created episode"
        );
        Ok((episode_id, analysis))
    }
}

/// The JSON object in a reply, leniently; anything unusable becomes the
/// neutral default analysis.
pub fn parse_session_analysis(text: &str) -> SessionAnalysis {
    let trimmed = text.trim();
    let json = match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if end >= start => &trimmed[start..=end],
        _ => trimmed,
    };
    serde_json::from_str(json).unwrap_or_else(|_| SessionAnalysis {
        summary: "Session with various tasks".to_string(),
        topics: vec!["general".to_string()],
        emotional_tone: "neutral".to_string(),
        outcome: "ongoing".to_string(),
        goals_mentioned: vec![],
        emotional_intensity: 0.3,
        had_errors: false,
        made_decisions: false,
    })
}

/// What the analysis model said about a session.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionAnalysis {
    #[serde(
        default = "default_session_summary",
        deserialize_with = "deserialize_session_summary"
    )]
    pub summary: String,
    #[serde(default, deserialize_with = "deserialize_lossy_string_vec")]
    pub topics: Vec<String>,
    #[serde(
        default = "default_emotional_tone",
        deserialize_with = "deserialize_emotional_tone"
    )]
    pub emotional_tone: String,
    #[serde(
        default = "default_session_outcome",
        deserialize_with = "deserialize_session_outcome"
    )]
    pub outcome: String,
    #[serde(default, deserialize_with = "deserialize_lossy_string_vec")]
    pub goals_mentioned: Vec<String>,
    #[serde(
        default = "default_emotional_intensity",
        deserialize_with = "deserialize_emotional_intensity"
    )]
    pub emotional_intensity: f32,
    /// The session ran into errors (failed commands, broken builds).
    #[serde(default, deserialize_with = "deserialize_lossy_bool")]
    pub had_errors: bool,
    /// The user made decisions or commitments in the session.
    #[serde(default, deserialize_with = "deserialize_lossy_bool")]
    pub made_decisions: bool,
}

fn deserialize_lossy_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Bool(flag) => flag,
        serde_json::Value::String(text) => text.trim().eq_ignore_ascii_case("true"),
        _ => false,
    })
}

fn default_session_summary() -> String {
    "Session with various tasks".to_string()
}

fn default_emotional_tone() -> String {
    "neutral".to_string()
}

fn default_session_outcome() -> String {
    "ongoing".to_string()
}

fn default_emotional_intensity() -> f32 {
    0.3
}

fn deserialize_string_value<'de, D>(deserializer: D, fallback: &str) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback)
        .to_string())
}

fn deserialize_session_summary<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "Session with various tasks")
}

fn deserialize_emotional_tone<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "neutral")
}

fn deserialize_session_outcome<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_value(deserializer, "ongoing")
}

fn deserialize_lossy_string_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let values = match value {
        serde_json::Value::Array(values) => values,
        serde_json::Value::String(value) => vec![serde_json::Value::String(value)],
        _ => return Ok(Vec::new()),
    };
    Ok(values
        .into_iter()
        .filter_map(|value| value.as_str().map(str::trim).map(str::to_string))
        .filter(|value| !value.is_empty())
        .collect())
}

fn deserialize_emotional_intensity<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let intensity = value
        .as_f64()
        .map(|value| value as f32)
        .or_else(|| value.as_str().and_then(|value| value.parse::<f32>().ok()))
        .unwrap_or_else(default_emotional_intensity);
    Ok(intensity.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use aidaemon_core::testing::{text, ScriptedProvider};
    use async_trait::async_trait;

    use super::*;

    #[test]
    fn analysis_keeps_valid_fields_when_siblings_are_null_or_malformed() {
        let analysis: SessionAnalysis = serde_json::from_str(
            r#"{
                "summary": {"unexpected":"object"},
                "topics": ["deployments", null, {"bad":true}, "reliability"],
                "emotional_tone": null,
                "outcome": "",
                "goals_mentioned": "Improve release reliability",
                "emotional_intensity": "1.7",
                "had_errors": "true"
            }"#,
        )
        .unwrap();
        assert_eq!(analysis.summary, "Session with various tasks");
        assert_eq!(analysis.topics, ["deployments", "reliability"]);
        assert_eq!(analysis.emotional_tone, "neutral");
        assert_eq!(analysis.outcome, "ongoing");
        assert_eq!(analysis.goals_mentioned, ["Improve release reliability"]);
        assert_eq!(analysis.emotional_intensity, 1.0);
        assert!(analysis.had_errors && !analysis.made_decisions);
        assert_eq!(
            parse_session_analysis("not json").summary,
            "Session with various tasks"
        );
    }

    #[test]
    fn transcripts_keep_the_users_words_and_compress_the_rest() {
        let entries = [
            TranscriptEntry {
                role: TurnRole::User,
                content: Some("Deploy the site".into()),
                tool_calls: vec![],
                tool_name: None,
            },
            TranscriptEntry {
                role: TurnRole::Assistant,
                content: Some("Deploying now.".into()),
                tool_calls: vec!["terminal".into()],
                tool_name: None,
            },
            TranscriptEntry {
                role: TurnRole::Tool,
                content: Some("x".repeat(500)),
                tool_calls: vec![],
                tool_name: Some("terminal".into()),
            },
        ];
        let transcript = clean_transcript(&entries);
        assert!(transcript.contains("User: Deploy the site"));
        assert!(transcript.contains("[Action: terminal]"));
        assert!(transcript.contains("Assistant: Deploying now."));
        assert!(transcript.contains("[terminal output: 500 chars]"));
    }

    #[derive(Default)]
    struct Episodes(Mutex<Vec<SessionEpisode>>);

    #[async_trait]
    impl EpisodeStore for Episodes {
        async fn record_session_episode(&self, episode: &SessionEpisode) -> anyhow::Result<i64> {
            self.0.lock().unwrap().push(episode.clone());
            Ok(7)
        }
    }

    #[tokio::test]
    async fn importance_comes_from_typed_signals() {
        let reply = r#"{"summary":"Fixed the release pipeline","topics":["ci"],"emotional_tone":"productive",
            "outcome":"resolved","goals_mentioned":[],"emotional_intensity":0.0,"had_errors":true,"made_decisions":true}"#;
        let analyst = SessionAnalyst::new(Arc::new(ScriptedProvider::new([text(reply)])), "mock");
        let store = Episodes::default();
        let now = Utc::now();
        let entry = TranscriptEntry {
            role: TurnRole::User,
            content: Some("The build is broken again".into()),
            tool_calls: vec![],
            tool_name: None,
        };
        let (id, analysis) = analyst
            .record_episode(
                &store,
                "telegram:synthetic-chat",
                &[entry],
                4,
                now,
                now,
                None,
            )
            .await
            .unwrap();
        assert_eq!(id, 7);
        assert_eq!(analysis.outcome, "resolved");
        let episode = &store.0.lock().unwrap()[0];
        assert_eq!(episode.summary, "Fixed the release pipeline");
        // Base 0.5 + errors 0.15 + decisions 0.1.
        assert!(
            (episode.importance - 0.75).abs() < 1e-6,
            "{}",
            episode.importance
        );
    }
}
