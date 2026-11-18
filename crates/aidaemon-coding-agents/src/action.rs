//! What a coding agent asks to do, in terms any embedding agent can judge.

use async_trait::async_trait;

/// The embedding agent's answer about an [`AgentAction`]: the kernel's own
/// verdict for a call a running tool makes ([`aidaemon_core::NestedCalls`]).
pub use aidaemon_core::NestedVerdict as Verdict;

/// One action a coding agent asked for permission to take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAction {
    /// Run a shell command (the script itself, unwrapped from the agent's
    /// own shell wrapper).
    RunCommand {
        command: String,
        working_dir: Option<String>,
    },
    /// Write `content` to `path` (a whole file, or a patch's text).
    WriteFile { path: String, content: String },
    /// Replace text in `path`.
    EditFile {
        path: String,
        edits: Vec<TextEdit>,
        replace_all: bool,
    },
    /// Read `path`.
    ReadFile { path: String },
    /// Search file contents (`pattern`) or names (`glob`) under `path`.
    Search {
        pattern: Option<String>,
        glob: Option<String>,
        path: Option<String>,
    },
    /// Fetch `url`.
    Fetch { url: String },
    /// Reach network host `host` (from a sandbox that blocks the network).
    ReachHost { host: String },
}

/// One replacement in [`AgentAction::EditFile`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    pub old_text: String,
    pub new_text: String,
}

impl AgentAction {
    /// The action as its owner reads it: what it is and what it targets.
    pub fn label(&self) -> String {
        let short = |text: &str| {
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            match text.char_indices().nth(160) {
                Some((cut, _)) => format!("{}…", &text[..cut]),
                None => text,
            }
        };
        match self {
            Self::RunCommand { command, .. } => format!("command: {}", short(command)),
            Self::WriteFile { path, .. } => format!("write: {path}"),
            Self::EditFile { path, .. } => format!("edit: {path}"),
            Self::ReadFile { path } => format!("read: {path}"),
            Self::Search {
                pattern,
                glob,
                path,
            } => format!(
                "search: {}{}",
                pattern.as_deref().or(glob.as_deref()).unwrap_or_default(),
                path.as_deref()
                    .map(|path| format!(" in {path}"))
                    .unwrap_or_default()
            ),
            Self::Fetch { url } => format!("fetch: {url}"),
            Self::ReachHost { host } => format!("network: {host}"),
        }
    }
}

/// Allows every action as asked, unjudged: an agent in full access whose
/// protocol still asks.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

#[async_trait]
impl ActionJudge for AllowAll {
    async fn judge(&self, _action: AgentAction) -> Verdict {
        Verdict::Allow
    }
}

/// Decides each action a hosted coding agent asks for: the embedding
/// agent's safeguards. It may wait for its owner's answer; the agent waits
/// with it.
#[async_trait]
pub trait ActionJudge: Send + Sync {
    async fn judge(&self, action: AgentAction) -> Verdict;
}
