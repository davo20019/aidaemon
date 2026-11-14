//! Secret and sensitive paths as a `before_tool` hook.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_core::{Hook, HookContext, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;

use super::{FilePolicy, SharedBackend, workspace_parts};
use crate::backend::BackendPath;
use crate::semantics::FileGuard;
use crate::workspace::Workspace;

/// Refuses a call whose declared paths hold secrets or must not be changed.
///
/// Paths are resolved through the [`ExecutionBackend`] the tool uses, and
/// symlinks are followed, so a link to a secret is judged as the secret.
///
/// [`ExecutionBackend`]: crate::backend::ExecutionBackend
pub struct ProtectedPathGuard {
    backend: SharedBackend,
    policy: Arc<dyn FilePolicy>,
    tools: HashMap<String, FileGuard>,
}

impl ProtectedPathGuard {
    pub fn new(backend: SharedBackend, policy: Arc<dyn FilePolicy>) -> Self {
        Self {
            backend,
            policy,
            tools: HashMap::new(),
        }
    }

    /// The guard for the bundled file tools confined to `workspace`.
    pub fn for_workspace(workspace: Workspace) -> Self {
        let (backend, policy) = workspace_parts(workspace);
        Self::new(backend, policy)
            .tool("read_file", FileGuard::Reads)
            .tool("write_file", FileGuard::Writes)
            .tool("edit_file", FileGuard::Writes)
    }

    /// State which of `tool`'s paths are judged.
    pub fn tool(mut self, tool: impl Into<String>, guard: FileGuard) -> Self {
        self.tools.insert(tool.into(), guard);
        self
    }

    /// Why a call of `tool` over these paths is refused, if it is. A path
    /// the backend cannot resolve is left to the tool to report.
    pub async fn refusal(&self, tool: &str, reads: &[String], writes: &[String]) -> Option<String> {
        match self.tools.get(tool).copied().unwrap_or_default() {
            FileGuard::None => None,
            FileGuard::Reads => {
                for raw in reads {
                    if self.resolved_any(raw, |path| self.holds_secret(path)).await {
                        return Some(match self.hint(raw).await {
                            Some(hint) => format!(
                                "{tool} does not open credential or secret files ({raw}). {hint}"
                            ),
                            None => format!(
                                "{tool} does not open credential or secret files ({raw}). Ask \
                                 the owner to share what is needed."
                            ),
                        });
                    }
                }
                None
            }
            FileGuard::SensitiveReads => {
                for raw in reads {
                    if self
                        .resolved_any(raw, |path| self.policy.is_sensitive(path))
                        .await
                    {
                        return Some(format!("{tool} cannot read sensitive path: {raw}"));
                    }
                }
                None
            }
            FileGuard::Writes => {
                for raw in writes {
                    if self
                        .resolved_any(raw, |path| self.policy.is_sensitive(path))
                        .await
                    {
                        return Some(match self.hint(raw).await {
                            Some(hint) => {
                                format!("{tool} cannot change sensitive path: {raw}. {hint}")
                            }
                            None => format!("{tool} cannot change sensitive path: {raw}"),
                        });
                    }
                }
                None
            }
        }
    }

    /// The policy's pointer for a refused path, as named or as it really is.
    async fn hint(&self, raw: &str) -> Option<String> {
        let path = self.backend.resolve_path(raw).await.ok()?;
        if let Some(hint) = self.policy.refusal_hint(Path::new(path.as_str())) {
            return Some(hint);
        }
        let real = self.real_path(&path).await?;
        self.policy.refusal_hint(&real)
    }

    /// `.env.example` style templates hold placeholders: they are judged by
    /// where they live, not by their `.env.` name.
    fn holds_secret(&self, path: &Path) -> bool {
        if is_env_template(path) {
            self.policy.is_secret_ignoring_name_shape(path)
        } else {
            self.policy.is_secret(path)
        }
    }

    /// Whether `question` holds for the path as named or as it really is.
    async fn resolved_any(&self, raw: &str, question: impl Fn(&Path) -> bool) -> bool {
        let Ok(path) = self.backend.resolve_path(raw).await else {
            return false;
        };
        question(Path::new(path.as_str()))
            || self
                .real_path(&path)
                .await
                .is_some_and(|real| question(&real))
    }

    /// `path` through its symlinks. A path that does not exist yet is taken
    /// through its nearest existing ancestor.
    async fn real_path(&self, path: &BackendPath) -> Option<PathBuf> {
        let mut existing = path.clone();
        let mut tail: Vec<String> = Vec::new();
        loop {
            if let Ok(real) = self.backend.canonicalize(&existing).await {
                let mut resolved = PathBuf::from(real.as_str());
                resolved.extend(tail.iter().rev());
                return Some(resolved);
            }
            tail.push(existing.file_name()?.to_string());
            existing = existing.parent()?;
        }
    }
}

fn is_env_template(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        let name = name.to_string_lossy().to_ascii_lowercase();
        matches!(
            name.as_str(),
            ".env.example" | ".env.sample" | ".env.template"
        )
    })
}

#[async_trait]
impl Hook for ProtectedPathGuard {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        match self
            .refusal(&call.name, &tool.access.reads, &tool.access.writes)
            .await
        {
            Some(reason) => ToolDecision::Deny { reason },
            None => ToolDecision::Allow,
        }
    }
}
