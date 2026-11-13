//! Filesystem scope as a `before_tool` hook, driven by access manifests.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use aidaemon_core::{Hook, HookContext, Safeguard, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;

use crate::path::{resolve_path, resolve_path_from};

/// What to do with a call whose access manifest is empty (undeclared).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndeclaredAccess {
    Allow,
    Deny,
}

/// Refuse calls whose declared `reads`/`writes` fall outside the allowed
/// roots.
///
/// Each path is resolved the way the operating system would open it, by
/// [`crate::resolve_path`]: relative paths are joined to the base directory
/// (default: the first root), then walked one component at a time,
/// following every symbolic link and applying `..` to the directory really
/// reached. A link inside a root that points out, a broken link whose target
/// is outside, and `link/..` all resolve outside and are refused. The roots
/// are resolved the same way, so a root given through a link still admits
/// its own files.
///
/// The guard checks paths, and a path check is not a boundary against a
/// hostile filesystem: a **hard link** inside a root to a file outside, or a
/// directory swapped for a link between the check and the write, gets
/// through. Run tools that write a directory other processes can change in
/// a sandbox (`aidaemon_tools::Sandbox`).
///
/// The guard checks paths only. Hosts and commands are other policies'
/// business ([`crate::ApprovalPolicy`] for commands); a manifest that
/// declares only those has no paths to check and is allowed.
#[derive(Debug, Clone)]
pub struct ScopeGuard {
    roots: Vec<PathBuf>,
    base: Option<PathBuf>,
    home: Option<PathBuf>,
    undeclared_read_only: UndeclaredAccess,
    undeclared_writer: UndeclaredAccess,
    path_free: HashSet<String>,
}

impl ScopeGuard {
    pub fn new<I, P>(roots: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self {
            roots: roots.into_iter().map(|r| absolute(r.as_ref())).collect(),
            base: None,
            home: None,
            undeclared_read_only: UndeclaredAccess::Allow,
            undeclared_writer: UndeclaredAccess::Deny,
            path_free: HashSet::new(),
        }
    }

    /// Directory relative paths are resolved against (default: first root).
    pub fn base_dir(mut self, base: impl AsRef<Path>) -> Self {
        self.base = Some(absolute(base.as_ref()));
        self
    }

    /// Expand `~` with this directory. Without it, `~` paths are refused.
    pub fn home_dir(mut self, home: impl AsRef<Path>) -> Self {
        self.home = Some(absolute(home.as_ref()));
        self
    }

    /// Treatment of empty manifests for read-only tools (default `Allow`).
    pub fn undeclared_read_only(mut self, policy: UndeclaredAccess) -> Self {
        self.undeclared_read_only = policy;
        self
    }

    /// Treatment of empty manifests for other tools (default `Deny`).
    pub fn undeclared_writer(mut self, policy: UndeclaredAccess) -> Self {
        self.undeclared_writer = policy;
        self
    }

    /// A tool known to touch no files (a memory or checklist tool, a
    /// business API): its empty access manifest is taken as "touches
    /// nothing" rather than "not declared", so neither undeclared policy
    /// applies. Paths it does declare are still checked.
    pub fn path_free_tool(mut self, name: impl Into<String>) -> Self {
        self.path_free.insert(name.into());
        self
    }

    /// The decision for one call, without a hook context.
    pub fn decide(&self, tool_name: &str, info: &ToolInfo) -> ToolDecision {
        if info.access.is_empty() {
            if self.path_free.contains(tool_name) {
                return ToolDecision::Allow;
            }
            let policy = if info.annotations.read_only {
                self.undeclared_read_only
            } else {
                self.undeclared_writer
            };
            return match policy {
                UndeclaredAccess::Allow => ToolDecision::Allow,
                UndeclaredAccess::Deny => ToolDecision::Deny {
                    reason: format!(
                        "blocked by scope policy: `{tool_name}` {} but did not declare which \
                         paths it touches, so it cannot be checked against the allowed \
                         directories ({})",
                        if info.annotations.read_only {
                            "can read files"
                        } else {
                            "can change things"
                        },
                        self.roots_list()
                    ),
                },
            };
        }
        let checks = info
            .access
            .reads
            .iter()
            .map(|p| ("read", p))
            .chain(info.access.writes.iter().map(|p| ("write", p)));
        for (kind, raw) in checks {
            let resolved = match self.resolve(raw) {
                Ok(resolved) => resolved,
                Err(why) => {
                    return ToolDecision::Deny {
                        reason: format!(
                            "blocked by scope policy: cannot {kind} `{raw}`: {why}; use a path \
                             under {}",
                            self.roots_list()
                        ),
                    }
                }
            };
            if !self.within_roots(&resolved) {
                return ToolDecision::Deny {
                    reason: format!(
                        "blocked by scope policy: cannot {kind} `{raw}` (resolves to `{}`), which \
                         is outside the allowed directories: {}",
                        resolved.display(),
                        self.roots_list()
                    ),
                };
            }
        }
        ToolDecision::Allow
    }

    /// The file `raw` names, as [`crate::resolve_path`] finds it: `~`
    /// expanded with the home directory, a relative path joined to the base
    /// directory. The same resolution the guard checks, for rules and tools
    /// that want to judge or use the path the guard judged.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let path = if raw == "~" || raw.starts_with("~/") {
            let Some(home) = self.home.as_ref() else {
                return Err("`~` paths are not resolvable here".to_string());
            };
            home.join(raw.trim_start_matches('~').trim_start_matches('/'))
        } else if raw.starts_with('~') {
            return Err("`~user` paths are not resolvable here".to_string());
        } else {
            PathBuf::from(raw)
        };
        let base = self
            .base
            .clone()
            .or_else(|| self.roots.first().cloned())
            .unwrap_or_else(|| PathBuf::from("/"));
        resolve_path_from(&path, &base).map_err(|error| error.to_string())
    }

    fn within_roots(&self, path: &Path) -> bool {
        self.roots.iter().any(|root| {
            let root = resolve_path(root).unwrap_or_else(|_| root.clone());
            path.starts_with(&root)
        })
    }

    fn roots_list(&self) -> String {
        if self.roots.is_empty() {
            return "(none configured)".into();
        }
        self.roots
            .iter()
            .map(|r| format!("`{}`", r.display()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `path` made absolute against `/`, as roots and directories are given.
fn absolute(path: &Path) -> PathBuf {
    Path::new("/").join(path)
}

#[async_trait]
impl Hook for ScopeGuard {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::Scope]
    }

    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.decide(&call.name, tool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{AccessManifest, ToolAnnotations};

    fn info(annotations: ToolAnnotations, reads: &[&str], writes: &[&str]) -> ToolInfo {
        ToolInfo {
            annotations,
            access: AccessManifest {
                reads: reads.iter().map(|s| s.to_string()).collect(),
                writes: writes.iter().map(|s| s.to_string()).collect(),
                ..AccessManifest::default()
            },
            effects: aidaemon_core::CallEffects::unknown(),
            outbound: Default::default(),
        }
    }

    fn denied(d: &ToolDecision) -> bool {
        matches!(d, ToolDecision::Deny { .. })
    }

    #[test]
    fn paths_inside_roots_are_allowed() {
        let guard = ScopeGuard::new(["/work/project"]);
        let w = ToolAnnotations::WRITE;
        assert_eq!(
            guard.decide("t", &info(w, &["/work/project/src/lib.rs"], &["notes.md"])),
            ToolDecision::Allow
        );
        assert_eq!(
            guard.decide("t", &info(w, &["src/../README.md"], &[])),
            ToolDecision::Allow
        );
    }

    #[test]
    fn dot_dot_escapes_are_denied_with_a_reason() {
        let guard = ScopeGuard::new(["/work/project"]);
        let d = guard.decide(
            "write_file",
            &info(ToolAnnotations::WRITE, &[], &["../other/secrets.txt"]),
        );
        match d {
            ToolDecision::Deny { reason } => {
                assert!(reason.contains("/work/other/secrets.txt"), "{reason}");
                assert!(
                    reason.contains("outside the allowed directories"),
                    "{reason}"
                );
            }
            other => panic!("expected deny, got {other:?}"),
        }
        // A sibling whose name shares a prefix is still outside.
        assert!(denied(&guard.decide(
            "t",
            &info(ToolAnnotations::READ_ONLY, &["/work/project-evil/x"], &[])
        )));
        assert!(denied(&guard.decide(
            "t",
            &info(
                ToolAnnotations::READ_ONLY,
                &["/work/project/../../etc/passwd"],
                &[]
            )
        )));
    }

    #[test]
    fn undeclared_access_defaults_depend_on_read_only() {
        let guard = ScopeGuard::new(["/work"]);
        assert_eq!(
            guard.decide("t", &info(ToolAnnotations::READ_ONLY, &[], &[])),
            ToolDecision::Allow
        );
        assert!(denied(
            &guard.decide("t", &info(ToolAnnotations::WRITE, &[], &[]))
        ));
        let lenient = ScopeGuard::new(["/work"]).undeclared_writer(UndeclaredAccess::Allow);
        assert_eq!(
            lenient.decide("t", &info(ToolAnnotations::WRITE, &[], &[])),
            ToolDecision::Allow
        );
        let strict = ScopeGuard::new(["/work"]).undeclared_read_only(UndeclaredAccess::Deny);
        assert!(denied(
            &strict.decide("t", &info(ToolAnnotations::READ_ONLY, &[], &[]))
        ));
    }

    #[test]
    fn path_free_tools_skip_the_undeclared_policies_but_not_declared_paths() {
        let guard = ScopeGuard::new(["/work"])
            .undeclared_read_only(UndeclaredAccess::Deny)
            .path_free_tool("recall")
            .path_free_tool("remember");
        assert_eq!(
            guard.decide("recall", &info(ToolAnnotations::READ_ONLY, &[], &[])),
            ToolDecision::Allow
        );
        assert_eq!(
            guard.decide("remember", &info(ToolAnnotations::WRITE, &[], &[])),
            ToolDecision::Allow
        );
        match guard.decide("read", &info(ToolAnnotations::READ_ONLY, &[], &[])) {
            ToolDecision::Deny { reason } => {
                assert!(reason.contains("can read files"), "{reason}")
            }
            other => panic!("expected deny, got {other:?}"),
        }
        assert!(denied(&guard.decide(
            "recall",
            &info(
                ToolAnnotations::READ_ONLY,
                &["/home/alice/.ssh/id_ed25519"],
                &[]
            )
        )));
    }

    #[test]
    fn home_paths_need_a_configured_home() {
        let guard = ScopeGuard::new(["/home/alice/project"]);
        assert!(denied(&guard.decide(
            "t",
            &info(ToolAnnotations::READ_ONLY, &["~/project/a"], &[])
        )));
        let guard = guard.home_dir("/home/alice");
        assert_eq!(
            guard.decide(
                "t",
                &info(ToolAnnotations::READ_ONLY, &["~/project/a"], &[])
            ),
            ToolDecision::Allow
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_out_of_root_are_always_caught() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();
        let target = root.path().join("link/file.txt");
        let target = target.to_str().unwrap();
        let guard = ScopeGuard::new([root.path()]);
        assert!(denied(
            &guard.decide("t", &info(ToolAnnotations::WRITE, &[], &[target]))
        ));
    }
}
