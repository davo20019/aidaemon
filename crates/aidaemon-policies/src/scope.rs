//! Filesystem scope as a `before_tool` hook, driven by access manifests.

use std::path::{Component, Path, PathBuf};

use aidaemon_core::{Hook, HookContext, Safeguard, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;

/// What to do with a call whose access manifest is empty (undeclared).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndeclaredAccess {
    Allow,
    Deny,
}

/// Refuse calls whose declared `reads`/`writes` fall outside the allowed
/// roots.
///
/// Paths are normalized lexically (`.` and `..` resolved, relative paths
/// joined to the base directory, which defaults to the first root). With
/// [`ScopeGuard::resolve_symlinks`], the longest existing ancestor of each
/// path is canonicalized too, so a symlink inside a root cannot point out.
///
/// The guard checks paths only. Hosts and commands are other policies'
/// business ([`crate::ApprovalPolicy`] for commands); a manifest that
/// declares only those has no paths to check and is allowed.
#[derive(Debug, Clone)]
pub struct ScopeGuard {
    roots: Vec<PathBuf>,
    base: Option<PathBuf>,
    home: Option<PathBuf>,
    resolve_symlinks: bool,
    undeclared_read_only: UndeclaredAccess,
    undeclared_writer: UndeclaredAccess,
}

impl ScopeGuard {
    pub fn new<I, P>(roots: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self {
            roots: roots
                .into_iter()
                .map(|r| normalize(r.as_ref(), Path::new("/")))
                .collect(),
            base: None,
            home: None,
            resolve_symlinks: false,
            undeclared_read_only: UndeclaredAccess::Allow,
            undeclared_writer: UndeclaredAccess::Deny,
        }
    }

    /// Directory relative paths are resolved against (default: first root).
    pub fn base_dir(mut self, base: impl AsRef<Path>) -> Self {
        self.base = Some(normalize(base.as_ref(), Path::new("/")));
        self
    }

    /// Expand `~` with this directory. Without it, `~` paths are refused.
    pub fn home_dir(mut self, home: impl AsRef<Path>) -> Self {
        self.home = Some(normalize(home.as_ref(), Path::new("/")));
        self
    }

    /// Canonicalize existing path prefixes (and the roots) before checking.
    pub fn resolve_symlinks(mut self, on: bool) -> Self {
        self.resolve_symlinks = on;
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

    /// The decision for one call, without a hook context.
    pub fn decide(&self, tool_name: &str, info: &ToolInfo) -> ToolDecision {
        if info.access.is_empty() {
            let policy = if info.annotations.read_only {
                self.undeclared_read_only
            } else {
                self.undeclared_writer
            };
            return match policy {
                UndeclaredAccess::Allow => ToolDecision::Allow,
                UndeclaredAccess::Deny => ToolDecision::Deny {
                    reason: format!(
                        "blocked by scope policy: `{tool_name}` can change things but did not \
                         declare which paths it touches, so it cannot be checked against the \
                         allowed directories ({})",
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
            let Some(resolved) = self.resolve(raw) else {
                return ToolDecision::Deny {
                    reason: format!(
                        "blocked by scope policy: cannot {kind} `{raw}`: `~` paths are not \
                         resolvable here; use a path under {}",
                        self.roots_list()
                    ),
                };
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

    fn resolve(&self, raw: &str) -> Option<PathBuf> {
        let raw = raw.trim();
        let path = if raw == "~" || raw.starts_with("~/") {
            let home = self.home.as_ref()?;
            home.join(raw.trim_start_matches('~').trim_start_matches('/'))
        } else if raw.starts_with('~') {
            return None;
        } else {
            PathBuf::from(raw)
        };
        let base = self
            .base
            .clone()
            .or_else(|| self.roots.first().cloned())
            .unwrap_or_else(|| PathBuf::from("/"));
        let normalized = normalize(&path, &base);
        Some(if self.resolve_symlinks {
            canonicalize_existing_prefix(&normalized)
        } else {
            normalized
        })
    }

    fn within_roots(&self, path: &Path) -> bool {
        self.roots.iter().any(|root| {
            let root = if self.resolve_symlinks {
                canonicalize_existing_prefix(root)
            } else {
                root.clone()
            };
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

/// Lexical normalization: join to `base` if relative, then resolve `.` and
/// `..` without touching the filesystem. `..` never climbs above `/`.
fn normalize(path: &Path, base: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let mut out = PathBuf::from("/");
    for component in joined.components() {
        match component {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

/// Canonicalize the longest existing ancestor and re-append the rest.
fn canonicalize_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut out = canonical;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (
            existing.file_name().map(|n| n.to_owned()),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return path.to_path_buf(),
        }
    }
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
    fn symlinks_out_of_root_are_caught_when_resolving() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();
        let target = root.path().join("link/file.txt");
        let target = target.to_str().unwrap();

        let lexical = ScopeGuard::new([root.path()]);
        assert_eq!(
            lexical.decide("t", &info(ToolAnnotations::WRITE, &[], &[target])),
            ToolDecision::Allow
        );
        let resolving = ScopeGuard::new([root.path()]).resolve_symlinks(true);
        assert!(denied(
            &resolving.decide("t", &info(ToolAnnotations::WRITE, &[], &[target]))
        ));
    }
}
