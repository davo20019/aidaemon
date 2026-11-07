use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Why a path argument was refused.
#[derive(Debug, thiserror::Error)]
pub enum PathError {
    #[error("path must not be empty")]
    Empty,
    #[error(
        "`{path}` is outside the workspace ({root}); use a path relative to the workspace root"
    )]
    Escapes { path: String, root: String },
    #[error("`{path}` uses `..` through a directory that does not exist; use a direct path")]
    UnresolvableParent { path: String },
    #[error("cannot resolve `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },
}

/// A root directory that confines every path argument a tool receives.
///
/// Paths are resolved relative to the root (absolute paths are accepted only
/// if they land inside it), then canonicalized so `..` and symlinks are
/// followed before the containment check. A path that does not exist yet is
/// allowed when its nearest existing ancestor is inside the root, so tools
/// can create new files and directories.
///
/// The check happens when a tool resolves its argument; a concurrent process
/// that swaps a directory for a symlink between the check and the I/O is not
/// defended against. Use a [`crate::Sandbox`] when the
/// workspace itself is shared with untrusted processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Open an existing directory as a workspace. The root is canonicalized.
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        let root = std::fs::canonicalize(root.as_ref())?;
        if !root.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("workspace root {} is not a directory", root.display()),
            ));
        }
        Ok(Self { root })
    }

    /// The canonical root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a path argument to a canonical absolute path inside the root.
    /// The path need not exist, but its nearest existing ancestor must be
    /// inside the root (after following symlinks).
    pub fn resolve(&self, path: &str) -> Result<PathBuf, PathError> {
        if path.trim().is_empty() {
            return Err(PathError::Empty);
        }
        let given = Path::new(path);
        let joined = if given.is_absolute() {
            given.to_path_buf()
        } else {
            self.root.join(given)
        };

        // Walk up to the deepest ancestor that exists (as a file, directory
        // or symlink); the rest is a tail of plain names to be created.
        let mut existing = joined.clone();
        let mut tail: Vec<OsString> = Vec::new();
        while existing.symlink_metadata().is_err() {
            match existing.components().next_back() {
                Some(Component::Normal(name)) => {
                    tail.push(name.to_os_string());
                    existing.pop();
                }
                Some(Component::CurDir) => {
                    existing.pop();
                }
                Some(Component::ParentDir) => {
                    return Err(PathError::UnresolvableParent {
                        path: path.to_string(),
                    })
                }
                _ => {
                    return Err(PathError::Escapes {
                        path: path.to_string(),
                        root: self.root.display().to_string(),
                    })
                }
            }
        }

        let mut resolved = std::fs::canonicalize(&existing).map_err(|source| PathError::Io {
            path: path.to_string(),
            source,
        })?;
        for name in tail.into_iter().rev() {
            resolved.push(name);
        }
        if !resolved.starts_with(&self.root) {
            return Err(PathError::Escapes {
                path: path.to_string(),
                root: self.root.display().to_string(),
            });
        }
        Ok(resolved)
    }

    /// Like [`Workspace::resolve`] but falls back to the lexical join, for
    /// declaring access manifests before the call is validated.
    pub fn declare(&self, path: &str) -> String {
        match self.resolve(path) {
            Ok(p) => p.display().to_string(),
            Err(_) => {
                let given = Path::new(path);
                if given.is_absolute() {
                    path.to_string()
                } else {
                    self.root.join(given).display().to_string()
                }
            }
        }
    }

    /// A path for display: relative to the root when inside it.
    pub fn display(&self, path: &Path) -> String {
        match path.strip_prefix(&self.root) {
            Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
            Ok(rel) => rel.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::new(dir.path()).unwrap();
        (dir, ws)
    }

    #[test]
    fn resolves_relative_and_new_paths() {
        let (_d, ws) = ws();
        std::fs::create_dir(ws.root().join("src")).unwrap();
        assert_eq!(ws.resolve("src").unwrap(), ws.root().join("src"));
        assert_eq!(
            ws.resolve("src/new/deep.txt").unwrap(),
            ws.root().join("src/new/deep.txt")
        );
        assert_eq!(
            ws.resolve("./src/../src/a").unwrap(),
            ws.root().join("src/a")
        );
        let abs = ws.root().join("src").display().to_string();
        assert_eq!(ws.resolve(&abs).unwrap(), ws.root().join("src"));
    }

    #[test]
    fn rejects_parent_escape() {
        let (_d, ws) = ws();
        let err = ws.resolve("../outside.txt").unwrap_err();
        assert!(matches!(err, PathError::Escapes { .. }), "{err}");
        assert!(ws.resolve("a/../../x").is_err());
        assert!(ws.resolve("").is_err());
    }

    #[test]
    fn rejects_absolute_outside() {
        let (_d, ws) = ws();
        let other = tempfile::tempdir().unwrap();
        let p = other.path().join("secret.txt");
        std::fs::write(&p, "synthetic").unwrap();
        assert!(ws.resolve(&p.display().to_string()).is_err());
        assert!(ws.resolve("/").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_pointing_outside() {
        let (_d, ws) = ws();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("secret.txt"), "synthetic").unwrap();
        std::os::unix::fs::symlink(other.path(), ws.root().join("link")).unwrap();
        let err = ws.resolve("link/secret.txt").unwrap_err();
        assert!(matches!(err, PathError::Escapes { .. }), "{err}");
        // New files through the outside symlink are refused as well.
        assert!(ws.resolve("link/new.txt").is_err());
        // A symlink that stays inside is fine.
        std::fs::create_dir(ws.root().join("real")).unwrap();
        std::os::unix::fs::symlink(ws.root().join("real"), ws.root().join("inner")).unwrap();
        assert_eq!(
            ws.resolve("inner/x").unwrap(),
            ws.root().join("real").join("x")
        );
    }
}
