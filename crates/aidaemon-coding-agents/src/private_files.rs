//! Owner-only scratch space for what a launch hands a coding agent: MCP
//! configs carrying the daemon's peer token, a run's Codex home. The shared
//! temp directory is world-listable on Linux, and a file written there and
//! then chmodded is readable in between, so these go under the user's own
//! `~/.aidaemon/run` (0700), each file created 0600 and new.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// `~/.aidaemon/run`, created owner-only (and narrowed if it was wider).
/// Falls back to the temp directory only when there is no home directory.
pub fn run_dir() -> std::io::Result<PathBuf> {
    let base = dirs::home_dir()
        .map(|home| home.join(".aidaemon").join("run"))
        .unwrap_or_else(|| std::env::temp_dir().join("aidaemon-run"));
    private_dir(&base)?;
    Ok(base)
}

/// Create `dir` (and its parents) and make it owner-only.
pub fn private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// A fresh directory under [`run_dir`], owner-only.
pub fn new_private_dir(prefix: &str) -> std::io::Result<PathBuf> {
    let dir = run_dir()?.join(format!("{prefix}-{}", uuid::Uuid::new_v4()));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&dir)?;
    Ok(dir)
}

/// Write `body` to a file that does not exist yet, owner-only from its
/// creation.
pub fn write_new_private_file(path: &Path, body: &[u8]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(body)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_private_file_is_new_and_owner_only() {
        let root = std::env::temp_dir().join(format!("aidaemon-private-{}", uuid::Uuid::new_v4()));
        let inner = root.join("run");
        private_dir(&inner).unwrap();
        let path = inner.join("mcp.json");
        write_new_private_file(&path, b"{\"token\":\"synthetic\"}").unwrap();
        assert!(write_new_private_file(&path, b"again").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(&inner), 0o700);
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
