use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::AsyncReadExt;

use super::process_control::{configure_command_for_process_group, terminate_process_tree};

/// Directories to skip during recursive file walks.
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".cache",
    ".parcel-cache",
    "coverage",
    ".turbo",
    ".gradle",
    ".idea",
    ".vs",
];

/// Sensitive path patterns that should not be written to.
pub const SENSITIVE_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "credentials",
    "id_rsa",
    "id_ed25519",
];

/// Resolves `~` and validates path safety. Returns canonical path.
pub fn validate_path(path: &str) -> anyhow::Result<PathBuf> {
    let expanded = shellexpand::tilde(path).to_string();
    let p = PathBuf::from(&expanded);

    // Check for path traversal
    let normalized = if p.is_absolute() {
        p.clone()
    } else {
        std::env::current_dir()?.join(&p)
    };

    // Reject paths that try to traverse with ..
    let path_str = normalized.to_string_lossy();
    if path_str.contains("/../") || path_str.ends_with("/..") {
        anyhow::bail!("Path traversal detected: {}", path);
    }

    Ok(normalized)
}

/// Returns true if the path matches any sensitive pattern.
pub fn is_sensitive_path(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    SENSITIVE_PATTERNS.iter().any(|pat| path_str.contains(pat))
}

/// Check if a file appears to be binary by reading first 8KB and looking for null bytes.
pub async fn is_binary_file(path: &Path) -> anyhow::Result<bool> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 8192];
    let n = file.read(&mut buf).await?;
    Ok(buf[..n].contains(&0))
}

/// Format file content with line numbers.
pub fn format_with_line_numbers(content: &str, start_line: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = start_line + lines.len();
    let width = total.to_string().len().max(3);
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>width$} | {}", start_line + i + 1, line, width = width))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Output from running a shell command.
#[derive(Debug)]
pub struct CommandOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

/// Run a shell command with timeout. Returns structured output.
pub async fn run_cmd(
    cmd: &str,
    working_dir: Option<&Path>,
    timeout_secs: u64,
) -> anyhow::Result<CommandOutput> {
    let start = std::time::Instant::now();

    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    if let Some(dir) = working_dir {
        command.current_dir(dir);
    }
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    command.kill_on_drop(true);
    configure_command_for_process_group(&mut command);

    let mut child = command.spawn()?;
    let child_pid = child.id().unwrap_or(0);
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stderr"))?;

    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });

    let timeout = Duration::from_secs(timeout_secs);
    let mut timed_out = false;
    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => Some(status),
        Ok(Err(e)) => anyhow::bail!("Command failed to execute: {}", e),
        Err(_) => {
            timed_out = true;
            terminate_process_tree(child_pid, &mut child, Duration::from_secs(2)).await;
            None
        }
    };

    if timed_out {
        stdout_task.abort();
        stderr_task.abort();
        let _ = stdout_task.await;
        let _ = stderr_task.await;
        anyhow::bail!("Command timed out after {}s", timeout_secs);
    }

    let stdout = match stdout_task.await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).to_string(),
        _ => String::new(),
    };
    let stderr = match stderr_task.await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).to_string(),
        _ => String::new(),
    };
    let status = status.expect("status must exist when command did not time out");

    let duration_ms = start.elapsed().as_millis() as u64;
    Ok(CommandOutput {
        exit_code: status.code().unwrap_or(-1),
        stdout,
        stderr,
        duration_ms,
    })
}

/// Returns true if the directory entry name should be skipped.
pub fn should_skip_dir(name: &str) -> bool {
    DEFAULT_IGNORE_DIRS.contains(&name)
}

/// Check if a shell command contains operators that could be dangerous.
pub fn contains_shell_operator(cmd: &str) -> bool {
    for ch in [';', '|', '`', '\n'] {
        if cmd.contains(ch) {
            return true;
        }
    }
    for op in ["&&", "||", "$(", ">(", "<("] {
        if cmd.contains(op) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_validate_path_home() {
        let result = validate_path("~/test.txt");
        assert!(result.is_ok());
        let p = result.unwrap();
        assert!(p.is_absolute());
        assert!(!p.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_validate_path_traversal() {
        let result = validate_path("/tmp/../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_absolute() {
        let result = validate_path("/tmp/test.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_sensitive_path() {
        assert!(is_sensitive_path(Path::new("/home/user/.ssh/id_rsa")));
        assert!(is_sensitive_path(Path::new("/home/user/.env")));
        assert!(is_sensitive_path(Path::new("/home/user/.gnupg/key")));
        assert!(!is_sensitive_path(Path::new("/home/user/code/main.rs")));
    }

    #[tokio::test]
    async fn test_is_binary_file_text() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "Hello, world!\nLine 2\n").unwrap();
        assert!(!is_binary_file(f.path()).await.unwrap());
    }

    #[tokio::test]
    async fn test_is_binary_file_binary() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10]).unwrap();
        assert!(is_binary_file(f.path()).await.unwrap());
    }

    #[test]
    fn test_format_with_line_numbers() {
        let result = format_with_line_numbers("foo\nbar\nbaz", 0);
        assert!(result.contains("  1 | foo"));
        assert!(result.contains("  2 | bar"));
        assert!(result.contains("  3 | baz"));
    }

    #[test]
    fn test_format_with_line_numbers_offset() {
        let result = format_with_line_numbers("line10\nline11", 9);
        assert!(result.contains("10 | line10"));
        assert!(result.contains("11 | line11"));
    }

    #[tokio::test]
    async fn test_run_cmd_echo() {
        let out = run_cmd("echo hello", None, 5).await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[tokio::test]
    async fn test_run_cmd_timeout() {
        let result = run_cmd("sleep 10", None, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_cmd_timeout_kills_child_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked.txt");
        let cmd = format!(
            "(sleep 2; echo leaked > \"{}\") & wait",
            marker.to_string_lossy()
        );

        let result = run_cmd(&cmd, None, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));

        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(
            !marker.exists(),
            "timed out command should not leave detached child writing files"
        );
    }

    #[tokio::test]
    async fn test_run_cmd_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let out = run_cmd("pwd", Some(dir.path()), 5).await.unwrap();
        assert_eq!(out.exit_code, 0);
    }

    #[test]
    fn test_should_skip_dir() {
        assert!(should_skip_dir(".git"));
        assert!(should_skip_dir("node_modules"));
        assert!(should_skip_dir("target"));
        assert!(!should_skip_dir("src"));
        assert!(!should_skip_dir("lib"));
    }

    #[test]
    fn test_contains_shell_operator() {
        assert!(contains_shell_operator("ls; rm"));
        assert!(contains_shell_operator("cat | grep"));
        assert!(contains_shell_operator("a && b"));
        assert!(contains_shell_operator("echo $(cmd)"));
        assert!(!contains_shell_operator("cargo build --release"));
        assert!(!contains_shell_operator("ls -la /tmp"));
    }
}
