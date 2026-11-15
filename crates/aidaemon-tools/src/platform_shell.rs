//! The shell a command line runs in: `sh -c` on Unix, Windows PowerShell on
//! Windows (always present, unlike `sh`). The command-effect checks are
//! written for POSIX shells and fail closed: a PowerShell line they cannot
//! read as a known observation or local work is treated as unknown, so it
//! asks first.

use tokio::process::Command;

/// The shell's name, as the owner and the model see it.
pub fn shell_name() -> &'static str {
    if cfg!(windows) { "powershell" } else { "sh" }
}

/// `script` run by this platform's shell.
pub fn shell_command(script: &str) -> Command {
    if cfg!(windows) {
        let mut command = Command::new("powershell.exe");
        command.args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]);
        command
    } else {
        let mut command = Command::new("sh");
        command.arg("-c").arg(script);
        command
    }
}

/// Whether this platform's shell is there to run commands.
pub fn shell_available() -> bool {
    if cfg!(windows) {
        true
    } else {
        std::path::Path::new("/bin/sh").exists()
    }
}
