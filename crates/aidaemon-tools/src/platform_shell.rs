//! The shell a command line runs in: `sh -c` on Unix, Windows PowerShell on
//! Windows (always present, unlike `sh`). The command-effect checks are
//! written for POSIX shells and fail closed: a PowerShell line they cannot
//! read as a known observation or local work is treated as unknown, so it
//! asks first.

use tokio::process::Command;

/// Appended to every PowerShell script so the shell exits as `sh` does:
/// with the last program's own exit code. `-Command` alone turns any
/// failing last step into 1, so `python x.py` exiting 7 read as 1.
pub const POWERSHELL_EXIT_EPILOGUE: &str =
    "\nif (-not $?) { if ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 } }\n";

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
            &format!("{script}{POWERSHELL_EXIT_EPILOGUE}"),
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

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    async fn exit_code(script: &str) -> Option<i32> {
        shell_command(script).output().await.unwrap().status.code()
    }

    #[tokio::test]
    async fn the_last_programs_exit_code_comes_through() {
        assert_eq!(exit_code("cmd /c exit 7").await, Some(7));
        assert_eq!(
            exit_code("Get-Item C:/synthetic-missing-path").await,
            Some(1)
        );
        assert_eq!(
            exit_code("cmd /c exit 3; Write-Output after").await,
            Some(0)
        );
        assert_eq!(exit_code("Write-Output ok").await, Some(0));
    }
}
