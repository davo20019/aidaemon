use std::time::Duration;

/// Configure commands to run in their own process group (Unix).
/// This allows terminating the full process tree on cancel/timeout.
#[cfg(unix)]
pub fn configure_command_for_process_group(command: &mut tokio::process::Command) {
    // SAFETY: pre_exec runs in the child process after fork and before exec.
    // We only call the async-signal-safe setpgid syscall.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
pub fn configure_command_for_process_group(command: &mut tokio::process::Command) {
    let _ = command;
}

#[cfg(unix)]
fn send_signal_to_process_group_or_pid(pid: u32, signal: i32) -> bool {
    if pid == 0 {
        return false;
    }
    let raw_pid = pid as libc::pid_t;
    // Prefer signalling the process group (negative pid), fallback to direct pid.
    let group_ok = unsafe { libc::kill(-raw_pid, signal) == 0 };
    if group_ok {
        return true;
    }
    unsafe { libc::kill(raw_pid, signal) == 0 }
}

/// Send SIGTERM to a process group (or fallback process pid).
#[cfg(unix)]
pub fn send_sigterm(pid: u32) -> bool {
    send_signal_to_process_group_or_pid(pid, libc::SIGTERM)
}

/// Send SIGKILL to a process group (or fallback process pid).
#[cfg(unix)]
pub fn send_sigkill(pid: u32) -> bool {
    send_signal_to_process_group_or_pid(pid, libc::SIGKILL)
}

/// Send graceful termination to a process via taskkill.
#[cfg(windows)]
pub fn send_sigterm(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Force-kill a process via taskkill /F.
#[cfg(windows)]
pub fn send_sigkill(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Best-effort process tree termination with TERM->KILL escalation.
pub async fn terminate_process_tree(pid: u32, child: &mut tokio::process::Child, grace: Duration) {
    #[cfg(unix)]
    {
        if !send_sigterm(pid) {
            let _ = child.start_kill();
        }

        let exited = tokio::time::timeout(grace, child.wait()).await;
        if exited.is_err() {
            let _ = send_sigkill(pid);
            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
        }
    }

    #[cfg(windows)]
    {
        // Try graceful taskkill first, then force-kill if needed
        if !send_sigterm(pid) {
            let _ = child.start_kill();
        }

        let exited = tokio::time::timeout(grace, child.wait()).await;
        if exited.is_err() {
            let _ = send_sigkill(pid);
            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
        }
    }
}
