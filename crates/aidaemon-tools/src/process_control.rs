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

/// Windows has no signals: ask the process tree to close (`taskkill /T`).
#[cfg(not(unix))]
pub fn send_sigterm(pid: u32) -> bool {
    taskkill(pid, false)
}

/// End the process tree at once (`taskkill /T /F`).
#[cfg(not(unix))]
pub fn send_sigkill(pid: u32) -> bool {
    taskkill(pid, true)
}

#[cfg(not(unix))]
fn taskkill(pid: u32, force: bool) -> bool {
    if pid == 0 {
        return false;
    }
    let mut command = std::process::Command::new("taskkill");
    command.args(["/PID", &pid.to_string(), "/T"]);
    if force {
        command.arg("/F");
    }
    command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
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

    #[cfg(not(unix))]
    {
        // The whole tree, as on Unix: a shell's children too.
        if !send_sigterm(pid) {
            let _ = child.start_kill();
        }
        if tokio::time::timeout(grace, child.wait()).await.is_err() {
            if !send_sigkill(pid) {
                let _ = child.start_kill();
            }
            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
        }
    }
}
