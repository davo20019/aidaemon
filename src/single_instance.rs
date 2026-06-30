//! Single-instance guard.
//!
//! Two daemons sharing one database silently race: each reclaims the other's
//! in-flight task as "orphaned" (so scheduled goals never finish), and they
//! fight over the terminal bridge. The failure is invisible — everything looks
//! up, but goals mysteriously die as `interrupted`. This guard makes a second
//! instance refuse to start instead.
//!
//! The lock is an advisory `flock` on a lock file beside the DB. The OS releases
//! it automatically when the holder exits — even on crash or `kill -9` — so
//! there is no stale-lock to clean up and no pidfile races. The held handle is
//! parked in a process-lifetime static so the lock is never dropped early.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;

static INSTANCE_LOCK: OnceLock<File> = OnceLock::new();

/// Acquire the single-instance lock at `lock_path`, held until this process
/// exits. Returns an error (naming the holder pid when recorded) if another
/// process already holds it; the caller should log it and refuse to start.
pub fn acquire(lock_path: &Path) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|e| anyhow::anyhow!("cannot open instance lock {}: {e}", lock_path.display()))?;

    if let Err(e) = try_lock_exclusive(&file) {
        let holder = std::fs::read_to_string(lock_path).unwrap_or_default();
        let holder = holder.trim();
        let who = if holder.is_empty() {
            String::new()
        } else {
            format!(" (pid {holder})")
        };
        anyhow::bail!(
            "another aidaemon instance is already running{who} — refusing to start a second one \
             that would race it over the same database and channels. Stop the other instance \
             first (lock: {}) [{e}]",
            lock_path.display()
        );
    }

    // Record our pid for diagnostics (best-effort; the lock itself is the guard).
    let _ = file.set_len(0);
    let _ = write!(file, "{}", std::process::id());
    let _ = file.flush();
    // Hold the handle for the whole process lifetime so the lock stays held.
    let _ = INSTANCE_LOCK.set(file);
    Ok(())
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // SAFETY: `file` owns a valid fd for the duration of this call.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> std::io::Result<()> {
    // No advisory flock on non-unix; single-instance is not enforced there.
    // The daemon's deployment targets are macOS/Linux, so this is acceptable.
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_on_a_held_lock_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aidaemon.db.lock");

        // Simulate another process holding the lock via a separate fd.
        let held = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        try_lock_exclusive(&held).expect("first holder acquires the lock");

        // A second acquire must be refused while the first holds it.
        assert!(
            acquire(&path).is_err(),
            "a second instance must be refused while the lock is held"
        );

        // Once the holder exits (fd closed), the lock is free again.
        drop(held);
        assert!(
            acquire(&path).is_ok(),
            "lock should be acquirable after the holder releases it"
        );
    }
}
