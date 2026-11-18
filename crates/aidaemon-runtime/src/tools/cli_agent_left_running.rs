//! Processes a delegated CLI agent left running after its run ended.
//!
//! The daemon never sees the commands a CLI agent (Claude Code, Codex,
//! Gemini CLI, ...) runs, so it cannot judge them before they run; it checks
//! afterwards instead. The agent starts with the run's id in its environment
//! ([`RUN_TAG_VAR`]). Every descendant that keeps its environment carries it:
//! `&`, `disown`, `nohup`, `setsid`, a double fork, and a tmux or screen
//! server the run itself started (with the shells in it). After the run
//! ends, [`find_left_running`] lists the live processes of the daemon's own
//! user that carry the tag, are still in the agent's process group, or
//! descend from either. They are reported, never stopped: a dev server the
//! agent started on purpose is often the point of the run.
//!
//! What escapes it: a process another parent started for the agent (launchd
//! through `launchctl` or `open`, a tmux or screen server that was already
//! running, a container runtime, a system service), and a process that
//! cleared its environment (`env -i`) and left the agent's process group.
//! On macOS the kernel does not show the environment of Apple platform
//! binaries (`/bin/sleep`, `/usr/bin/perl`), so one of those is found only
//! while it is in the agent's group or under a tagged process it can see.
//! Linux reads `/proc`; macOS runs `ps`; other systems have no check.

#![cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]

use std::collections::{BTreeSet, HashMap};

/// The environment variable a delegated CLI agent's processes carry.
pub(crate) const RUN_TAG_VAR: &str = "AIDAEMON_CLI_RUN";

/// How many processes a report lists by name; the rest are counted.
const MAX_LISTED: usize = 10;

/// How much of a command line a report keeps.
const MAX_COMMAND_CHARS: usize = 200;

/// One process the run left running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LeftRunningProcess {
    pub pid: u32,
    pub command: String,
}

/// The processes a run left running: the first [`MAX_LISTED`] by pid, and
/// how many there are in all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LeftRunning {
    pub processes: Vec<LeftRunningProcess>,
    pub total: usize,
}

impl LeftRunning {
    pub(crate) fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The section the model reads in the run's result.
    pub(crate) fn result_section(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut text = String::from(
            "## Left running\nThe agent's run has ended, but these processes it started are \
             still running. The daemon did not stop them. Tell the owner what each one is and \
             whether it should keep running (stop one with `kill <pid>`):",
        );
        for process in &self.processes {
            text.push_str(&format!("\n- pid {}: {}", process.pid, process.command));
        }
        let unlisted = self.total.saturating_sub(self.processes.len());
        if unlisted > 0 {
            text.push_str(&format!("\n- and {unlisted} more"));
        }
        Some(text)
    }

    /// The short line the owner reads with the run's end.
    pub(crate) fn owner_line(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let named: Vec<String> = self
            .processes
            .iter()
            .take(3)
            .map(|process| format!("pid {} ({})", process.pid, shorten(&process.command, 60)))
            .collect();
        let unlisted = self.total.saturating_sub(named.len());
        let more = if unlisted > 0 {
            format!(" and {unlisted} more")
        } else {
            String::new()
        };
        Some(format!(
            "Still running after the agent finished: {}{more}.",
            named.join(", ")
        ))
    }
}

/// A live process as the scan read it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessRow {
    pid: u32,
    ppid: u32,
    pgid: u32,
    uid: u32,
    zombie: bool,
    tagged: bool,
    command: String,
}

/// The processes the run left: of the daemon's user, alive, carrying the
/// tag or in the agent's process `group`, and everything under them.
/// `group` is the agent's own pid; it has been waited for, so a process now
/// holding that pid is a reuse, not the agent.
fn select(rows: &[ProcessRow], group: Option<u32>, own_pid: u32, own_uid: u32) -> LeftRunning {
    let candidates: Vec<&ProcessRow> = rows
        .iter()
        .filter(|row| {
            row.uid == own_uid
                && !row.zombie
                && row.pid != own_pid
                && row.pid != 0
                && Some(row.pid) != group
        })
        .collect();
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for row in &candidates {
        children.entry(row.ppid).or_default().push(row.pid);
    }
    let mut found: BTreeSet<u32> = candidates
        .iter()
        .filter(|row| row.tagged || (group.is_some_and(|g| g != 0) && Some(row.pgid) == group))
        .map(|row| row.pid)
        .collect();
    let mut pending: Vec<u32> = found.iter().copied().collect();
    while let Some(pid) = pending.pop() {
        for child in children.get(&pid).into_iter().flatten() {
            if found.insert(*child) {
                pending.push(*child);
            }
        }
    }
    let commands: HashMap<u32, &str> = candidates
        .iter()
        .map(|row| (row.pid, row.command.as_str()))
        .collect();
    LeftRunning {
        total: found.len(),
        processes: found
            .iter()
            .take(MAX_LISTED)
            .map(|pid| LeftRunningProcess {
                pid: *pid,
                command: shorten(commands.get(pid).copied().unwrap_or(""), MAX_COMMAND_CHARS),
            })
            .collect(),
    }
}

fn shorten(text: &str, max_chars: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(max_chars).collect();
    cut.push('…');
    cut
}

/// Split `count` whitespace-separated fields off the front of `line`; the
/// rest, trimmed, is the last element.
fn leading_fields(line: &str, count: usize) -> Option<(Vec<&str>, &str)> {
    let mut fields = Vec::with_capacity(count);
    let mut rest = line.trim_start();
    for _ in 0..count {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        fields.push(&rest[..end]);
        rest = rest[end..].trim_start();
    }
    Some((fields, rest.trim_end()))
}

/// The pids in `ps -E -o pid=,command=` output (macOS: the environment
/// follows the command line) whose environment holds `RUN_TAG_VAR=tag`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn tagged_pids_from_ps_env(output: &str, tag: &str) -> BTreeSet<u32> {
    let wanted = format!("{RUN_TAG_VAR}={tag}");
    output
        .lines()
        .filter_map(|line| {
            let (fields, rest) = leading_fields(line, 1)?;
            let pid = fields[0].parse::<u32>().ok()?;
            rest.split_whitespace()
                .any(|token| token == wanted)
                .then_some(pid)
        })
        .collect()
}

/// Rows from `ps -o pid=,ppid=,pgid=,uid=,stat=,command=` output.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn rows_from_ps(output: &str, tagged: &BTreeSet<u32>) -> Vec<ProcessRow> {
    output
        .lines()
        .filter_map(|line| {
            let (fields, command) = leading_fields(line, 5)?;
            let pid = fields[0].parse().ok()?;
            Some(ProcessRow {
                pid,
                ppid: fields[1].parse().ok()?,
                pgid: fields[2].parse().ok()?,
                uid: fields[3].parse().ok()?,
                zombie: fields[4].starts_with('Z'),
                tagged: tagged.contains(&pid),
                command: command.to_string(),
            })
        })
        .collect()
}

/// State, parent pid and process group from `/proc/<pid>/stat`. The command
/// name sits in parentheses and may hold spaces or parentheses itself, so
/// the fields are read after the last `)`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn proc_stat_fields(stat: &str) -> Option<(char, u32, u32)> {
    let after = &stat[stat.rfind(')')? + 1..];
    let mut fields = after.split_whitespace();
    let state = fields.next()?.chars().next()?;
    let ppid = fields.next()?.parse().ok()?;
    let pgid = fields.next()?.parse().ok()?;
    Some((state, ppid, pgid))
}

/// The command name in `/proc/<pid>/stat`, for a process with no argv.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn proc_stat_name(stat: &str) -> &str {
    match (stat.find('('), stat.rfind(')')) {
        (Some(open), Some(close)) if open < close => &stat[open + 1..close],
        _ => "",
    }
}

/// Whether a NUL-separated `/proc/<pid>/environ` holds `RUN_TAG_VAR=tag`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn environ_has_tag(environ: &[u8], tag: &str) -> bool {
    let wanted = format!("{RUN_TAG_VAR}={tag}");
    environ
        .split(|byte| *byte == 0)
        .any(|entry| entry == wanted.as_bytes())
}

/// A NUL-separated `/proc/<pid>/cmdline` as one line.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cmdline_text(cmdline: &[u8]) -> String {
    cmdline
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(String::from_utf8_lossy)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(unix)]
fn own_uid() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { libc::getuid() }
}

#[cfg(target_os = "macos")]
fn read_rows(tag: &str) -> Option<Vec<ProcessRow>> {
    let ps = |args: &[&str]| -> Option<String> {
        let output = std::process::Command::new("/bin/ps")
            .args(args)
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    };
    // The environment first: a process that starts between the two reads is
    // simply untagged.
    let tagged = tagged_pids_from_ps_env(&ps(&["-ax", "-ww", "-E", "-o", "pid=,command="])?, tag);
    let rows = ps(&["-ax", "-ww", "-o", "pid=,ppid=,pgid=,uid=,stat=,command="])?;
    Some(rows_from_ps(&rows, &tagged))
}

#[cfg(target_os = "linux")]
fn read_rows(tag: &str) -> Option<Vec<ProcessRow>> {
    use std::os::unix::fs::MetadataExt;
    let mut rows = Vec::new();
    for entry in std::fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let dir = entry.path();
        let Ok(metadata) = std::fs::metadata(&dir) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(dir.join("stat")) else {
            continue;
        };
        let Some((state, ppid, pgid)) = proc_stat_fields(&stat) else {
            continue;
        };
        // Another user's environment is unreadable; it is untagged.
        let tagged = std::fs::read(dir.join("environ"))
            .map(|environ| environ_has_tag(&environ, tag))
            .unwrap_or(false);
        let command = std::fs::read(dir.join("cmdline"))
            .map(|cmdline| cmdline_text(&cmdline))
            .unwrap_or_default();
        rows.push(ProcessRow {
            pid,
            ppid,
            pgid,
            uid: metadata.uid(),
            zombie: matches!(state, 'Z' | 'X'),
            tagged,
            command: if command.is_empty() {
                format!("[{}]", proc_stat_name(&stat))
            } else {
                command
            },
        });
    }
    Some(rows)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn scan(tag: &str, group: Option<u32>) -> LeftRunning {
    match read_rows(tag) {
        Some(rows) => select(&rows, group, std::process::id(), own_uid()),
        None => LeftRunning::default(),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn scan(_tag: &str, _group: Option<u32>) -> LeftRunning {
    LeftRunning::default()
}

/// The live processes the run tagged `tag` left behind, after its agent
/// (`group`, its pid and process group) was waited for.
pub(crate) async fn find_left_running(tag: String, group: Option<u32>) -> LeftRunning {
    tokio::task::spawn_blocking(move || scan(&tag, group))
        .await
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, ppid: u32, pgid: u32, tagged: bool, command: &str) -> ProcessRow {
        ProcessRow {
            pid,
            ppid,
            pgid,
            uid: 501,
            zombie: false,
            tagged,
            command: command.to_string(),
        }
    }

    #[test]
    fn macos_ps_environment_marks_only_the_exact_tag() {
        let env_output = "\
  101 node server.js --port 3000 HOME=/Users/alice AIDAEMON_CLI_RUN=run-a PATH=/usr/bin
  102 sleep 300
  103 python3 -m http.server AIDAEMON_CLI_RUN=run-ab
  104 echo AIDAEMON_CLI_RUN=run-a-not HOME=/Users/alice
   99 vim notes.md TERM=xterm AIDAEMON_CLI_RUN=run-a
";
        let tagged = tagged_pids_from_ps_env(env_output, "run-a");
        assert_eq!(tagged.into_iter().collect::<Vec<_>>(), vec![99, 101]);
    }

    #[test]
    fn macos_ps_rows_keep_spaced_commands_and_zombies() {
        let tagged: BTreeSet<u32> = [101].into_iter().collect();
        let output = "\
  101     1   100   501 S    node server.js --port 3000
  102   101   100   501 Z+   (sleep)
  103     1   103     0 Ss   /usr/sbin/cron
 garbage line
";
        let rows = rows_from_ps(output, &tagged);
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[0],
            row(101, 1, 100, true, "node server.js --port 3000")
        );
        assert!(rows[1].zombie);
        assert_eq!(rows[2].uid, 0);
    }

    #[test]
    fn linux_proc_fields_are_read_after_the_command_name() {
        let stat = "4242 (my (odd) server) S 1 4200 4200 0 -1 4194560 152 0 0 0";
        assert_eq!(proc_stat_fields(stat), Some(('S', 1, 4200)));
        assert_eq!(proc_stat_name(stat), "my (odd) server");
        let environ = b"HOME=/home/alice\0AIDAEMON_CLI_RUN=run-a\0PATH=/usr/bin\0";
        assert!(environ_has_tag(environ, "run-a"));
        assert!(!environ_has_tag(environ, "run"));
        assert!(!environ_has_tag(b"AIDAEMON_CLI_RUN=run-ab\0", "run-a"));
        assert_eq!(
            cmdline_text(b"node\0server.js\0--port\x003000\0"),
            "node server.js --port 3000"
        );
    }

    #[test]
    fn selection_takes_tagged_grouped_and_their_descendants_of_the_owner() {
        let mut other_user = row(300, 1, 300, true, "someone else's");
        other_user.uid = 0;
        let mut zombie = row(301, 1, 200, true, "(defunct)");
        zombie.zombie = true;
        let rows = vec![
            // Detached with setsid: its own group, found by its tag.
            row(210, 1, 210, true, "node server.js"),
            // A platform binary whose environment is hidden, under it.
            row(211, 210, 210, false, "/bin/sleep 300"),
            // Left in the agent's group without a visible tag.
            row(220, 1, 200, false, "sleep 300"),
            // Holds the agent's reaped pid: a reuse.
            row(200, 1, 200, false, "unrelated"),
            // Unrelated work of the same user.
            row(400, 1, 400, false, "zsh"),
            // The daemon itself.
            row(42, 1, 42, true, "aidaemon"),
            other_user,
            zombie,
        ];
        let found = select(&rows, Some(200), 42, 501);
        assert_eq!(found.total, 3);
        let pids: Vec<u32> = found.processes.iter().map(|p| p.pid).collect();
        assert_eq!(pids, vec![210, 211, 220]);
        let line = found.owner_line().unwrap();
        assert!(line.contains("pid 210 (node server.js)"), "{line}");
        assert!(
            found
                .result_section()
                .unwrap()
                .contains("- pid 220: sleep 300")
        );
        assert!(
            select(&rows, None, 42, 501)
                .processes
                .iter()
                .all(|p| p.pid != 220)
        );
        assert!(select(&[], Some(200), 42, 501).result_section().is_none());
    }

    #[test]
    fn a_long_list_is_capped_and_counted() {
        let rows: Vec<ProcessRow> = (1000..1015)
            .map(|pid| row(pid, 1, pid, true, &"x".repeat(500)))
            .collect();
        let found = select(&rows, None, 1, 501);
        assert_eq!(found.total, 15);
        assert_eq!(found.processes.len(), MAX_LISTED);
        assert!(found.processes[0].command.chars().count() <= MAX_COMMAND_CHARS + 1);
        assert!(found.result_section().unwrap().ends_with("- and 5 more"));
        assert!(found.owner_line().unwrap().contains("and 12 more"));
    }

    /// Run by [`a_detached_tagged_process_is_found_by_its_tag`] as the
    /// process the agent leaves behind: this test binary is not an Apple
    /// platform binary, so macOS shows its environment.
    #[test]
    #[ignore = "helper process for a_detached_tagged_process_is_found_by_its_tag"]
    fn left_running_helper_waits() {
        if std::env::var_os("AIDAEMON_LEFT_RUNNING_HELPER").is_some() {
            std::thread::sleep(std::time::Duration::from_secs(120));
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_detached_tagged_process_is_found_by_its_tag() {
        use std::os::unix::process::CommandExt;
        let tag = uuid::Uuid::new_v4().to_string();
        let exe = std::env::current_exe().unwrap();
        // A double fork: the helper is reparented and its launcher exits,
        // as `(cmd &)` in an agent's shell would leave it.
        let status = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("(\"$0\" --exact --ignored tools::cli_agent_left_running::tests::left_running_helper_waits >/dev/null 2>&1 &)")
            .arg(&exe)
            .env(RUN_TAG_VAR, &tag)
            .env("AIDAEMON_LEFT_RUNNING_HELPER", "1")
            .process_group(0)
            .status()
            .unwrap();
        assert!(status.success());

        let mut found = LeftRunning::default();
        for _ in 0..100 {
            // No group: the helper must be found by its environment alone.
            found = scan(&tag, None);
            if !found.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        for process in &found.processes {
            // SAFETY: kill only signals the helper this test started.
            unsafe { libc::kill(process.pid as libc::pid_t, libc::SIGKILL) };
        }
        assert_eq!(found.total, 1, "{found:?}");
        assert!(
            found.processes[0]
                .command
                .contains("--exact --ignored tools::cli_agent_left_running"),
            "{found:?}"
        );
        assert!(scan(&uuid::Uuid::new_v4().to_string(), None).is_empty());
    }
}
