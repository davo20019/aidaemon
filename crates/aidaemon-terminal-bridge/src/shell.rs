//! The PTY shell process an agent runs in, and its events.

use super::*;

pub(crate) struct ShellProcess {
    pub(crate) child: Box<dyn portable_pty::Child + Send + Sync>,
    pub(crate) stdin_tx: Option<Sender<Vec<u8>>>,
    pub(crate) master: Box<dyn MasterPty + Send>,
}

#[derive(Debug)]
pub(crate) enum ShellEvent {
    Output {
        session_id: String,
        data: String,
    },
    ReviewProgress {
        session_id: String,
        request_id: String,
        stage: String,
        message: String,
    },
    ReviewResult {
        session_id: String,
        request_id: String,
        payload: Value,
    },
    ReviewError {
        session_id: String,
        request_id: String,
        code: String,
        message: String,
    },
    ReviewStream {
        session_id: String,
        request_id: String,
        stream: String,
        data: String,
    },
}

impl ShellProcess {
    pub(crate) async fn spawn(
        shell: &str,
        cwd: &Path,
        session_id: &str,
        events_tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    ) -> anyhow::Result<Self> {
        let pty_system = NativePtySystem::default();
        let pair = pty_system.openpty(PtySize {
            rows: PTY_DEFAULT_ROWS,
            cols: PTY_DEFAULT_COLS,
            pixel_width: 0,
            pixel_height: 0,
        })?;

        let mut cmd = CommandBuilder::new(shell);
        if cfg!(windows) {
            cmd.arg("-NoLogo");
        } else {
            cmd.arg("-i");
        }
        cmd.cwd(cwd);
        // Start from a clean environment so parent process/session markers
        // (including Claude nesting vars) cannot leak into the bridged PTY.
        cmd.env_clear();
        // Prevent nested Claude Code session detection when launching `claude`
        // inside the bridged terminal shell.
        cmd.env_remove("CLAUDECODE");
        cmd.env_remove("CLAUDE_CODE");
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }

        let child = pair.slave.spawn_command(cmd)?;
        drop(pair.slave);

        let mut reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;

        let (stdin_tx, stdin_rx) = mpsc::channel::<Vec<u8>>();
        let read_session_id = session_id.to_string();
        std::thread::Builder::new()
            .name("terminal-bridge-pty-reader".to_string())
            .spawn(move || {
                let mut buf = [0u8; 4096];
                let mut utf8_carry = Vec::<u8>::new();
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let decoded = decode_utf8_stream_chunk(&mut utf8_carry, &buf[..n]);
                            if decoded.is_empty() {
                                continue;
                            }
                            if events_tx
                                .send(ShellEvent::Output {
                                    session_id: read_session_id.clone(),
                                    data: decoded,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !utf8_carry.is_empty() {
                    let tail = String::from_utf8_lossy(&utf8_carry).to_string();
                    if !tail.is_empty() {
                        let _ = events_tx.send(ShellEvent::Output {
                            session_id: read_session_id.clone(),
                            data: tail,
                        });
                    }
                }
            })?;

        std::thread::Builder::new()
            .name("terminal-bridge-pty-writer".to_string())
            .spawn(move || {
                let mut writer = writer;
                while let Ok(bytes) = stdin_rx.recv() {
                    if bytes.is_empty() {
                        continue;
                    }
                    if writer.write_all(&bytes).is_err() {
                        break;
                    }
                    let _ = writer.flush();
                }
            })?;

        Ok(Self {
            child,
            stdin_tx: Some(stdin_tx),
            master: pair.master,
        })
    }

    pub(crate) async fn write_stdin(&mut self, data: &str) -> anyhow::Result<()> {
        let Some(tx) = self.stdin_tx.as_ref() else {
            anyhow::bail!("shell stdin channel closed");
        };
        tx.send(data.as_bytes().to_vec())
            .map_err(|_| anyhow::anyhow!("failed to send stdin to PTY writer"))?;
        Ok(())
    }

    pub(crate) fn resize(&mut self, cols: u16, rows: u16) -> anyhow::Result<()> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        self.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        Ok(())
    }

    pub(crate) async fn stop(&mut self) {
        self.stdin_tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub(crate) fn try_wait_code(&mut self) -> anyhow::Result<Option<i32>> {
        match self.child.try_wait()? {
            Some(status) => Ok(Some(exit_status_code(status))),
            None => Ok(None),
        }
    }
}

pub(crate) fn decode_utf8_stream_chunk(carry: &mut Vec<u8>, incoming: &[u8]) -> String {
    if incoming.is_empty() {
        return String::new();
    }
    carry.extend_from_slice(incoming);
    let mut out = String::new();
    loop {
        match std::str::from_utf8(carry.as_slice()) {
            Ok(valid) => {
                out.push_str(valid);
                carry.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0
                    && let Ok(prefix) = std::str::from_utf8(&carry[..valid_up_to])
                {
                    out.push_str(prefix);
                }
                if let Some(error_len) = err.error_len() {
                    let drain_to = valid_up_to.saturating_add(error_len).min(carry.len());
                    let invalid = &carry[valid_up_to..drain_to];
                    if invalid.len() == 1 {
                        match invalid[0] {
                            // C1 controls can appear in terminal streams (not UTF-8 text).
                            // Normalize the common ones instead of printing replacement glyphs.
                            0x9B => out.push_str("\u{001b}["),
                            0x9D => out.push_str("\u{001b}]"),
                            0x90 => out.push_str("\u{001b}P"),
                            0x9C => out.push_str("\u{001b}\\"),
                            b if (0x80..=0x9F).contains(&b) => {}
                            _ => out.push('\u{FFFD}'),
                        }
                    } else {
                        out.push('\u{FFFD}');
                    }
                    carry.drain(..drain_to);
                    if carry.is_empty() {
                        break;
                    }
                } else {
                    // Incomplete UTF-8 sequence at the end. Keep trailing bytes for next read.
                    carry.drain(..valid_up_to);
                    break;
                }
            }
        }
    }
    out
}

pub(crate) fn exit_status_code(status: ExitStatus) -> i32 {
    status.exit_code() as i32
}

pub(crate) fn build_child_env() -> Vec<(String, String)> {
    let keys = [
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "COLORTERM",
        "XDG_RUNTIME_DIR",
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
        "EDITOR",
        "VISUAL",
    ];
    let mut out = Vec::new();
    for key in keys {
        if let Ok(v) = std::env::var(key)
            && !v.is_empty()
        {
            out.push((key.to_string(), v));
        }
    }
    // PTY sessions should always advertise terminal capabilities even when
    // aidaemon is started from a non-interactive context.
    out.push(("TERM".to_string(), "xterm-256color".to_string()));
    if !out.iter().any(|(k, _)| k == "COLORTERM") {
        out.push(("COLORTERM".to_string(), "truecolor".to_string()));
    }
    // Ensure Node.js-based tools (chalk, kleur, etc.) emit colors even when
    // their own TTY detection is overly cautious.
    out.push(("FORCE_COLOR".to_string(), "1".to_string()));
    out
}
