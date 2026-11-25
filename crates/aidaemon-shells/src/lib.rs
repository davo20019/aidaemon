//! Conversation shells: one pseudo-terminal per conversation, started in
//! that conversation's folder and kept running while the owner looks at
//! another one.
//!
//! The owner's apps attach to a shell (its output so far, then what it
//! writes as it writes it) and type into it; the agent reads its screen and
//! may type into it once the owner allows. The [`ShellRegistry`] owns every
//! shell; it knows nothing about the client gateway, its wire format or the
//! agent's tools.
//!
//! Invariants:
//! - At most one running shell per conversation; opening it again attaches
//!   to the one that runs.
//! - A shell starts with a clean environment (the owner's login settings,
//!   never the daemon's variables or secrets).
//! - A shell ends when the owner closes it, its program exits or its
//!   conversation goes; never on a clock.

// Feature unification only (cargo-hakari); keeps the unused-dependency lint quiet.
use workspace_hack as _;

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use portable_pty::{ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Variables a shell keeps from the daemon's environment: who and where the
/// owner is and how text is encoded. Everything else (keys, tokens, the
/// daemon's own settings) is left out; a login shell sets the rest.
const KEPT_VARIABLES: &[&str] = &[
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "PATH",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    // Windows: where the system and the owner's folders are.
    "SystemRoot",
    "SystemDrive",
    "windir",
    "ComSpec",
    "PATHEXT",
    "USERPROFILE",
    "USERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "ProgramData",
    "ProgramFiles",
    "TEMP",
    "TMP",
];

/// How shells behave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellsConfig {
    /// Shells may be opened at all.
    pub enabled: bool,
    /// Shells running at once, across conversations.
    pub max_shells: usize,
    /// Output kept per shell for an app that attaches later, in bytes.
    pub backlog_bytes: usize,
    /// Lines the agent's view of a shell keeps above its screen.
    pub scrollback_lines: usize,
}

impl Default for ShellsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_shells: 16,
            backlog_bytes: 512 * 1024,
            scrollback_lines: 2000,
        }
    }
}

/// A shell's id: random, never derived from its conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShellId(String);

impl ShellId {
    fn random() -> Self {
        Self(format!("sh_{}", uuid::Uuid::new_v4().simple()))
    }

    pub fn from_string(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ShellId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A shell as the registry describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellInfo {
    pub id: ShellId,
    pub session_id: String,
    /// The folder it started in.
    pub folder: PathBuf,
    /// What the program called its window (often the folder it is in), or
    /// empty.
    pub title: String,
    pub cols: u16,
    pub rows: u16,
    /// `None` while it runs; its exit code once it ended.
    pub exit_code: Option<u32>,
    pub started_ms: u64,
    /// When it last wrote anything.
    pub last_output_ms: u64,
}

impl ShellInfo {
    pub fn running(&self) -> bool {
        self.exit_code.is_none()
    }
}

/// What changed among the shells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellChange {
    /// Opened, retitled, resized or ended.
    Updated(ShellInfo),
    /// Gone: closed, or its conversation went.
    Closed { id: ShellId, session_id: String },
}

/// An app attached to a shell: what it wrote so far, then what it writes.
pub struct Attachment {
    /// The output kept so far, to replay into a fresh terminal.
    pub backlog: String,
    /// What it writes from now on. A receiver that falls behind gets
    /// `Lagged` and should attach again.
    pub output: broadcast::Receiver<Arc<str>>,
}

/// The agent's view of a shell: its screen as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenText {
    /// Lines above the screen (oldest first), up to what was asked for.
    pub above: Vec<String>,
    /// The screen's lines, trailing blank lines left out.
    pub screen: Vec<String>,
    /// Where the cursor is on the screen (row, column), from 0.
    pub cursor: (u16, u16),
    /// A full-screen program (an editor, a pager) has the screen.
    pub full_screen: bool,
}

/// Why a shell could not be opened or used.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShellError {
    #[error("shells are turned off")]
    Disabled,
    #[error("no such shell")]
    NotFound,
    #[error("that shell has ended")]
    Ended,
    #[error("too many shells are open")]
    TooMany,
    #[error("the folder does not exist")]
    NoFolder,
    #[error("{0}")]
    Failed(String),
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn nonzero(n: u16) -> std::num::NonZeroU16 {
    std::num::NonZeroU16::new(n.max(1)).unwrap_or(std::num::NonZeroU16::MIN)
}

/// The title a program sets on its window, caught as the screen parses.
#[derive(Default)]
struct TitleWatch {
    title: Option<String>,
}

impl vt100::Callbacks for TitleWatch {
    fn set_window_title(&mut self, _: &mut vt100::Screen, title: &[u8]) {
        let title: String = String::from_utf8_lossy(title)
            .chars()
            .filter(|c| !c.is_control())
            .take(120)
            .collect();
        self.title = Some(title);
    }
}

/// Bytes to text across reads: a character cut between two reads waits
/// for the rest; bytes that are no text become U+FFFD.
#[derive(Default)]
struct Utf8Carry {
    pending: Vec<u8>,
}

impl Utf8Carry {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut rest: &[u8] = &self.pending;
        loop {
            match std::str::from_utf8(rest) {
                Ok(text) => {
                    out.push_str(text);
                    rest = &[];
                    break;
                }
                Err(error) => {
                    let (good, after) = rest.split_at(error.valid_up_to());
                    // Valid by `valid_up_to`.
                    out.push_str(std::str::from_utf8(good).unwrap_or_default());
                    match error.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            rest = &after[bad..];
                        }
                        // Cut short: the rest comes with the next read.
                        None => {
                            rest = after;
                            break;
                        }
                    }
                }
            }
        }
        self.pending = rest.to_vec();
        out
    }
}

/// What a shell wrote: kept for late attachers, parsed for the agent, sent
/// to whoever is attached. One lock, so an attach never misses or repeats
/// a write.
struct Output {
    backlog: String,
    limit: usize,
    parser: vt100::Parser<TitleWatch>,
    tx: broadcast::Sender<Arc<str>>,
}

impl Output {
    fn write(&mut self, text: &str) {
        self.parser.process(text.as_bytes());
        self.backlog.push_str(text);
        if self.backlog.len() > self.limit {
            // Cut at a line start so the replay never begins inside an
            // escape sequence (which ends before a line does).
            let mut cut = self.backlog.len() - self.limit;
            while !self.backlog.is_char_boundary(cut) {
                cut += 1;
            }
            let cut = self.backlog[cut..]
                .find('\n')
                .map(|at| cut + at + 1)
                .unwrap_or(cut);
            self.backlog.drain(..cut);
        }
        let _ = self.tx.send(Arc::from(text));
    }
}

struct Shell {
    info: Mutex<ShellInfo>,
    output: Mutex<Output>,
    writer: Mutex<Box<dyn Write + Send>>,
    master: Mutex<Box<dyn MasterPty + Send>>,
    killer: Mutex<Box<dyn ChildKiller + Send + Sync>>,
}

struct Inner {
    config: ShellsConfig,
    shells: Mutex<HashMap<ShellId, Arc<Shell>>>,
    changes: broadcast::Sender<ShellChange>,
}

/// Every conversation's shell.
#[derive(Clone)]
pub struct ShellRegistry {
    inner: Arc<Inner>,
}

impl ShellRegistry {
    pub fn new(config: ShellsConfig) -> Self {
        let (changes, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(Inner {
                config,
                shells: Mutex::new(HashMap::new()),
                changes,
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.inner.config.enabled
    }

    /// Changes to any shell, from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<ShellChange> {
        self.inner.changes.subscribe()
    }

    pub fn list(&self) -> Vec<ShellInfo> {
        let mut all: Vec<ShellInfo> = lock(&self.inner.shells)
            .values()
            .map(|s| lock(&s.info).clone())
            .collect();
        all.sort_by_key(|a| a.started_ms);
        all
    }

    pub fn get(&self, id: &ShellId) -> Option<ShellInfo> {
        self.shell(id).map(|s| lock(&s.info).clone())
    }

    /// The conversation's shell, running or ended.
    pub fn for_session(&self, session_id: &str) -> Option<ShellInfo> {
        lock(&self.inner.shells)
            .values()
            .map(|s| lock(&s.info).clone())
            .find(|info| info.session_id == session_id)
    }

    fn shell(&self, id: &ShellId) -> Option<Arc<Shell>> {
        lock(&self.inner.shells).get(id).cloned()
    }

    /// The conversation's running shell, or a new one in `folder` (an
    /// ended one is replaced).
    pub fn open(
        &self,
        session_id: &str,
        folder: &Path,
        cols: u16,
        rows: u16,
    ) -> Result<ShellInfo, ShellError> {
        if !self.enabled() {
            return Err(ShellError::Disabled);
        }
        if let Some(info) = self.for_session(session_id) {
            if info.running() {
                return Ok(info);
            }
            self.forget(&info.id);
        }
        if !folder.is_dir() {
            return Err(ShellError::NoFolder);
        }
        let running = lock(&self.inner.shells).len();
        if running >= self.inner.config.max_shells {
            return Err(ShellError::TooMany);
        }
        let (cols, rows) = (cols.clamp(2, 1000), rows.clamp(1, 500));
        let shell = self.spawn(session_id, folder, cols, rows)?;
        let info = lock(&shell.info).clone();
        lock(&self.inner.shells).insert(info.id.clone(), shell);
        info!(shell = %info.id, session_id, "Conversation shell started");
        let _ = self.inner.changes.send(ShellChange::Updated(info.clone()));
        Ok(info)
    }

    fn spawn(
        &self,
        session_id: &str,
        folder: &Path,
        cols: u16,
        rows: u16,
    ) -> Result<Arc<Shell>, ShellError> {
        let failed = |what: &str, e: &dyn std::fmt::Display| {
            warn!(error = %e, "Conversation shell: {what} failed");
            ShellError::Failed(format!("Could not start the shell ({what})"))
        };
        let pair = native_pty_system()
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| failed("opening a terminal", &e))?;
        // The owner's login shell (from the account, or `$SHELL`).
        let mut command = CommandBuilder::new_default_prog();
        command.env_clear();
        for key in KEPT_VARIABLES {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command.env("TERM", "xterm-256color");
        command.env("COLORTERM", "truecolor");
        command.cwd(folder);
        let mut child = pair
            .slave
            .spawn_command(command)
            .map_err(|e| failed("starting the program", &e))?;
        drop(pair.slave);
        let killer = child.clone_killer();
        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| failed("reading the terminal", &e))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| failed("writing to the terminal", &e))?;

        let (tx, _) = broadcast::channel(1024);
        let now = now_ms();
        let id = ShellId::random();
        let shell = Arc::new(Shell {
            info: Mutex::new(ShellInfo {
                id: id.clone(),
                session_id: session_id.to_string(),
                folder: folder.to_path_buf(),
                title: String::new(),
                cols,
                rows,
                exit_code: None,
                started_ms: now,
                last_output_ms: now,
            }),
            output: Mutex::new(Output {
                backlog: String::new(),
                limit: self.inner.config.backlog_bytes.max(4096),
                parser: vt100::Parser::new_with_callbacks(
                    nonzero(rows),
                    nonzero(cols),
                    self.inner.config.scrollback_lines,
                    TitleWatch::default(),
                ),
                tx,
            }),
            writer: Mutex::new(writer),
            master: Mutex::new(pair.master),
            killer: Mutex::new(killer),
        });

        // Its output, on a thread of its own: the terminal's reads block.
        let changes = self.inner.changes.clone();
        let reading = Arc::downgrade(&shell);
        std::thread::Builder::new()
            .name("aidaemon-shell-read".into())
            .spawn(move || {
                let mut carry = Utf8Carry::default();
                let mut buf = [0u8; 16 * 1024];
                loop {
                    let n = match reader.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    let Some(shell) = reading.upgrade() else {
                        break;
                    };
                    let text = carry.push(&buf[..n]);
                    if text.is_empty() {
                        continue;
                    }
                    let retitled = {
                        let mut output = lock(&shell.output);
                        output.write(&text);
                        output.parser.callbacks_mut().title.take()
                    };
                    let mut info = lock(&shell.info);
                    info.last_output_ms = now_ms();
                    if let Some(title) = retitled
                        && title != info.title
                    {
                        info.title = title;
                        let _ = changes.send(ShellChange::Updated(info.clone()));
                    }
                }
            })
            .map_err(|e| failed("reading the terminal", &e))?;

        // Its end, on another: the wait blocks too.
        let changes = self.inner.changes.clone();
        let waiting = Arc::downgrade(&shell);
        std::thread::Builder::new()
            .name("aidaemon-shell-wait".into())
            .spawn(move || {
                let code = child.wait().map(|s| s.exit_code()).unwrap_or(1);
                if let Some(shell) = waiting.upgrade() {
                    let info = {
                        let mut info = lock(&shell.info);
                        info.exit_code = Some(code);
                        info.clone()
                    };
                    debug!(shell = %info.id, code, "Conversation shell ended");
                    let _ = changes.send(ShellChange::Updated(info));
                }
            })
            .map_err(|e| failed("watching the program", &e))?;
        Ok(shell)
    }

    /// Attach to a shell: what it wrote so far and what it writes next.
    pub fn attach(&self, id: &ShellId) -> Result<Attachment, ShellError> {
        let shell = self.shell(id).ok_or(ShellError::NotFound)?;
        let output = lock(&shell.output);
        Ok(Attachment {
            backlog: output.backlog.clone(),
            output: output.tx.subscribe(),
        })
    }

    /// Type into a shell: keys, a paste, the terminal's answers.
    pub fn write(&self, id: &ShellId, data: &str) -> Result<(), ShellError> {
        let shell = self.shell(id).ok_or(ShellError::NotFound)?;
        if !lock(&shell.info).running() {
            return Err(ShellError::Ended);
        }
        let mut writer = lock(&shell.writer);
        writer
            .write_all(data.as_bytes())
            .and_then(|()| writer.flush())
            .map_err(|e| ShellError::Failed(format!("Could not type into the shell: {e}")))
    }

    /// The terminal's size changed (in cells).
    pub fn resize(&self, id: &ShellId, cols: u16, rows: u16) -> Result<(), ShellError> {
        let shell = self.shell(id).ok_or(ShellError::NotFound)?;
        let (cols, rows) = (cols.clamp(2, 1000), rows.clamp(1, 500));
        {
            let mut info = lock(&shell.info);
            if (info.cols, info.rows) == (cols, rows) {
                return Ok(());
            }
            info.cols = cols;
            info.rows = rows;
        }
        lock(&shell.output)
            .parser
            .set_size(nonzero(rows), nonzero(cols));
        lock(&shell.master)
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| ShellError::Failed(format!("Could not resize the shell: {e}")))
    }

    /// The shell's screen as text, with up to `above` lines over it.
    pub fn screen(&self, id: &ShellId, above: usize) -> Result<ScreenText, ShellError> {
        let shell = self.shell(id).ok_or(ShellError::NotFound)?;
        let mut output = lock(&shell.output);
        let screen = output.parser.screen_mut();
        let (rows, cols) = screen.size();
        let lines = |screen: &vt100::Screen| -> Vec<String> {
            screen
                .rows(0, cols.get())
                .map(|row| row.trim_end().to_string())
                .collect()
        };
        let mut shown = lines(screen);
        while shown.last().is_some_and(String::is_empty) {
            shown.pop();
        }
        let cursor = screen.cursor_position();
        let full_screen = screen.alternate_screen();
        // Lines above: scroll the view back a screen at a time (the view
        // is the parser's own; it is put back before the lock goes).
        let mut older: Vec<String> = Vec::new();
        if above > 0 && !full_screen {
            let rows = usize::from(rows.get());
            let mut back = 0;
            while older.len() < above {
                let want = back + rows;
                screen.set_scrollback(want);
                let got = screen.scrollback();
                if got == back {
                    break;
                }
                // The view's first `got - back` rows are new to us.
                let page = lines(screen);
                let fresh = (got - back).min(page.len());
                let mut chunk: Vec<String> = page[..fresh].to_vec();
                chunk.append(&mut older);
                older = chunk;
                back = got;
            }
            screen.set_scrollback(0);
            if older.len() > above {
                older.drain(..older.len() - above);
            }
        }
        Ok(ScreenText {
            above: older,
            screen: shown,
            cursor,
            full_screen,
        })
    }

    /// End a shell and forget it.
    pub fn close(&self, id: &ShellId) -> Result<(), ShellError> {
        let shell = lock(&self.inner.shells)
            .remove(id)
            .ok_or(ShellError::NotFound)?;
        let session_id = {
            let info = lock(&shell.info);
            info.session_id.clone()
        };
        let _ = lock(&shell.killer).kill();
        info!(shell = %id, "Conversation shell closed");
        let _ = self.inner.changes.send(ShellChange::Closed {
            id: id.clone(),
            session_id,
        });
        Ok(())
    }

    /// A conversation went (archived, deleted): its shell goes with it.
    pub fn close_session(&self, session_id: &str) {
        if let Some(info) = self.for_session(session_id) {
            let _ = self.close(&info.id);
        }
    }

    fn forget(&self, id: &ShellId) {
        let _ = self.close(id);
    }

    /// End every shell (the daemon stops).
    pub fn close_all(&self) {
        let ids: Vec<ShellId> = lock(&self.inner.shells).keys().cloned().collect();
        for id in ids {
            let _ = self.close(&id);
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        for shell in lock(&self.shells).values() {
            let _ = lock(&shell.killer).kill();
        }
    }
}

/// `shared_terminal`, the agent's side of a conversation's shell.
#[cfg(feature = "tool")]
pub mod tool;

#[cfg(test)]
mod tests;
