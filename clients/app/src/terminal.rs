//! The app in a terminal, as a standalone program: NSH's terminal surface
//! (`AppView`) running the Rust root, on the terminal's alternate screen
//! with the mouse captured. Input goes through NSH's own crossterm
//! conversion, the one its terminal host uses.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::{event, execute, terminal};
use nsh_platform_tui::{AppOptions, AppView, RunEvent, run_app};
use ratatui_core::terminal::Terminal;
use ratatui_crossterm::CrosstermBackend;

/// Restores the terminal however the run ends.
struct Session;

impl Session {
    fn enter() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            event::EnableMouseCapture,
            event::EnableBracketedPaste
        )?;
        Ok(Self)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            event::DisableBracketedPaste,
            event::DisableMouseCapture,
            terminal::LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
    }
}

/// The terminal's own clipboard: a copy goes to the system clipboard through
/// OSC 52 (most terminals, and over SSH and tmux with `set-clipboard on`),
/// and is kept here for pastes inside the app.
#[derive(Default)]
struct Osc52 {
    text: Mutex<Option<String>>,
}

impl nsh_core::clipboard::ClipboardBackend for Osc52 {
    fn text(&self) -> Option<String> {
        self.text.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_text(&self, text: Option<String>) {
        if let Some(text) = &text {
            let mut out = io::stdout();
            let _ = write!(out, "\x1b]52;c;{}\x07", base64(text.as_bytes()));
            let _ = out.flush();
        }
        *self.text.lock().unwrap_or_else(|e| e.into_inner()) = text;
    }
}

fn base64(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ABC[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn run() -> io::Result<()> {
    let mut options = AppOptions::default();
    options.ui_sets = vec![nsh_base::registry(), nsh_app_aidaemon::registry()];
    options.clipboard = Some(Arc::new(Osc52::default()));
    // Drafts, pins and the open conversation outlive the run.
    if let Some(home) = dirs::home_dir() {
        options.storage = Some(nsh_engine::FileStorage::open(
            home.join(".aidaemon").join("app-terminal"),
        ));
    }
    let _session = Session::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut view = AppView::new(options);
    // The first layout gets the terminal's size.
    terminal.draw(|frame| view.render(frame, frame.area()))?;
    view.load(nsh_platform_tui::Root::of::<nsh_app_aidaemon::App>())
        .map_err(|error| io::Error::other(error.to_string()))?;
    run_app(&mut terminal, &mut view, |wait: Duration| {
        if event::poll(wait.max(Duration::from_millis(1)))? {
            Ok(nsh_tui_host::convert(event::read()?))
        } else {
            Ok(RunEvent::Idle)
        }
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(super::base64(b""), "");
        assert_eq!(super::base64(b"f"), "Zg==");
        assert_eq!(super::base64(b"fo"), "Zm8=");
        assert_eq!(super::base64(b"foo"), "Zm9v");
        assert_eq!(super::base64(b"hello world"), "aGVsbG8gd29ybGQ=");
    }
}
