//! The app in a terminal, as a standalone program: NSH's terminal surface
//! (`AppView`) running the Rust root, on the terminal's alternate screen
//! with the mouse captured. Input goes through NSH's own crossterm
//! conversion, the one its terminal host uses.

use std::io;
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

pub fn run() -> io::Result<()> {
    let mut options = AppOptions::default();
    options.ui_sets = vec![nsh_base::registry(), nsh_app_aidaemon::registry()];
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
