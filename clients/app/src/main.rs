//! The app on this machine: a window on the desktop backend, or with
//! `--tui` the same app in this terminal.

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

#[cfg(target_os = "linux")]
use nsh_platform_linux as platform;
#[cfg(target_os = "macos")]
use nsh_platform_macos as platform;
#[cfg(target_os = "windows")]
use nsh_platform_windows as platform;

mod terminal;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn main() -> impl std::process::Termination {
    if std::env::args().skip(1).any(|arg| arg == "--tui") {
        std::process::exit(match terminal::run() {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("aidaemon: {error}");
                1
            }
        });
    }
    env_logger::init();
    platform::run_app(
        platform::AppSource::Rust(platform::Root::of::<nsh_app_aidaemon::App>()),
        platform::AppOptions {
            title: "AIdaemon".to_owned(),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            app_id: "ai.aidaemon.app".to_owned(),
            size: (1180, 780),
            ..Default::default()
        },
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
compile_error!(
    "Run this binary on macOS, Windows or Linux; use the TSX entry with nsh run for iOS and Android."
);
