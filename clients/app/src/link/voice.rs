//! The desktop's own voice, for reading a reply aloud: the program each
//! system ships for it, given the text on its input. nsh has no speech
//! output of its own (it declined one: apps integrate the platform's), so
//! this is the app's platform layer.
//!
//! - macOS: `say`.
//! - Windows: the system's voice (`System.Speech`) through PowerShell.
//! - Linux: speech-dispatcher's `spd-say`, else `espeak-ng` or `espeak`;
//!   none installed means no voice.
//!
//! A reading is stopped by ending its program (and, for speech-dispatcher,
//! cancelling what it still queues: the daemon speaks, not `spd-say`).

use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

/// How this desktop speaks: a program reading the text from its input, and
/// what else stopping it takes.
struct Voice {
    program: PathBuf,
    args: Vec<&'static str>,
    /// Run after ending the program, to stop what it handed on.
    cancel: Option<(PathBuf, Vec<&'static str>)>,
}

/// The voice this desktop has, if any.
fn voice() -> Option<Voice> {
    #[cfg(target_os = "macos")]
    {
        let program = PathBuf::from("/usr/bin/say");
        return program.exists().then(|| Voice {
            program,
            args: vec!["-f", "-"],
            cancel: None,
        });
    }
    #[cfg(target_os = "windows")]
    {
        // The input is UTF-8 (the console's code page would garble other
        // scripts); the voice plays on the default device.
        const SPEAK: &str = "[Console]::InputEncoding = [Text.Encoding]::UTF8; \
            Add-Type -AssemblyName System.Speech; \
            $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
            $s.SetOutputToDefaultAudioDevice(); \
            $s.Speak([Console]::In.ReadToEnd())";
        return find_program("powershell.exe").map(|program| Voice {
            program,
            args: vec!["-NoProfile", "-NonInteractive", "-Command", SPEAK],
            cancel: None,
        });
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(program) = find_program("spd-say") {
            // -e: the text from its input; -w: return once it is spoken.
            return Some(Voice {
                cancel: Some((program.clone(), vec!["-C"])),
                program,
                args: vec!["-e", "-w"],
            });
        }
        return ["espeak-ng", "espeak"].iter().find_map(|name| {
            find_program(name).map(|program| Voice {
                program,
                args: vec!["--stdin"],
                cancel: None,
            })
        });
    }
    #[allow(unreachable_code)]
    None
}

/// `name` on the `PATH`, when it is there.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn find_program(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

/// Whether this desktop has a voice to read with.
pub fn available() -> bool {
    voice().is_some()
}

/// Text being read aloud by the desktop's voice. Dropping it stops it.
pub struct Reading {
    child: Arc<Mutex<Option<Child>>>,
    cancel: Option<(PathBuf, Vec<&'static str>)>,
}

impl Reading {
    /// Read `text` aloud; `ended` runs once, on the engine thread, when it is
    /// done or stopped. `None` when there is no voice or it would not start.
    pub fn start(text: &str, ended: impl FnOnce() + 'static) -> Option<Self> {
        let voice = voice()?;
        let mut command = Command::new(&voice.program);
        command
            .args(&voice.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt as _;
            // No console window flashes up for it.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().ok()?;
        if let Some(mut input) = child.stdin.take() {
            // Closed when dropped: the program reads to its end.
            let _ = input.write_all(text.as_bytes());
        }
        let child = Arc::new(Mutex::new(Some(child)));
        let watched = child.clone();
        // Waited for off the engine thread; the end is reported back on it,
        // so `ended` (which touches signals) runs there.
        nsh_core::prelude::spawn(async move {
            let _ = nsh_core::prelude::blocking(move || {
                loop {
                    let running = watched
                        .lock()
                        .ok()
                        .and_then(|mut child| {
                            child.as_mut().map(|c| matches!(c.try_wait(), Ok(None)))
                        })
                        .unwrap_or(false);
                    if !running {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(150));
                }
            })
            .await;
            ended();
        });
        Some(Self {
            child,
            cancel: voice.cancel,
        })
    }

    /// Stop reading now.
    pub fn stop(&self) {
        let ended_now = match self.child.lock() {
            Ok(mut child) => match child.as_mut() {
                Some(child) => {
                    let running = matches!(child.try_wait(), Ok(None));
                    if running {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    running
                }
                None => false,
            },
            Err(_) => false,
        };
        if ended_now && let Some((program, args)) = &self.cancel {
            let _ = Command::new(program)
                .args(args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

impl Drop for Reading {
    fn drop(&mut self) {
        self.stop();
    }
}
