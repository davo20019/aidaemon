//! The shared reducer fixtures: each file in `tests/fixtures/reducer/` is a
//! list of server frames and the state they must leave, as JSON.
//!
//! `UPDATE_GOLDEN=1 cargo test -p aidaemon-client-core --test reducer`
//! rewrites each file's `state` from the Rust reducer, which is the
//! reference: review the diff before committing it.

use std::path::PathBuf;

use aidaemon_client_core::State;
use aidaemon_client_protocol::ServerFrame;
use serde_json::Value;

fn fixtures() -> Vec<(PathBuf, Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reducer");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "golden"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            let value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            (path, value)
        })
        .collect()
}

/// The state `frames` leave, as JSON.
pub fn run(frames: &[Value]) -> Value {
    let mut state = State::default();
    for frame in frames {
        let frame: ServerFrame =
            serde_json::from_value(frame.clone()).unwrap_or_else(|e| panic!("frame {frame}: {e}"));
        state.apply(frame);
    }
    serde_json::to_value(&state).unwrap()
}

#[test]
fn every_reducer_fixture_leaves_its_state() {
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let all = fixtures();
    assert!(all.len() >= 8, "fixtures missing: {}", all.len());
    for (path, mut fixture) in all {
        let frames = fixture["frames"].as_array().cloned().unwrap_or_default();
        let got = run(&frames);
        if update {
            fixture["state"] = got;
            let mut text = serde_json::to_string_pretty(&fixture).unwrap();
            text.push('\n');
            std::fs::write(&path, text).unwrap();
            continue;
        }
        assert_eq!(
            got,
            fixture["state"],
            "{} leaves another state (UPDATE_GOLDEN=1 rewrites it from this reducer)",
            path.display()
        );
    }
}

#[test]
fn the_fixtures_cover_the_planned_cases() {
    let names: Vec<String> = fixtures()
        .iter()
        .map(|(path, _)| path.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    for wanted in [
        "snapshot_then_events",
        "streaming_into_final_message",
        "approval_resolved_elsewhere",
        "task_start_and_finish",
        "resume_after_gap",
        "unknown_event_kind",
    ] {
        assert!(names.iter().any(|n| n == wanted), "no fixture {wanted}");
    }
}
