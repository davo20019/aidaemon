use super::{terminal_call_effects, terminal_call_effects_on};
use aidaemon_core::{CallEffects, EffectClass, EffectKey};

fn run(command: &str) -> CallEffects {
    terminal_call_effects(
        &*aidaemon_tools::testing::active_execution_backend(),
        &serde_json::json!({"action": "run", "command": command}).to_string(),
    )
}

fn egress(target: &str) -> EffectKey {
    EffectKey {
        class: EffectClass::Egress,
        target: target.to_string(),
    }
}

#[test]
fn terminal_curl_post_declares_egress_host() {
    let effects = run("curl -X POST https://h.example.invalid:8443/x -d '{}'");
    assert_eq!(effects.keys, vec![egress("h.example.invalid:8443")]);
    assert!(!effects.complete);
}

#[test]
fn terminal_curl_default_port() {
    let effects = run("curl https://h.example.invalid/x");
    assert_eq!(effects.keys, vec![egress("h.example.invalid:443")]);
    assert!(!effects.complete);
}

/// Live 2026-11-13: a copy into the work area followed by `ls` and
/// `date +%s` is read in full (`understood`): its one effect is the
/// copy's destination. A script it cannot read stays not understood.
#[test]
fn a_copy_with_observations_is_read_in_full() {
    let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
    let effects = run(&format!(
        "cp {dir}/inbox/5b9944c7_screenshot.png {dir}/site/homepage.png; \
             ls -l {dir}/site/homepage.png; date +%s"
    ));
    assert!(effects.understood, "{effects:?}");
    assert!(!effects.complete);
    assert_eq!(
        effects.keys,
        vec![
            aidaemon_tools::command_effects::path_effect(
                EffectClass::Write,
                &format!("{dir}/site/homepage.png"),
                None
            )
            .unwrap()
        ]
    );
    assert!(!run("bash run.sh").understood);
    assert!(!run(&format!("cp $(ls {dir}) {dir}/b")).understood);
}

#[test]
fn terminal_script_is_incomplete() {
    for command in ["bash run.sh", "python3 tool.py"] {
        assert!(!run(command).complete, "{command}");
    }
    let script = terminal_call_effects(
        &*aidaemon_tools::testing::active_execution_backend(),
        &serde_json::json!({"action": "run", "script": "ls\nbash run.sh"}).to_string(),
    );
    assert!(!script.complete);
}

#[test]
fn terminal_redirect_declares_write() {
    let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
    let path = format!("{dir}/f");
    let effects = run(&format!("echo x >> {path}"));
    assert!(!effects.complete);
    assert_eq!(
        effects.keys,
        vec![
            aidaemon_tools::command_effects::path_effect(EffectClass::Write, &path, None).unwrap()
        ]
    );
    assert!(effects.keys[0].target.ends_with("/f"));
}

#[test]
fn terminal_relative_paths_resolve_against_working_dir() {
    let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
    let effects = terminal_call_effects(
        &*aidaemon_tools::testing::active_execution_backend(),
        &serde_json::json!({"action": "run", "command": "echo x > out.txt", "working_dir": dir})
            .to_string(),
    );
    assert!(!effects.complete);
    assert_eq!(
        effects.keys,
        vec![
            aidaemon_tools::command_effects::path_effect(
                EffectClass::Write,
                &format!("{dir}/out.txt"),
                None
            )
            .unwrap()
        ]
    );
    // Without a working directory the target is undeterminable.
    assert!(run("echo x > out.txt").keys.is_empty());
}

#[test]
fn terminal_remote_backend_paths_are_incomplete() {
    let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
    let arguments = serde_json::json!({
        "action": "run",
        "command": format!("curl -o {dir}/page https://h.example.invalid/ && rm -rf {dir}/old"),
    })
    .to_string();
    let local = terminal_call_effects_on(&arguments, true);
    assert!(!local.complete);
    assert_eq!(local.keys.len(), 3);
    // On an SSH or Docker workspace the paths are not on this disk.
    let remote = terminal_call_effects_on(&arguments, false);
    assert!(!remote.complete);
    assert_eq!(remote.keys, vec![egress("h.example.invalid:443")]);
}

/// C2: effects and execution read the action through one parser; an
/// action execution would run but that is not spelled canonically is
/// undeclared rather than read as `check`.
#[test]
fn terminal_action_is_parsed_once() {
    use super::TerminalAction;
    for action in ["Check", " run ", "RUN", "check ", "Kill"] {
        assert_eq!(TerminalAction::parse(action), TerminalAction::Unrecognized);
        // Execution runs the command for these, as before.
        assert!(TerminalAction::parse(action).runs_command(), "{action}");
        let arguments = serde_json::json!({
            "action": action,
            "command": "curl -d @/tmp/x https://a.example.invalid/"
        })
        .to_string();
        assert_eq!(
            terminal_call_effects(
                &*aidaemon_tools::testing::active_execution_backend(),
                &arguments
            ),
            CallEffects::unknown(),
            "{action}"
        );
    }
    assert_eq!(TerminalAction::parse("check"), TerminalAction::Check);
    assert_eq!(TerminalAction::parse("run"), TerminalAction::Run);
    assert!(!TerminalAction::parse("check").runs_command());
}

#[test]
fn terminal_check_reads_and_other_actions_are_undeclared() {
    let check = terminal_call_effects(
        &*aidaemon_tools::testing::active_execution_backend(),
        r#"{"action":"check","pid":42}"#,
    );
    assert_eq!(check, CallEffects::none());
    assert_eq!(
        terminal_call_effects(
            &*aidaemon_tools::testing::active_execution_backend(),
            r#"{"action":"trust_all"}"#
        ),
        CallEffects::none()
    );
    for other in [r#"{"action":"kill","pid":42}"#, "{"] {
        assert_eq!(
            terminal_call_effects(&*aidaemon_tools::testing::active_execution_backend(), other),
            CallEffects::unknown(),
            "{other}"
        );
    }
}
