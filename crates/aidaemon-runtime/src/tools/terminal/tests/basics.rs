//! Log labels, line counts and receipt protocol basics.

use super::*;

/// A log label names the program, never a fragment or the command's
/// text (live 2026-11-21: `>/…/count3.txt; for …` was labelled
/// `count3.txt;`).
#[test]
fn a_log_label_names_the_program_and_never_the_command() {
    for (command, program) in [
        ("mkdir -p /tmp/site && cd /tmp/site", "mkdir"),
        ("/usr/bin/git status", "git"),
        ("FOO=1 BAR=two npm run dev", "npm"),
        (
            ">/Users/alice/count3.txt; for i in 1 2; do echo $i; done",
            "for",
        ),
        ("> out.txt echo hi", "echo"),
        ("2>/dev/null ls", "ls"),
        ("ls;", "ls"),
        ("$(secret) arg", "shell"),
        ("'quoted thing' x", "shell"),
        ("", "shell"),
    ] {
        assert_eq!(logged_program(command), program, "{command}");
    }
    let label = LoggedCommand("curl -H 'Authorization: Bearer synthetic' x").to_string();
    assert!(label.starts_with("curl ["), "{label}");
    assert!(!label.contains("Bearer"), "{label}");
}

/// A command's progress is how much it printed, counted on both
/// streams, a last line without its newline included.
#[test]
fn a_commands_progress_counts_its_lines() {
    assert_eq!(printed_lines(b"", b""), 0);
    assert_eq!(printed_lines(b"one\ntwo\n", b"partial"), 3);
}

#[test]
fn background_completion_semantics_prove_what_the_launch_declared() {
    // The launch receipt is `Backgrounded` and provisional; only the
    // terminal receipt of the same invocation can credit the
    // process-state mutation, so both must derive the same effect set.
    let observed_only = ToolCallSemantics::observation();
    let completion = background_process_semantics(&observed_only);
    assert!(completion.mutates_state());
    assert!(completion.observes_state());
    assert!(
        completion
            .mutation_effects
            .satisfies(ToolMutationEffects::PROCESS_STATE)
    );

    let writes = ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE);
    let completion = background_process_semantics(&writes);
    assert!(
        completion
            .mutation_effects
            .satisfies(ToolMutationEffects::LOCAL_SOURCE_WRITE)
    );
    assert!(
        completion
            .mutation_effects
            .satisfies(ToolMutationEffects::PROCESS_STATE)
    );
    assert_eq!(completion, background_process_semantics(&writes));
}

#[test]
fn terminal_receipt_protocol_is_selected_per_action() {
    assert_eq!(
        terminal_receipt_kind(r#"{"action":"run","command":"/usr/bin/false"}"#),
        crate::tools::traits::ToolReceiptKind::Process
    );
    assert_eq!(
        terminal_receipt_kind(r#"{"action":"trust_all"}"#),
        crate::tools::traits::ToolReceiptKind::Generic
    );
}
