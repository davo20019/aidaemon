//! One model for every helper run, whichever program does the work: a
//! sub-agent (`spawn_agent`) or a CLI coding agent (`cli_agent`).
//!
//! Both are declared the same way ([`HelperSpec`]: what the work is called,
//! what it is for, the folders made fresh for it and what it will change),
//! get their authority the same way (the parent turn's, narrowed by that
//! declaration through [`crate::authority::narrow_for_child`]), are named
//! the same way ([`HelperSpec::label`]) and end the same way: [`nothing_done`]
//! decides from typed records alone whether a run that ended did its work.
//! Live 2026-11-13: two codex runs sent to install Drupal and WordPress were
//! refused `ddev` by their own sandbox, changed nothing, and still read
//! "Done" under the name "codex coding agent", because a CLI agent's run had
//! none of this.

use crate::authority::{ChildDeclaration, InheritedAuthority, ResolvedAuthority};
use crate::run_state::{RunEnd, RunEvidence, RunFailure};

/// What one helper call declared about the work it starts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HelperSpec {
    /// A short name the owner reads for the work ("Install Drupal").
    pub title: Option<String>,
    /// What the helper is for, in the caller's words.
    pub mission: String,
    /// Folders made fresh for this work (absolute, created empty by
    /// [`crate::spawn_tool::claim_new_folders`]).
    pub new_folders: Vec<String>,
    /// Files or directories the helper will change.
    pub scope: Vec<String>,
}

impl HelperSpec {
    /// The one name for the work: its title, else its mission.
    pub fn name(&self) -> &str {
        self.title
            .as_deref()
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .unwrap_or(self.mission.trim())
    }

    /// The label the owner reads on every card and line for this run, the
    /// same for every helper ([`crate::spawn::child_status_label`]): its
    /// name in plain words, else `fallback` (what the program is).
    pub fn label(&self, fallback: &str) -> String {
        if self.name().is_empty() {
            fallback.trim().to_string()
        } else {
            crate::spawn::child_status_label(self.name())
        }
    }

    /// The declaration its authority is narrowed by: its scope and the
    /// folders made for it.
    pub fn declaration(&self, parent_session: &str) -> ChildDeclaration {
        ChildDeclaration {
            parent_session: parent_session.to_string(),
            scope: self.scope.clone(),
            new_folders: self.new_folders.clone(),
            precedence: None,
            claim: None,
        }
    }
}

/// A helper's authority: the parent turn's, narrowed by what its call
/// declared, through the same [`crate::authority::narrow_for_child`] for a
/// sub-agent and a CLI agent. A CLI agent has no resolution of its own: the
/// parent's is its starting point.
pub fn helper_authority(
    parent: &InheritedAuthority,
    declaration: ChildDeclaration,
    own: Option<ResolvedAuthority>,
) -> ResolvedAuthority {
    let own = own.unwrap_or_else(|| ResolvedAuthority {
        authority: parent.authority.clone(),
        owner_level: parent.owner_level,
        task_access: parent.task_access.clone(),
    });
    let mut parent = parent.clone();
    parent.child_declaration = Some(declaration);
    crate::authority::narrow_for_child(&parent, own)
}

/// Where a narrowed authority lets the helper write, as paths: `None` when
/// it is not confined to any (it writes where its parent may).
pub fn write_roots(resolved: &ResolvedAuthority) -> Option<Vec<String>> {
    let access = resolved.task_access.as_ref()?;
    Some(
        access
            .write_targets
            .iter()
            .map(|target| target.value.clone())
            .collect(),
    )
}

/// What a helper's typed records say about its work once it ended. Its
/// reply is never part of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkRecord {
    /// It ended with a typed success of its own (a sub-agent's end record).
    /// A CLI agent has no typed verdict: its process ending is not one.
    pub succeeded: bool,
    /// Work it started still runs: that explains a stop.
    pub still_running: bool,
    /// Its steps that needed a change, in order, each with whether a
    /// receipt confirmed it. A step whose only effect is delivering
    /// something to the owner is not a change: it is listed here with
    /// whether its delivery record says it went out.
    pub change_steps: Vec<(String, bool)>,
    /// Some step of it declared a change other than a delivery. A run that
    /// declared none was sent to look or to deliver, and never ends as
    /// "nothing was changed".
    pub declares_change: bool,
    /// A receipt shows it changed something.
    pub changed_something: bool,
    /// Its run-end delivery record shows something it declared reached the
    /// owner: a delivery is work done.
    pub delivered: bool,
}

/// A run that did none of its work, from its typed records: it did not
/// succeed, it declared a change, none of its steps was done, nothing it did
/// changed anything, nothing it declared was delivered, and no work still
/// running explains the stop. A step the owner declined does not change
/// that: it is said beside it ([`crate::spawn::run_phase`]). Such a run
/// failed: whatever its reply says, nothing it was sent to do happened. The
/// one rule for every helper. A run sent to look or to deliver declared no
/// change and is never judged by it (live 2026-11-13: two "Capture …
/// screenshot" helpers delivered their photos and ended "nothing was
/// changed").
pub fn nothing_done(record: &WorkRecord) -> Option<RunFailure> {
    if record.succeeded || record.still_running {
        return None;
    }
    let a_step_done = record.change_steps.iter().any(|(_, done)| *done);
    if !record.declares_change
        || record.change_steps.is_empty()
        || a_step_done
        || record.changed_something
        || record.delivered
    {
        return None;
    }
    Some(RunFailure::NothingDone {
        undone: record
            .change_steps
            .iter()
            .map(|(step, _)| plain_step(step))
            .collect(),
    })
}

/// A step's own words as the owner reads them: one line, without a
/// trailing period.
pub fn plain_step(step: &str) -> String {
    let line = step.split_whitespace().collect::<Vec<_>>().join(" ");
    line.trim_end_matches('.').to_string()
}

/// How a CLI agent's run ended, from what the runtime observed of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramEnd {
    /// Its exit status; `None`: it was killed or its status was lost.
    pub exit_code: Option<i32>,
    /// The runtime stopped it (a loop, a stop, an authentication failure).
    pub stopped: bool,
    /// The run was given write authority: its work needs a change.
    pub expects_change: bool,
    /// Its write roots differ from how they were when it started.
    pub changed_something: bool,
    /// What it was sent to do, as the owner reads it.
    pub step: String,
}

/// How a CLI agent's run ended: an error exit failed, a stop by the runtime
/// failed; a normal exit succeeded only if it did its work, by the same
/// [`nothing_done`] rule a sub-agent's run is judged by.
pub fn program_phase(end: &ProgramEnd) -> (RunEnd, RunEvidence) {
    if end.stopped || end.exit_code != Some(0) {
        return (
            RunEnd::Failed(RunFailure::Exited {
                code: end.exit_code,
            }),
            RunEvidence::default(),
        );
    }
    let record = WorkRecord {
        succeeded: false,
        still_running: false,
        change_steps: if end.expects_change {
            vec![(end.step.clone(), false)]
        } else {
            Vec::new()
        },
        declares_change: end.expects_change,
        changed_something: end.changed_something,
        delivered: false,
    };
    match nothing_done(&record) {
        Some(failure) => (RunEnd::Failed(failure), RunEvidence::default()),
        None => (RunEnd::Succeeded, RunEvidence::default()),
    }
}

/// What every helper's mission carries besides its task, whichever program
/// runs it: the folders made for it, and that its own duration is the
/// daemon's to state ([`crate::specialists::compose::OWN_DURATION_RULE`]).
pub fn mission_rules(new_folders: &[String]) -> String {
    format!(
        "{}\n\n{}",
        crate::spawn_tool::new_folders_note(new_folders),
        crate::specialists::compose::OWN_DURATION_RULE
    )
}

/// What a delegated CLI agent's sandbox lets it do: typed facts the parent
/// reads in the tool's description and its prompt, so it chooses between a
/// CLI agent and a sub-agent from what each can actually do (live
/// 2026-11-13: two codex runs sent to install sites with `ddev` were refused
/// `ddev` and `docker` by their own sandbox within seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSandbox {
    /// It can reach the network.
    pub network: bool,
    /// It can run programs installed outside the paths it is given (and so
    /// the services they start: containers, local dev sites).
    pub programs_outside_its_paths: bool,
}

/// The sandbox every delegated CLI run gets (`cli_agent`'s adapters
/// enforce it: its scoped profile turns the network off and reaches only a
/// minimal system set plus the run's own read and write paths).
pub const DELEGATED_CLI_SANDBOX: CliSandbox = CliSandbox {
    network: false,
    programs_outside_its_paths: false,
};

impl CliSandbox {
    /// What it cannot do, in plain words; empty when nothing is withheld.
    fn cannot(&self) -> Vec<&'static str> {
        let mut cannot = Vec::new();
        if !self.network {
            cannot.push("reach the network");
        }
        if !self.programs_outside_its_paths {
            cannot.push(
                "run programs installed outside the paths it is given (containers, local dev \
                 environments, package managers, services)",
            );
        }
        cannot
    }

    /// The sandbox in plain words, for the tool's description.
    pub fn describe(&self) -> String {
        let cannot = self.cannot();
        if cannot.is_empty() {
            return "Its sandbox lets it run anything you can.".to_string();
        }
        format!(
            "Its sandbox cannot {}. Use it for coding work in a repository it is given; \
             work that needs those goes to spawn_agent, which runs with your own access.",
            cannot.join(", or ")
        )
    }

    /// The system prompt's section on delegating to a CLI agent: when to
    /// choose it, from what its sandbox can do, and how to declare a run.
    pub fn delegation_section(&self) -> String {
        format!(
            "\n\n## CLI Agent Delegation\n\
             `cli_agent` runs an installed coding agent on this machine. {}\n\
             Give each run a `title` (what the work is, in a few plain words: the owner reads \
             it on the run's card), its `working_dir`, and `new_folders` for a folder it \
             creates. Do not send the same task to several agents or run two in the same \
             working_dir. After delegating, do not redo its work with direct tools: review its \
             result. If it fails or does nothing, do not retry it unchanged: do the work with \
             your direct tools or a sub-agent.",
            self.describe()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(steps: &[(&str, bool)], changed: bool) -> WorkRecord {
        WorkRecord {
            change_steps: steps
                .iter()
                .map(|(step, done)| (step.to_string(), *done))
                .collect(),
            declares_change: !steps.is_empty(),
            changed_something: changed,
            ..WorkRecord::default()
        }
    }

    #[test]
    fn a_run_that_changed_nothing_it_was_sent_to_change_did_nothing() {
        assert_eq!(
            nothing_done(&record(&[("Install WordPress.", false)], false)),
            Some(RunFailure::NothingDone {
                undone: vec!["Install WordPress".to_string()]
            })
        );
        assert_eq!(
            nothing_done(&record(&[("Install WordPress", false)], true)),
            None
        );
        assert_eq!(
            nothing_done(&record(&[("Install WordPress", true)], false)),
            None
        );
        // Nothing to change: a look around is its work.
        assert_eq!(nothing_done(&record(&[], false)), None);
    }

    /// Live 2026-11-13: a helper sent to capture and deliver a homepage
    /// screenshot delivered it and ended "Failed: nothing was changed". A
    /// delivery is work done, and a run that declared no change is never
    /// judged as having changed nothing.
    #[test]
    fn a_run_sent_to_look_or_deliver_did_its_work() {
        let delivered = WorkRecord {
            delivered: true,
            ..record(
                &[("Start the WordPress project only if it is stopped", false)],
                false,
            )
        };
        assert_eq!(nothing_done(&delivered), None);
        let looked = WorkRecord {
            declares_change: false,
            ..record(
                &[("Deliver your WordPress homepage screenshot", false)],
                false,
            )
        };
        assert_eq!(nothing_done(&looked), None);
    }

    /// Live 2026-11-13: codex exited 0 after its own sandbox refused `ddev`,
    /// with nothing created in its folder: the run failed, typed.
    #[test]
    fn a_cli_run_refused_by_its_sandbox_is_failed_with_a_typed_reason() {
        let refused = ProgramEnd {
            exit_code: Some(0),
            stopped: false,
            expects_change: true,
            changed_something: false,
            step: "Install WordPress in claude-wordpress-7".to_string(),
        };
        let (end, _) = program_phase(&refused);
        assert!(matches!(
            end,
            RunEnd::Failed(RunFailure::NothingDone { ref undone }) if undone == &vec![
                "Install WordPress in claude-wordpress-7".to_string()
            ]
        ));
        let did_it = ProgramEnd {
            changed_something: true,
            ..refused.clone()
        };
        assert_eq!(program_phase(&did_it).0, RunEnd::Succeeded);
        let looked = ProgramEnd {
            expects_change: false,
            ..refused.clone()
        };
        assert_eq!(program_phase(&looked).0, RunEnd::Succeeded);
        let errored = ProgramEnd {
            exit_code: Some(2),
            changed_something: true,
            ..refused
        };
        let (end, _) = program_phase(&errored);
        assert_eq!(end, RunEnd::Failed(RunFailure::Exited { code: Some(2) }));
    }

    #[test]
    fn a_helper_is_named_from_its_task_in_plain_words() {
        let spec = HelperSpec {
            title: Some("Install Drupal in /Users/someone/projects/claude-drupal-7".to_string()),
            mission: "You are an execution agent.".to_string(),
            ..HelperSpec::default()
        };
        assert_eq!(
            spec.label("codex coding agent"),
            "Install Drupal in claude-drupal-7"
        );
        let untitled = HelperSpec {
            mission: "Install WordPress".to_string(),
            ..HelperSpec::default()
        };
        assert_eq!(untitled.label("codex coding agent"), "Install WordPress");
    }

    /// A CLI agent with new folders may write in them, and only where its
    /// parent could: the same narrowing a sub-agent gets.
    #[test]
    fn a_cli_helpers_new_folders_are_where_it_writes() {
        use aidaemon_tools::semantics::{
            ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind,
        };
        let parent_access = ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: Vec::new(),
            write_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/Users/someone/projects")
                    .unwrap(),
            ],
            adapter_read_targets: Vec::new(),
        };
        let parent = InheritedAuthority {
            authority: aidaemon_core::Authority::read_only(),
            owner_level: None,
            task_access: Some(parent_access),
            mode_session: None,
            untrusted_context: false,
            origin_request: None,
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            learning: None,
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
            known_resources: Default::default(),
            lineage: Vec::new(),
            card_title: None,
            card_steps: Vec::new(),
            parent_task: None,
        };
        let spec = HelperSpec {
            title: Some("Install Drupal".to_string()),
            mission: "Install Drupal".to_string(),
            new_folders: vec!["/Users/someone/projects/claude-drupal-7".to_string()],
            scope: Vec::new(),
        };
        let narrowed = helper_authority(&parent, spec.declaration("chat"), None);
        assert_eq!(
            write_roots(&narrowed),
            Some(vec!["/Users/someone/projects/claude-drupal-7".to_string()])
        );
        // A folder outside what the parent could change is not granted.
        let outside = HelperSpec {
            new_folders: vec!["/tmp/elsewhere".to_string()],
            ..spec
        };
        let narrowed = helper_authority(&parent, outside.declaration("chat"), None);
        assert_eq!(write_roots(&narrowed), Some(Vec::new()));
    }
}
