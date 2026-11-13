//! The typed state every status card renders from.
//!
//! A chat's card (the live one during a turn and the one a turn leaves
//! behind) shows three things: the plan's steps, each helper run and the
//! turn's own current activity. The runtime owns the records; the channels
//! render them. Both sides read these types, so no surface ever rebuilds a
//! status from model text, a label string or a flag.

use serde::{Deserialize, Serialize};

/// Where a helper run stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Running,
    /// Its own end record says it did the work.
    Succeeded,
    /// It ended without doing the work: an error, a typed failure.
    Failed,
    /// It ended before the work was done: it stopped short by its own
    /// typed verdict, was cancelled, or made no progress for its limit.
    Stopped,
}

impl RunPhase {
    pub fn is_terminal(self) -> bool {
        self != Self::Running
    }

    /// A helper's header on its card: a background helper's says so, one
    /// the turn waits on reads as part of the turn's work.
    pub fn helper_headline(self, background: bool) -> &'static str {
        if background {
            return self.headline();
        }
        match self {
            Self::Running => "\u{23f3} A helper is working",
            Self::Succeeded => "\u{2705} A helper finished",
            Self::Stopped => "\u{26a0}\u{fe0f} A helper didn't finish",
            Self::Failed => "\u{274c} A helper failed",
        }
    }

    /// A background helper's header: on its card and as the first line of
    /// its result in the chat.
    pub fn headline(self) -> &'static str {
        match self {
            Self::Running => "\u{23f3} Running in the background",
            Self::Succeeded => "\u{2705} Finished in the background",
            Self::Stopped => "\u{26a0}\u{fe0f} Didn't finish in the background",
            Self::Failed => "\u{274c} Failed in the background",
        }
    }
}

/// What kind of work a step is, as the owner reads it. Set from the tool's
/// typed identity and semantics, never from its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    /// Reading a file or document.
    Read,
    /// Searching files or listing what is there.
    Search,
    /// Writing or editing files.
    Write,
    /// Running a command or program.
    Run,
    /// Searching or reading the web.
    Web,
    /// Handing work to a helper.
    Delegate,
    /// Reading what was remembered.
    Recall,
    /// Saving or changing what is remembered.
    Remember,
    /// Setting a reminder or a schedule.
    Remind,
    /// Any other look at the state of things.
    Check,
    /// Any other change.
    Change,
    /// Between steps: the model decides what comes next.
    Think,
}

impl ActivityKind {
    /// The plain words the owner reads for it.
    pub fn words(self) -> &'static str {
        match self {
            Self::Read => "Reading files",
            Self::Search => "Looking through files",
            Self::Write => "Changing files",
            Self::Run => "Running a command",
            Self::Web => "Looking things up online",
            Self::Delegate => "Asking a helper",
            Self::Recall => "Checking what I remember",
            Self::Remember => "Updating what I remember",
            Self::Remind => "Setting a reminder",
            Self::Check => "Checking things",
            Self::Change => "Making a change",
            Self::Think => "Working out the next step",
        }
    }
}

/// What a helper, or the turn itself, is doing now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// A step of this kind is running.
    Doing(ActivityKind),
    /// A step of this kind handed its work on to something that keeps
    /// running: the work goes on, so it never reads as finished.
    HandedOff(ActivityKind),
    /// The latest line a running command printed (secrets redacted).
    Output(String),
}

impl Step {
    /// The plain words the owner reads for it.
    pub fn words(&self) -> String {
        match self {
            Self::Doing(kind) => kind.words().to_string(),
            Self::HandedOff(ActivityKind::Delegate) => {
                "Handed to a helper; it works in the background".to_string()
            }
            Self::HandedOff(kind) => format!("{} · goes on in the background", kind.words()),
            Self::Output(line) => format!("{}: {}", ActivityKind::Run.words(), line.trim()),
        }
    }
}

/// One helper as every card shows it: the runtime's record of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperView {
    /// The name the owner reads for it.
    pub label: String,
    pub phase: RunPhase,
    /// It works in the background, past the turn that started it.
    pub background: bool,
    /// Its latest step while it runs.
    pub step: Option<Step>,
}

impl HelperView {
    /// A helper that is running with `step`.
    pub fn running(label: &str, background: bool, step: Option<Step>) -> Self {
        Self {
            label: label.to_string(),
            phase: RunPhase::Running,
            background,
            step,
        }
    }
}

/// How a plan step's state is known.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepEvidence {
    /// Typed evidence settled it (a receipt, a helper's run record), or
    /// it is not settled yet.
    #[default]
    Settled,
    /// Completed on the executing model's word alone. A step that changes
    /// nothing (a check, a read) needs no receipt: the reply reports what
    /// it found. Only a step that declared a side effect is flagged.
    ModelSaid,
    /// It was handed to a helper that never started, so nothing worked
    /// on it.
    HelperNotStarted,
}

impl StepEvidence {
    pub fn is_settled(&self) -> bool {
        *self == Self::Settled
    }
}
