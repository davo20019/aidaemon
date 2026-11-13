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
    /// A command is running, and this is the latest line it printed
    /// (secrets redacted): shown as its output, never as a label.
    Output(String),
    /// Its work ended and its result is being written up for the chat.
    Reporting,
}

impl ActivityKind {
    /// Whether its words say what is being done. `Check` and `Change` are
    /// the kinds of a tool without one of its own: "Checking things" says
    /// nothing, so a card shows no step rather than that filler.
    pub fn says_what(self) -> bool {
        !matches!(self, Self::Check | Self::Change)
    }
}

impl Step {
    /// Whether a card shows this step: a typed step that says what is being
    /// done. A generic kind of work ([`ActivityKind::says_what`]) is shown
    /// as nothing.
    pub fn is_shown(&self) -> bool {
        match self {
            Self::Doing(kind) => kind.says_what(),
            // Work that goes on elsewhere says so, whatever its kind.
            Self::HandedOff(_) | Self::Output(_) | Self::Reporting => true,
        }
    }

    /// The plain words the owner reads for it.
    pub fn words(&self) -> String {
        match self {
            Self::Doing(kind) => kind.words().to_string(),
            Self::HandedOff(ActivityKind::Delegate) => {
                "Handed to a helper; it works in the background".to_string()
            }
            Self::HandedOff(kind) => format!("{} · goes on in the background", kind.words()),
            Self::Output(line) => match line.trim() {
                "" => ActivityKind::Run.words().to_string(),
                line => format!("{} · last output: {line}", ActivityKind::Run.words()),
            },
            Self::Reporting => "Writing up the result".to_string(),
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
    /// Once it ended, how long it took and when, as the runtime measured it
    /// and already worded for the owner ("Took 21s (2:21:16 PM to 2:21:37 PM
    /// ET)."). Never worked out from anything a model wrote.
    pub took: Option<String>,
    /// Once it ended, how long after the owner's request that started it,
    /// as the runtime measured it: what "how long did it take" means to the
    /// owner, request to finish.
    pub since_request: Option<SinceRequest>,
}

/// How long after the owner's request a helper ended, measured by the
/// runtime: the seconds (so a card with several helpers knows which ended
/// last) and the same span already worded for the owner ("4m 10s after
/// your request (5:37:57 AM to 5:42:07 AM ET)").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinceRequest {
    pub secs: u64,
    pub words: String,
}

impl HelperView {
    /// A helper that is running with `step`.
    pub fn running(label: &str, background: bool, step: Option<Step>) -> Self {
        Self {
            label: label.to_string(),
            phase: RunPhase::Running,
            background,
            step,
            took: None,
            since_request: None,
        }
    }
}

/// A plan as the card shows it: its own rendering, how many steps it has
/// and how many of them are done. The step count, not the text, decides how
/// the card ends; the counts, with the card's run records, decide its
/// header ([`PlanCard::header_with_runs`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanCard {
    pub text: String,
    pub steps: usize,
    /// Its steps done.
    pub done: usize,
    /// Its records say how it ended: no step is left for its turn to do
    /// (each is done, set aside, failed, or handed to a run). Only a settled
    /// plan's header is a verdict the card's runs could contradict; one
    /// still being worked through keeps its own ("Working · step 2 of 4").
    pub settled: bool,
}

impl PlanCard {
    /// The header the card shows above this plan's steps, from the plan's
    /// own state and the phases of the runs on the same card together, so
    /// the header never says something the runs' lines (or the card's total)
    /// contradict. While any run on the card still runs, the work goes on:
    /// "continuing in the background", whatever the plan's own steps read
    /// (live 2026-11-13: "Stopped short · 2 of 4 done" above two helpers that
    /// were still working, then finished). Once all ended, a plan whose
    /// steps all read done on a card where a run failed or stopped says how
    /// that run ended. Otherwise the plan's own header. `None`: the plan's
    /// own header stands (a one-step plan has none).
    pub fn header_with_runs(&self, runs: &[RunPhase]) -> Option<String> {
        if self.steps < 2 || runs.is_empty() || !self.settled {
            return None;
        }
        if runs.iter().any(|phase| !phase.is_terminal()) {
            return Some(format!(
                "\u{23f3} Continuing in the background \u{b7} {} of {} done",
                self.done, self.steps
            ));
        }
        let worst = runs
            .iter()
            .copied()
            .filter(|phase| matches!(phase, RunPhase::Failed | RunPhase::Stopped))
            .max_by_key(|phase| matches!(phase, RunPhase::Failed))?;
        (self.done == self.steps).then(|| {
            format!(
                "{} \u{b7} {} of {} done",
                worst.helper_headline(false),
                self.done,
                self.steps
            )
        })
    }
}

/// How a turn's card ends, decided once from its typed state for every
/// channel path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardEnd {
    /// The card is the record of the work: it stays, settled to its final
    /// header (each helper's end, the plan's closing state).
    Keep,
    /// The card was only progress for one step: it goes, and the reply
    /// says what was done.
    Collapse,
}

impl CardEnd {
    /// A card with a helper on it, or a plan of more than one step, is
    /// kept; only a single-step card with no helper collapses into the
    /// reply.
    pub fn of(plan: Option<&PlanCard>, helpers: usize) -> Self {
        if helpers > 0 || plan.is_some_and(|plan| plan.steps > 1) {
            Self::Keep
        } else {
            Self::Collapse
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
