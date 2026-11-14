//! A run's measured times as text: for the parent (one marked timing line
//! per child), for the owner's card and for the footer under an answer.
//! Every figure is the runtime's measurement, never model text.

use std::time::Duration;

use crate::turn::sliding_window::MEASURED_TIMING_MARK;

/// A child's run as the daemon measured it: when it was started and when it
/// finished, when its own steps (its tool calls) ran, apart from its
/// start-up and its report, and how long it waited for the owner's answer
/// on a card, from its own clock.
pub struct MeasuredRun {
    pub started: chrono::DateTime<chrono::Utc>,
    pub finished: chrono::DateTime<chrono::Utc>,
    pub steps: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    pub waited: Duration,
}

impl MeasuredRun {
    /// The run of a child whose clock is `clock`, started at `started` and
    /// finished now.
    pub fn ended_now(
        started: chrono::DateTime<chrono::Utc>,
        clock: &aidaemon_channels::owner_wait::OwnerWaitClock,
    ) -> Self {
        Self {
            started,
            finished: chrono::Utc::now(),
            steps: clock.steps(),
            waited: clock.waited(),
        }
    }

    /// When its work began: its first step, else when it was started.
    pub fn began(&self) -> chrono::DateTime<chrono::Utc> {
        self.steps.map_or(self.started, |(first, _)| first)
    }

    /// From its first step to its report, its start-up left out (live
    /// 2026-11-13: "Ran 1m 19s" from its first to its last step, beside a
    /// footer time from its start), to the whole second
    /// ([`crate::run_state::measured_span`]).
    pub fn span(&self) -> crate::run_state::MeasuredSpan {
        crate::run_state::measured_span(self.began(), self.finished)
    }

    /// How long it took ([`Self::span`]).
    pub fn took(&self) -> chrono::Duration {
        self.span().took()
    }

    /// Its first step to its report, on the owner's clock. A range under
    /// ten minutes gives seconds, so the stated times agree with the
    /// duration ("8:34:43 PM to 8:37:15 PM ET").
    pub fn range(&self) -> String {
        let span = self.span();
        owner_clock_range(span.from, span.to)
    }
}

/// `from` to `to` in the owner's zone and 12-hour clock, with seconds when
/// the range is under ten minutes.
pub fn owner_clock_range(
    from: chrono::DateTime<chrono::Utc>,
    to: chrono::DateTime<chrono::Utc>,
) -> String {
    if (to - from) >= chrono::Duration::minutes(10) {
        return format!(
            "{} to {}",
            aidaemon_work::schedule::owner_time(&from),
            aidaemon_work::schedule::owner_time(&to)
        );
    }
    let local = |at: chrono::DateTime<chrono::Utc>| at.with_timezone(&chrono::Local);
    let zone = aidaemon_work::schedule::owner_zone_label(local(to).offset().local_minus_utc());
    format!(
        "{} to {} {zone}",
        local(from).format("%-I:%M:%S %p"),
        local(to).format("%-I:%M:%S %p")
    )
}

pub fn compact(duration: chrono::Duration) -> String {
    aidaemon_tools::duration_format::compact_chrono_duration(
        duration,
        aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
    )
}

/// When a child ran and how long it took, measured by the daemon, for the
/// parent: one timing line naming the helper (how long it took, from its
/// first step to its report, and when, in the owner's zone and 12-hour
/// clock, and its wait for the owner), then that it is quoted, never
/// recomputed. The owner reads the same measurement on the card
/// ([`took_line`]). The line
/// keeps [`MEASURED_TIMING_MARK`], so a later turn still sees it when this
/// result is cut to a summary. A parent sees which of its children
/// overlapped, and the times, durations and waits it reports are these
/// measurements, never arithmetic on times the child printed (live
/// 2026-11-12: "about 16 minutes (17:44 to 18:00)", and a "5-minute wait for
/// approval" the model guessed).
pub fn run_window(name: &str, run: &MeasuredRun) -> String {
    let took = run.took();
    format!(
        "{MEASURED_TIMING_MARK}{name} took {} ({}s), {}.{}\n\
         These are the daemon's measurements, already on the owner's card: quote them as \
         given if you mention them (the owner's time zone, 12-hour clock), never work one \
         out; times in the sub-agent's own reply are its estimates.",
        compact(took),
        took.num_seconds(),
        run.range(),
        owner_wait_note(run.waited),
    )
}

/// A failed child's measured run, for the parent: how long it ran before it
/// failed, which is never a time to finish its work. Its work's time to
/// finish exists only once something finishes it: quote a later helper's
/// measured time, or the run's own measured end when it does the work itself.
pub fn failed_run_window(name: &str, run: &MeasuredRun) -> String {
    let took = run.took();
    format!(
        "{MEASURED_TIMING_MARK}{name} ran {} ({}s), {}, and failed.{}\n\
         These are the daemon's measurements: it failed after this long, so this is not how \
         long its work takes. If you start another helper for this work, its own measured time \
         is the time to finish; if you do the work yourself, the daemon measures it from this \
         failure to the end of your turn and puts that time under your reply. Never estimate \
         one.",
        compact(took),
        took.num_seconds(),
        run.range(),
        owner_wait_note(run.waited),
    )
}

/// How long a child waited for the owner's answer on its cards, from its
/// own wait clock; empty when it never waited.
pub fn owner_wait_note(waited: Duration) -> String {
    if waited.as_secs() == 0 {
        return String::new();
    }
    format!(
        " Of that, it waited {} for the owner's answer on an approval card.",
        aidaemon_tools::duration_format::compact_seconds(
            i64::try_from(waited.as_secs()).unwrap_or(i64::MAX),
            aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
        )
    )
}

/// A background helper's measured run for the chat, from its run record:
/// its one labelled time set ([`crate::run_state::RunTimes::line`]), the
/// same line its card shows. `None` while it runs.
pub fn background_timing(run: &crate::run_state::RunState) -> Option<String> {
    Some(format!("{MEASURED_TIMING_MARK}{}", run.times()?.line()))
}

/// How long after the owner's request `done` was, for a card: its seconds
/// and its words ("4m 10s after your request (5:37:57 AM to 5:42:07 AM
/// ET)"), measured to the whole second like every span the owner reads.
pub fn since_request_view(
    requested: chrono::DateTime<chrono::Utc>,
    done: chrono::DateTime<chrono::Utc>,
) -> aidaemon_core::card::SinceRequest {
    let span = crate::run_state::measured_span(requested, done);
    aidaemon_core::card::SinceRequest {
        secs: u64::try_from(span.took().num_seconds()).unwrap_or_default(),
        words: format!(
            "{} after your request ({})",
            compact(span.took()),
            owner_clock_range(span.from, span.to)
        ),
    }
}

/// The answer's time this close after the last run's end says nothing the
/// run's own "ended … after your request" did not (live 2026-11-13: one
/// run "ended 16s after your request", then "Finished 24s after your
/// request" read as a second figure for the same thing).
pub const NOTABLE_ANSWER_GAP_SECS: i64 = 30;

/// A request's whole time as the owner reads it, from their message to
/// the answer: "Answer sent 10m 12s after your request (5:30:24 PM to
/// 5:40:36 PM ET)." The one renderer of a request's total, under every
/// final answer. It is shown only when it adds to the runs' own times:
/// several runs, work the parent did itself after them
/// (`parent_worked_after`), no run end on record, or a gap of
/// [`NOTABLE_ANSWER_GAP_SECS`] or more between the last run's end and the
/// answer. Otherwise the run's "ended … after your request" already says
/// it.
pub fn answer_sent_line(
    clock: crate::run_state::RequestClock,
    parent_worked_after: bool,
    answered: chrono::DateTime<chrono::Utc>,
) -> Option<String> {
    let gap_is_notable = clock.last_run_ended.is_none_or(|ended| {
        crate::run_state::measured_span(ended, answered)
            .took()
            .num_seconds()
            >= NOTABLE_ANSWER_GAP_SECS
    });
    (clock.runs != 1 || parent_worked_after || gap_is_notable).then(|| {
        format!(
            "Answer sent {}.",
            since_request_view(clock.requested, answered).words
        )
    })
}

/// What the runtime measured of a request a turn finished after helpers it
/// waited on, worded for the owner: the work the turn finished itself after
/// a helper (`by_me`, only when it changed something), and the request's
/// total (`total`, when the turn knows when the request came in and it adds
/// something: [`answer_sent_line`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestFinished {
    pub by_me: Option<String>,
    pub total: Option<String>,
}

impl RequestFinished {
    /// The lines under the answer, each marked as a measurement so a later
    /// turn keeps it.
    pub fn footer(&self) -> Option<String> {
        let lines: Vec<String> = [
            self.by_me.as_deref().map(|line| line.replace('\n', " ")),
            self.total.clone(),
        ]
        .into_iter()
        .flatten()
        .map(|line| format!("{MEASURED_TIMING_MARK}{line}"))
        .collect();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }
}

/// The measured end of a request a turn finished after a helper it waited
/// on ended, from the records: `requested` (when the owner's message came
/// in), `after` (when the last helper ended, how many it waited on, and
/// whether the turn then changed anything itself) and `answered` (now).
pub fn request_finished(
    requested: Option<chrono::DateTime<chrono::Utc>>,
    after: &crate::engine::scope::AfterHelper,
    answered: chrono::DateTime<chrono::Utc>,
) -> RequestFinished {
    RequestFinished {
        by_me: after.did_work.then(|| {
            format!(
                "Finished by me after the helper: {}\n{}",
                after.helper.trim(),
                took_line(
                    crate::run_state::measured_span(after.since, answered),
                    Duration::ZERO
                )
            )
        }),
        total: requested.and_then(|requested| {
            answer_sent_line(
                crate::run_state::RequestClock {
                    requested,
                    runs: after.helpers,
                    last_run_ended: Some(after.since),
                },
                after.did_work,
                answered,
            )
        }),
    }
}

/// How long a helper took and when, for the owner, from what the runtime
/// measured: "Took 21s (2:21:16 PM to 2:21:37 PM ET)." and its wait for the
/// owner's answer. A run with no request on record: one with a request
/// gives its whole time set ([`crate::run_state::RunTimes::line`]).
pub fn took_line(span: crate::run_state::MeasuredSpan, waited: Duration) -> String {
    crate::run_state::RunTimes::new(span.from, span.to, None, waited).line()
}

/// What the runtime measured of a background run as it ends: now, its
/// first tool step and its wait for the owner.
pub fn measured_end(
    clock: &aidaemon_channels::owner_wait::OwnerWaitClock,
) -> crate::run_state::RunMeasure {
    crate::run_state::RunMeasure {
        ended_at: chrono::Utc::now(),
        work_began: clock.steps().map(|(first, _)| first),
        waited: clock.waited(),
    }
}
