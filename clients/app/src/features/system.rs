//! The System page: how the daemon itself is doing. Version and models,
//! what needs a look (failing jobs and checks, overloaded queues), token
//! use, model call timings, scheduled goals, background jobs and health
//! checks. Read on open, every 30 seconds while open, and on Refresh.

use aidaemon_client_protocol::{CheckStatus, GoalRunInfo, SystemStatus};
use nsh_core::prelude::*;

use crate::store::{Store, SystemRead, now_ms};
use crate::ui::{
    self, Glyph, Size, Tone, Variant, badge, colors, section_label, size, status, time, type_scale,
};

pub fn page(store: Store) -> View {
    // Fresh while open (the interval stops with the page).
    set_interval(std::time::Duration::from_secs(30), move || {
        store.refresh_system()
    });
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(header(store))
        .child(
            scroll()
                .flex_1()
                .child(view().items_center().child(dynamic(move || {
                    let read = store.system();
                    Some(untracked(move || body(store, read)))
                }))),
        )
}

fn header(store: Store) -> View {
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .border_b_1()
        .border_color(colors::LINE)
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                ui::icon_btn(Glyph::Sidebar, "Conversations", Size::Md, move || {
                    store.open_sidebar()
                })
            },
        ))
        .child(
            text("System")
                .flex_1()
                .px_2()
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(
            text(move || match store.system() {
                SystemRead::Ready { at_ms, .. } => format!("Updated {}", time::clock(at_ms)),
                _ => String::new(),
            })
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
        )
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Md, move || {
            store.refresh_system()
        }))
}

fn body(store: Store, read: SystemRead) -> View {
    let column = view()
        .w_full()
        .max_w(px(size::COLUMN))
        .px_4()
        .py_4()
        .gap_4();
    match read {
        SystemRead::NotYet | SystemRead::Loading => {
            column.child(ui::working("Reading the daemon's status…"))
        }
        SystemRead::Failed(reason) => column.child(ui::banner(reason, Tone::Danger).child(
            ui::btn("Try again", Variant::Secondary, Size::Sm, move || {
                store.refresh_system()
            }),
        )),
        SystemRead::Ready { status, .. } => column
            .child(overview(&status))
            .child(attention(&status))
            .child(runs(store, &status))
            .child(usage(&status))
            .child(schedules(&status))
            .child(jobs(&status))
            .child(checks(&status)),
    }
}

/// A titled block.
fn section(title: &'static str) -> View {
    view()
        .gap_1()
        .child(section_label(title).role(AccessibilityRole::Header))
}

/// A label and its value on one line.
fn fact(label: impl Into<SharedString>, value: impl Into<SharedString>) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_3()
        .min_h(px(ui::fit::row() - 4.0))
        .px_2()
        .child(
            text(label.into())
                .flex_1()
                .number_of_lines(1)
                .text_size(px(ui::fit::body()))
                .text_color(colors::MUTED),
        )
        .child(
            text(value.into())
                .number_of_lines(1)
                .selectable(true)
                .text_size(px(ui::fit::body())),
        )
}

fn card() -> View {
    ui::surface().p_2().gap(px(2.0))
}

fn overview(s: &SystemStatus) -> View {
    let mut c = card()
        .child(fact("Version", s.version.clone()))
        .child(fact("Running for", time::span(s.uptime_secs)));
    if let Some(model) = &s.model {
        c = c.child(fact("Model", model.clone()));
    }
    if !s.fallback_models.is_empty() {
        c = c.child(fact("Falls back to", s.fallback_models.join(", ")));
    }
    if let Some(calls) = &s.model_calls {
        c = c
            .child(fact(
                "Model calls, last 24h",
                format!(
                    "{}{}",
                    calls.total,
                    if calls.failed > 0 {
                        format!(" · {} failed", calls.failed)
                    } else {
                        String::new()
                    }
                ),
            ))
            .child(fact(
                "Response time",
                format!(
                    "{} typical · {} slow (p95)",
                    seconds(calls.p50_ms),
                    seconds(calls.p95_ms)
                ),
            ));
    }
    section("Daemon").child(c)
}

fn seconds(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else {
        format!("{:.1} s", ms as f64 / 1000.0)
    }
}

/// What needs a look, first: failing jobs and checks, overloaded queues.
fn attention(s: &SystemStatus) -> View {
    let mut problems: Vec<String> = Vec::new();
    for job in s.jobs.iter().filter(|j| j.consecutive_failures > 0) {
        problems.push(format!(
            "Background job {} failed {} time{} in a row{}",
            job.name,
            job.consecutive_failures,
            if job.consecutive_failures == 1 {
                ""
            } else {
                "s"
            },
            job.last_error
                .as_deref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default()
        ));
    }
    for check in s
        .checks
        .iter()
        .filter(|c| !c.paused && c.status == CheckStatus::Unhealthy)
    {
        problems.push(format!(
            "{} is down{}",
            check.name,
            check
                .error
                .as_deref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default()
        ));
    }
    for run in &s.runs {
        match run_verdict(run) {
            (RunVerdict::Failed, _) => problems.push(format!("\"{}\" failed", run.description)),
            (RunVerdict::Silent, _) => problems.push(format!(
                "\"{}\" finished without reporting anything",
                run.description
            )),
            _ => {}
        }
    }
    for queue in s.queues.iter().filter(|q| q.overloaded) {
        problems.push(format!(
            "The {} queue is overloaded ({} of {})",
            queue.name, queue.depth, queue.capacity
        ));
    }
    if problems.is_empty() {
        return view().child(status("Everything is running normally", Tone::Success));
    }
    let mut list = view().gap_1();
    for problem in problems {
        list = list.child(
            text(problem)
                .selectable(true)
                .text_size(px(ui::fit::small()))
                .text_color(colors::TEXT),
        );
    }
    section("Needs a look").child(
        view()
            .gap_2()
            .p_3()
            .rounded(px(ui::radius::SURFACE))
            .bg(Tone::Warning.tint())
            .role(AccessibilityRole::Alert)
            .child(list),
    )
}

fn tokens(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", n as f64 / 1_000.0),
        _ => format!("{:.2}M", n as f64 / 1_000_000.0),
    }
}

/// Tokens per day (all models), newest first, with a bar for scale.
fn usage(s: &SystemStatus) -> View {
    if s.usage.is_empty() {
        return section("Token use, last 7 days").child(
            text("No model calls recorded.")
                .px_2()
                .text_size(px(type_scale::SM))
                .text_color(colors::FAINT),
        );
    }
    let mut days: Vec<(String, u64, u64, u64)> = Vec::new();
    for row in &s.usage {
        match days.iter_mut().find(|d| d.0 == row.day) {
            Some(day) => {
                day.1 += row.input_tokens;
                day.2 += row.output_tokens;
                day.3 += row.requests;
            }
            None => days.push((
                row.day.clone(),
                row.input_tokens,
                row.output_tokens,
                row.requests,
            )),
        }
    }
    let most = days.iter().map(|d| d.1 + d.2).max().unwrap_or(1).max(1);
    let total: u64 = days.iter().map(|d| d.1 + d.2).sum();
    let mut c = card();
    for (day, input, output, requests) in days {
        let share = (input + output) as f32 / most as f32;
        c = c.child(
            view()
                .gap(px(4.0))
                .px_2()
                .py_1()
                .child(
                    view()
                        .flex_row()
                        .gap_2()
                        .child(
                            text(day)
                                .flex_1()
                                .font_family(ui::theme::MONO)
                                .text_size(px(type_scale::SM)),
                        )
                        .child(
                            text(format!(
                                "{} in · {} out · {requests} calls",
                                tokens(input),
                                tokens(output)
                            ))
                            .text_size(px(type_scale::SM))
                            .text_color(colors::MUTED),
                        ),
                )
                .child(ui::meter(share)),
        );
    }
    section("Token use, last 7 days").child(c).child(
        text(format!("{} tokens in all", tokens(total)))
            .px_2()
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunVerdict {
    Done,
    /// Finished, but left no report: green only by its exit, not its work.
    Silent,
    Failed,
    Running,
    Stopped,
}

/// A run judged by what it left behind, and the badge that says so.
fn run_verdict(run: &GoalRunInfo) -> (RunVerdict, (&'static str, Tone)) {
    match run.status.as_str() {
        "running" | "pending" => (RunVerdict::Running, ("Running", Tone::Accent)),
        "failed" | "error" | "blocked" => (RunVerdict::Failed, ("Failed", Tone::Danger)),
        "cancelled" | "canceled" | "abandoned" => (RunVerdict::Stopped, ("Stopped", Tone::Neutral)),
        _ if run.outcome.is_none() => (RunVerdict::Silent, ("No report", Tone::Warning)),
        _ => (RunVerdict::Done, ("Done", Tone::Success)),
    }
}

/// What background goals did in the last day, each judged by its report.
fn runs(store: Store, s: &SystemStatus) -> View {
    if s.runs.is_empty() {
        return view();
    }
    let now = now_ms();
    let mut c = card();
    for run in &s.runs {
        let (_, (label, tone)) = run_verdict(run);
        let when = format!(
            "{} {}",
            time::day_label(run.started_ms, now),
            time::clock(run.started_ms)
        );
        let took = run
            .finished_ms
            .map(|f| format!(" · took {}", seconds(f.saturating_sub(run.started_ms))))
            .unwrap_or_default();
        let mut entry = view()
            .gap(px(2.0))
            .px_2()
            .py_1()
            .child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(
                        text(run.description.clone())
                            .flex_1()
                            .number_of_lines(2)
                            .text_size(px(ui::fit::body())),
                    )
                    .child(badge(label, tone)),
            )
            .child(
                text(
                    run.outcome
                        .clone()
                        .unwrap_or_else(|| "It left no report of what it did.".into()),
                )
                .selectable(true)
                .number_of_lines(4)
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
            );
        let meta = text(format!("{when}{took}"))
            .flex_1()
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT);
        entry = match run.session_id.clone() {
            Some(session) => entry.child(view().flex_row().items_center().child(meta).child(
                ui::btn("Open", Variant::Ghost, Size::Sm, move || {
                    store.select(Some(session.clone()))
                }),
            )),
            None => entry.child(meta),
        };
        c = c.child(entry);
    }
    section("While you were away").child(c)
}

fn schedules(s: &SystemStatus) -> View {
    if s.schedules.is_empty() {
        return view();
    }
    let now = now_ms();
    let mut c = card();
    for sched in &s.schedules {
        let when = match (sched.paused, sched.next_run_ms) {
            (true, _) => "Paused".to_owned(),
            (false, Some(next)) if next > now => {
                format!("Next {} {}", time::day_label(next, now), time::clock(next))
            }
            (false, Some(_)) => "Due now".to_owned(),
            (false, None) => "Not scheduled".to_owned(),
        };
        c = c.child(
            view()
                .gap(px(2.0))
                .px_2()
                .py_1()
                .child(
                    text(sched.description.clone())
                        .number_of_lines(2)
                        .text_size(px(ui::fit::body())),
                )
                .child(
                    text(format!("{} · {when}", sched.schedule))
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                ),
        );
    }
    section("Scheduled goals").child(c)
}

fn jobs(s: &SystemStatus) -> View {
    if s.jobs.is_empty() && s.queues.is_empty() {
        return view();
    }
    let now = now_ms();
    let mut c = card();
    for job in &s.jobs {
        let (label, tone) = if job.running {
            ("Running".to_owned(), Tone::Accent)
        } else if job.consecutive_failures > 0 {
            (
                format!("Failing ×{}", job.consecutive_failures),
                Tone::Danger,
            )
        } else {
            (
                job.last_success_ms
                    .map(|at| format!("OK · {}", time::ago(at, now)))
                    .unwrap_or_else(|| "Not run yet".into()),
                Tone::Neutral,
            )
        };
        c = c.child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .px_2()
                .min_h(px(ui::fit::row() - 4.0))
                .child(
                    text(job.name.replace('_', " "))
                        .flex_1()
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body())),
                )
                .child(
                    text(format!("every {}", time::span(job.every_secs)))
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                )
                .child(badge(label, tone)),
        );
    }
    for queue in &s.queues {
        c = c.child(fact(
            format!("Queue: {}", queue.name),
            format!(
                "{} of {}{}",
                queue.depth,
                queue.capacity,
                if queue.dropped > 0 {
                    format!(" · {} dropped", queue.dropped)
                } else {
                    String::new()
                }
            ),
        ));
    }
    section("Background jobs").child(c)
}

fn checks(s: &SystemStatus) -> View {
    if s.checks.is_empty() {
        return view();
    }
    let now = now_ms();
    let mut c = card();
    for check in &s.checks {
        let (label, tone) = match (check.paused, check.status) {
            (true, _) => ("Paused", Tone::Neutral),
            (_, CheckStatus::Healthy) => ("Healthy", Tone::Success),
            (_, CheckStatus::Degraded) => ("Slow", Tone::Warning),
            (_, CheckStatus::Unhealthy) => ("Down", Tone::Danger),
            (_, CheckStatus::Unknown) => ("Not checked yet", Tone::Neutral),
        };
        let detail = [
            check.latency_ms.map(|ms| seconds(u64::from(ms))),
            check
                .checked_ms
                .map(|at| format!("checked {}", time::ago(at, now))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ");
        c = c.child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .px_2()
                .py_1()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(
                            text(check.name.clone())
                                .number_of_lines(1)
                                .text_size(px(ui::fit::body())),
                        )
                        .child(
                            text(format!("{} · {detail}", check.target))
                                .number_of_lines(1)
                                .text_size(px(type_scale::XS))
                                .text_color(colors::FAINT),
                        ),
                )
                .child(badge(label, tone)),
        );
    }
    section("Health checks").child(c)
}
