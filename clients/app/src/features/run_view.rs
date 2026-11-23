//! One run's view, opened from its row over the composer: what it was
//! asked, its steps as its coding agent reported them, its plan, the
//! permissions it asked for, what it reported at its end and what it cost,
//! read from the run's record (`get_run`) and read again as it moves.

use aidaemon_client_protocol::{
    RunDetail, RunPermissionOutcome, RunPhase, RunPlanStatus, RunStepState, RunSummary,
};
use nsh_core::prelude::*;

use crate::store::{Read, Store};
use crate::ui::{self, Glyph, Size, Tone, colors, section_label, type_scale};

pub fn run_view(store: Store, id: String) -> View {
    let summary = {
        let id = id.clone();
        computed(move || store.all_runs().into_iter().find(|run| run.id == id))
    };
    view()
        .w_full()
        .flex_1()
        .bg(colors::SIDEBAR)
        .border_l_1()
        .border_color(colors::LINE)
        .child(
            view()
                .h(px(ui::fit::bar()))
                .flex_row()
                .items_center()
                .gap_2()
                .px_3()
                .child(ui::icon_btn(Glyph::Back, "Back", Size::Md, move || {
                    store.close_run()
                }))
                .child(
                    text(move || summary.with(|run| run.as_ref().map(title).unwrap_or_default()))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD),
                )
                .child(
                    text(move || {
                        store.tick();
                        summary.with(|run| run.as_ref().map(span).unwrap_or_default())
                    })
                    .tabular_nums()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                ),
        )
        .child(
            scroll()
                .flex_1()
                .child(view().px_3().pb_4().gap_3().child(dynamic(move || {
                    let read = store.run_view();
                    Some(untracked(move || match read {
                        Read::Ready(detail) => body(detail),
                        Read::Failed(why) => note(format!("Could not read this run: {why}")),
                        _ => note("Reading what it did…".to_string()),
                    }))
                }))),
        )
}

/// The run's name: its agent, else its label.
fn title(run: &RunSummary) -> String {
    let state = match run.phase {
        RunPhase::Running => "running",
        RunPhase::Succeeded => "finished",
        RunPhase::Failed => "failed",
        RunPhase::Stopped if run.stopped_by_owner => "stopped by you",
        RunPhase::Stopped => "stopped",
        RunPhase::Unknown => "",
    };
    let name = run.agent.clone().unwrap_or_else(|| run.label.clone());
    if state.is_empty() {
        name
    } else {
        format!("{name} · {state}")
    }
}

fn span(run: &RunSummary) -> String {
    let end = run.ended_at_ms.unwrap_or_else(crate::store::now_ms);
    ui::time::span(end.saturating_sub(run.started_at_ms) / 1000)
}

fn note(words: String) -> View {
    view().px_1().pt_2().child(
        text(words)
            .text_size(px(type_scale::SM))
            .text_color(colors::FAINT),
    )
}

fn body(detail: RunDetail) -> View {
    let mut page = view().gap_3().pt_2();
    if !detail.task.is_empty() {
        page = page.child(section(
            "Asked",
            view().child(
                text(detail.task.clone())
                    .selectable(true)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED),
            ),
        ));
    }
    if !detail.plan.is_empty() {
        let mut plan = view().gap_1();
        for item in &detail.plan {
            let (glyph, tone) = match item.status {
                RunPlanStatus::Done => (Glyph::Check, Tone::Success),
                RunPlanStatus::InProgress => (Glyph::Work, Tone::Accent),
                _ => (Glyph::Dot, Tone::Neutral),
            };
            plan = plan.child(line(glyph, tone, item.text.clone(), false));
        }
        page = page.child(section("Plan", plan));
    }
    let steps = if detail.steps.is_empty() {
        note("No steps were kept for this run.".to_string())
    } else {
        let mut steps = view().gap_1();
        for step in &detail.steps {
            let (glyph, tone) = match step.state {
                RunStepState::Done => (Glyph::Check, Tone::Success),
                RunStepState::Failed => (Glyph::Warning, Tone::Danger),
                _ => (Glyph::Work, Tone::Accent),
            };
            steps = steps.child(line(glyph, tone, step.words.clone(), step.by_helper));
        }
        steps
    };
    page = page.child(section("Steps", steps));
    if !detail.permissions.is_empty() {
        let mut asked = view().gap_1();
        for permission in &detail.permissions {
            let (glyph, tone, outcome) = match permission.outcome {
                RunPermissionOutcome::Allowed => (Glyph::Check, Tone::Success, "allowed"),
                RunPermissionOutcome::Refused => (Glyph::Warning, Tone::Danger, "refused"),
                RunPermissionOutcome::Declined => (Glyph::Warning, Tone::Warning, "declined"),
                RunPermissionOutcome::Unanswered => (Glyph::Warning, Tone::Warning, "not answered"),
                RunPermissionOutcome::Unknown => (Glyph::Dot, Tone::Neutral, ""),
            };
            let mut words = format!("{} · {outcome}", permission.label);
            if let Some(why) = &permission.reason {
                words.push_str(&format!(": {why}"));
            }
            asked = asked.child(line(glyph, tone, words, false));
        }
        page = page.child(section("Permissions", asked));
    }
    if let Some(result) = &detail.result {
        page = page.child(section(
            "Reported",
            view().child(
                text(result.clone())
                    .selectable(true)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED),
            ),
        ));
    }
    let mut facts = Vec::new();
    if let Some(usage) = &detail.usage {
        let mut tokens = format!(
            "{} tokens in, {} out",
            ui::time::compact_count(usage.input_tokens),
            ui::time::compact_count(usage.output_tokens)
        );
        if let Some(cost) = usage.cost_usd {
            tokens.push_str(&format!(" · ${cost:.2}"));
        }
        facts.push(tokens);
    }
    if let Some(dir) = &detail.working_dir {
        facts.push(format!("In {dir}"));
    }
    if let Some(session) = &detail.agent_session_id {
        facts.push(format!("Its session: {session}"));
    }
    if !facts.is_empty() {
        let mut list = view().gap_1();
        for fact in facts {
            list = list.child(
                text(fact)
                    .selectable(true)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            );
        }
        page = page.child(list);
    }
    page
}

fn section(title: &'static str, content: View) -> View {
    view().gap_1().child(section_label(title)).child(content)
}

/// One step, plan item or permission: its mark and its words; a helper's
/// step sits in under the agent's own.
fn line(glyph: Glyph, tone: Tone, words: String, nested: bool) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .pl(px(if nested { 16.0 } else { 0.0 }))
        .child(ui::icon(glyph).text_color(tone.color()))
        .child(
            text(words)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(2)
                .selectable(true)
                .text_size(px(type_scale::SM)),
        )
}
