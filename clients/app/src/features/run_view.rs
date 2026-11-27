//! A coding agent's view, opened from one of its runs: the owner's thread
//! with that agent, as a chat. Each run is one exchange: what it was asked
//! (the owner's own prompt or follow-up on the right, clearly marked "You";
//! aidaemon's delegation on the left in an inset block) and what the agent
//! answered (its changed files and checks up front, its reply — previewed and
//! collapsible when verbose so long outputs never block navigation — and its
//! steps, plan, permissions and cost folded from its record (`get_run`)).
//!
//! When a thread has multiple exchanges, a fixed navigation bar under the
//! header steps directly between exchanges (`Previous`, `Next`, `Latest`)
//! without scrolling past verbose responses.
//!
//! An agent that kept its session takes a follow-up at the foot: the
//! owner's message goes to the same agent, in its own session (`in_run`),
//! continuing the thread's latest run; the view moves to the run that takes
//! it as soon as that starts.

use aidaemon_client_protocol::{
    ModelChoice, RunDetail, RunPermissionOutcome, RunPhase, RunPlanStatus, RunStepItem,
    RunStepKind, RunStepState, RunSummary,
};
use nsh_base::{button, input};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::store::{DraftFileState, Read, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, section_label, type_scale};

/// Verbosity thresholds for collapsing long agent replies so verbose CLI
/// agents (such as Antigravity CLI) never bury checks, steps or later turns.
#[derive(Clone, Copy)]
pub(crate) struct CollapseBudget {
    pub(crate) trigger_chars: usize,
    pub(crate) trigger_lines: usize,
    pub(crate) preview_chars: usize,
    pub(crate) preview_lines: usize,
}

const FOCUSED_REPLY_BUDGET: CollapseBudget = CollapseBudget {
    trigger_chars: 900,
    trigger_lines: 18,
    preview_chars: 460,
    preview_lines: 9,
};

const OTHER_REPLY_BUDGET: CollapseBudget = CollapseBudget {
    trigger_chars: 420,
    trigger_lines: 8,
    preview_chars: 260,
    preview_lines: 5,
};

pub fn run_view(store: Store, id: String) -> View {
    // The agent's runs, oldest first: this one's thread.
    let thread = {
        let id = id.clone();
        computed(move || thread_of(&store.all_runs(), &id))
    };
    let latest = computed(move || thread.with(|runs| runs.last().cloned()));
    // In a direct coding-agent conversation (Code mode pinned to an agent),
    // every message in the thread was written by the owner, never delegated
    // by aidaemon.
    let direct_agent = computed(move || {
        latest.with(|run| {
            run.as_ref().is_some_and(|run| {
                matches!(
                    store.model_of(Some(&run.session_id)).0,
                    ModelChoice::Agent { .. }
                )
            })
        })
    });
    // Who it is: its name ("Codex 2", or the owner's "Designer"), how it
    // is headed ("Designer · Codex"), and whether it waits on the owner.
    let who = computed(move || {
        let Some(first) = thread.with(|runs| runs.first().cloned()) else {
            return (String::new(), String::new(), false);
        };
        let agent = first.agent.clone().unwrap_or_else(|| first.label.clone());
        let person = store
            .participants()
            .into_iter()
            .find(|p| p.thread == first.id);
        let (name, title) = person.as_ref().map_or_else(
            || (agent.clone(), agent.clone()),
            |p| (p.name.clone(), p.title()),
        );
        let waiting = person.is_some_and(|p| p.waiting);
        (name, title, waiting)
    });
    let ready = computed(move || matches!(store.run_view(), Read::Ready(_)));
    let scroller = element_ref();
    let opened = element_ref();
    // The earlier exchanges, folded until asked for: the view opens on the
    // latest (or the run it was opened on), at the top of its exchange,
    // where its question is. Expanding earlier exchanges does not yank the
    // viewport back down to `opened`.
    let earlier_shown = signal(false);
    effect(move || {
        let Some(_) = ready.try_get() else {
            return;
        };
        let Some(_) = thread.try_get() else {
            return;
        };
        if untracked(|| earlier_shown.try_get()).unwrap_or(true) {
            return;
        }
        untracked(|| {
            let mut options = nsh_core::RevealOptions::default();
            options.animated = false;
            opened.reveal(options);
        });
    });
    let nav_id = id.clone();
    let composer = Composer {
        draft: signal(String::new()),
        sent: signal(None),
    };
    // What the owner opened or closed, kept here: a run's update draws its
    // exchange again, and what was open stays open.
    let folds: Folds = signal(Vec::new());
    view()
        .w_full()
        .flex_1()
        .bg(colors::SIDEBAR)
        .border_l_1()
        .border_color(colors::LINE)
        .child(header(store, latest, who))
        .child(exchange_nav(store, thread, nav_id))
        .child(scroll().flex_1().element_ref(&scroller).child(
            view().px_3().pt_3().pb_4().gap_4().child(dynamic(move || {
                // Drawn again once the opened run's record is read or the
                // session's direct-agent state changes.
                let _ = ready.get();
                let direct = direct_agent.get();
                let runs = thread.get();
                let id = id.clone();
                let total = runs.len();
                // Folded: the runs before the one the view is on.
                let at = runs.iter().position(|run| run.id == id).unwrap_or(0);
                let folded = if earlier_shown.get() { 0 } else { at };
                Some(untracked(move || {
                    let mut list = view().gap_4();
                    if folded > 0 {
                        let words = match folded {
                            1 => "1 earlier exchange".to_owned(),
                            n => format!("{n} earlier exchanges"),
                        };
                        list = list.child(view().flex_row().justify_center().child(ui::btn(
                            words,
                            Variant::Outline,
                            Size::Sm,
                            move || earlier_shown.set(true),
                        )));
                    } else if at > 0 {
                        list = list.child(view().flex_row().justify_center().child(ui::btn(
                            "Hide earlier exchanges",
                            Variant::Ghost,
                            Size::Sm,
                            move || earlier_shown.set(false),
                        )));
                    }
                    let mut first_visible = true;
                    for (index, run) in runs.into_iter().enumerate().skip(folded) {
                        if !first_visible {
                            list = list.child(ui::separator());
                        }
                        first_visible = false;
                        let this = run.id == id;
                        let exchange = exchange(
                            store, composer, folds, latest, run, index, total, this, direct,
                        );
                        list = list.child(if this {
                            exchange.element_ref(&opened)
                        } else {
                            exchange
                        });
                    }
                    list
                }))
            })),
        ))
        .child(follow_up(store, composer, latest, who, direct_agent))
}

/// The follow-up composer's state, shared with the exchanges above it: a
/// prompt's Edit puts its words in the draft, and Try again sends them.
#[derive(Clone, Copy)]
struct Composer {
    draft: Signal<String>,
    /// What was sent, and to which run, until a run continues it.
    sent: Signal<Option<(String, String)>>,
}

impl Composer {
    /// Send `words` as a follow-up to `latest`, the thread's latest run;
    /// the composer shows them as sending until a run continues it.
    fn send(self, store: Store, latest: &RunSummary, words: &str) -> bool {
        let words = words.trim();
        if words.is_empty() || !store.follow_up(&latest.session_id, &latest.id, words) {
            return false;
        }
        self.sent.set(Some((words.to_string(), latest.id.clone())));
        true
    }
}

/// Top bar: back button, agent name with tone-coded status dot, a desktop
/// width toggle, and a Stop action while its latest background run is working.
fn header(
    store: Store,
    latest: Memo<Option<RunSummary>>,
    who: Memo<(String, String, bool)>,
) -> View {
    let status_info = computed(move || {
        let waiting = who.with(|(_, _, waiting)| *waiting);
        latest.with(|run| {
            run.as_ref()
                .map(|run| state_with_tone(run, waiting))
                .unwrap_or_else(|| (String::new(), Tone::Neutral))
        })
    });
    let stoppable = computed(move || {
        latest.with(|run| {
            run.as_ref()
                .filter(|r| r.phase.is_running() && r.background)
                .map(|r| r.id.clone())
        })
    });
    let resizable_desktop = computed(move || !store.is_compact() && window().width.0 >= 1180.0);
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_2()
        .px_3()
        .border_b_1()
        .border_color(colors::LINE)
        .child(ui::icon_btn(Glyph::Back, "Back", Size::Md, move || {
            store.close_run()
        }))
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .child(
                    text(move || who.with(|(_, title, _)| title.clone()))
                        .number_of_lines(1)
                        .role(AccessibilityRole::Header)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD),
                )
                .child(when(
                    move || status_info.with(|(words, _)| !words.is_empty()),
                    move || {
                        view()
                            .flex_row()
                            .items_center()
                            .gap_1p5()
                            .child(
                                view()
                                    .w(px(6.0))
                                    .h(px(6.0))
                                    .rounded_full()
                                    .bg(move || status_info.with(|(_, tone)| tone.color()))
                                    .accessibility_elements_hidden(true),
                            )
                            .child(
                                text(move || status_info.with(|(words, _)| words.clone()))
                                    .number_of_lines(1)
                                    .text_size(px(type_scale::XS))
                                    .text_color(move || {
                                        status_info.with(|(_, tone)| match tone {
                                            Tone::Neutral => colors::FAINT,
                                            t => t.color(),
                                        })
                                    }),
                            )
                    },
                )),
        )
        .child(when(
            move || resizable_desktop.get(),
            move || {
                view().child(dynamic(move || {
                    let expanded = store.run_pane_expanded();
                    let (glyph, label) = if expanded {
                        (Glyph::Shrink, "Narrower pane")
                    } else {
                        (Glyph::Expand, "Wider pane")
                    };
                    Some(untracked(move || {
                        ui::icon_btn(glyph, label, Size::Sm, move || {
                            store.toggle_run_pane_expanded()
                        })
                    }))
                }))
            },
        ))
        .child(when(
            move || stoppable.with(Option::is_some),
            move || {
                ui::btn("Stop", Variant::Ghost, Size::Sm, move || {
                    if let Some(run_id) = stoppable.get_untracked() {
                        store.stop_run(&run_id);
                    }
                })
            },
        ))
}

/// Fixed stepper under the header when a thread has two or more exchanges:
/// jump directly to the previous, next or latest exchange without scrolling
/// past a verbose response.
fn exchange_nav(store: Store, thread: Memo<Vec<RunSummary>>, id: String) -> View {
    let nav = computed(move || {
        thread.with(|runs| {
            let total = runs.len();
            if total <= 1 {
                return None;
            }
            let at = runs.iter().position(|run| run.id == id).unwrap_or(0);
            let prev = at
                .checked_sub(1)
                .and_then(|i| runs.get(i))
                .map(|r| r.id.clone());
            let next = runs.get(at + 1).map(|r| r.id.clone());
            let last = (at + 1 < total).then(|| runs[total - 1].id.clone());
            Some((at, total, prev, next, last))
        })
    });
    view().child(dynamic(move || {
        let (at, total, prev, next, last) = nav.get()?;
        Some(untracked(move || {
            let has_prev = prev.is_some();
            let has_next = next.is_some();
            let mut controls = view().flex_row().items_center().gap_1();
            if let Some(last_id) = last {
                controls = controls.child(ui::btn("Latest", Variant::Ghost, Size::Sm, move || {
                    store.open_run(&last_id)
                }));
            }
            controls = controls
                .child(ui::icon_btn_disabled(
                    Glyph::Up,
                    "Previous exchange",
                    Size::Sm,
                    !has_prev,
                    move || {
                        if let Some(prev_id) = &prev {
                            store.open_run(prev_id);
                        }
                    },
                ))
                .child(ui::icon_btn_disabled(
                    Glyph::Down,
                    "Next exchange",
                    Size::Sm,
                    !has_next,
                    move || {
                        if let Some(next_id) = &next {
                            store.open_run(next_id);
                        }
                    },
                ));
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .px_3()
                .py_1()
                .border_b_1()
                .border_color(colors::LINE)
                .child(
                    text(format!("Exchange {} of {total}", at + 1))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .tabular_nums()
                        .text_size(px(ui::fit::small()))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(colors::MUTED),
                )
                .child(controls)
        }))
    }))
}

/// The agent's state and tone, under its name: what its latest run is doing.
fn state_with_tone(run: &RunSummary, waiting: bool) -> (String, Tone) {
    if waiting {
        return ("Waiting on you".into(), Tone::Warning);
    }
    match run.phase {
        RunPhase::Running => ("Working".into(), Tone::Accent),
        _ if run.can_continue => ("Ready for a follow-up".into(), Tone::Success),
        RunPhase::Succeeded => ("Finished".into(), Tone::Neutral),
        RunPhase::Failed => ("Failed".into(), Tone::Danger),
        RunPhase::Stopped if run.stopped_by_owner => ("Stopped by you".into(), Tone::Warning),
        RunPhase::Stopped => ("Stopped".into(), Tone::Warning),
        RunPhase::Unknown => (String::new(), Tone::Neutral),
    }
}

/// One run as an exchange: optional multi-exchange strip, what it was asked,
/// then what the agent answered; `this`: the run the view was opened on,
/// with its steps.
#[allow(clippy::too_many_arguments)]
fn exchange(
    store: Store,
    composer: Composer,
    folds: Folds,
    latest: Memo<Option<RunSummary>>,
    run: RunSummary,
    index: usize,
    total: usize,
    this: bool,
    direct: bool,
) -> View {
    let mut block = view().gap_2();
    if total > 1 {
        block = block.child(exchange_meta(store, &run, index, this));
    }
    block
        .child(asked(store, composer, latest, &run, this, direct))
        .child(answer(store, composer, latest, &run, this))
        .child(if this {
            steps(store, &run, folds)
        } else {
            more(store, &run)
        })
}

/// Quiet metadata strip over each exchange in a multi-exchange thread:
/// index, status glyph, duration/clock time, and Open when viewing another.
fn exchange_meta(store: Store, run: &RunSummary, index: usize, this: bool) -> View {
    let (glyph, tone) = match run.phase {
        RunPhase::Running => (Glyph::Work, Tone::Accent),
        RunPhase::Succeeded => (Glyph::Check, Tone::Success),
        RunPhase::Failed => (Glyph::Warning, Tone::Danger),
        RunPhase::Stopped => (Glyph::Warning, Tone::Warning),
        RunPhase::Unknown => (Glyph::Dot, Tone::Neutral),
    };
    let mut caption = format!("#{}", index + 1);
    if let Some(ended) = run.ended_at_ms {
        let dur = ui::time::span(ended.saturating_sub(run.started_at_ms) / 1000);
        if !dur.is_empty() {
            caption.push_str(&format!(" · {dur}"));
        }
    } else if run.phase == RunPhase::Running {
        caption.push_str(" · working");
    }
    let clock = ui::time::clock(run.started_at_ms);
    if !clock.is_empty() {
        caption.push_str(&format!(" · {clock}"));
    }
    let mut row = view()
        .flex_row()
        .items_center()
        .gap_1p5()
        .child(
            ui::icon_inline(glyph)
                .text_color(tone.color())
                .accessibility_elements_hidden(true),
        )
        .child(
            text(caption)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .tabular_nums()
                .text_size(px(type_scale::XS))
                .font_weight(if this {
                    FontWeight::SEMIBOLD
                } else {
                    FontWeight::NORMAL
                })
                .text_color(if this { colors::MUTED } else { colors::FAINT }),
        );
    if !this {
        let open_id = run.id.clone();
        row = row.child(ui::btn("Open", Variant::Ghost, Size::Sm, move || {
            store.open_run(&open_id)
        }));
    }
    row
}

/// Whether `run`'s prompt came directly from the owner (an `in_run` follow-up
/// or any turn in a direct coding-agent conversation) rather than aidaemon
/// delegating on their behalf.
fn is_owner_prompt(run: &RunSummary, direct: bool) -> bool {
    direct || run.asked.as_ref().is_some_and(|asked| asked.by_owner)
}

/// Daemon-written recovery preamble marker (`RunProgress::note`) and prior
/// conversation wrapper (`CliAgentTool::prepend_direct_conversation_context`)
/// prepended to prompts for model continuity; stripped when rendering runs so
/// owner bubbles show only what the user actually wrote.
const INTERRUPTED_MARKER: &str = "[Where your last turn stopped]";
const INTERRUPTED_END: &str = "Check the current state of those files before redoing anything, then go on from there with the message below.";
const PRIOR_CONVERSATION_MARKER: &str = "## Prior conversation";
const CURRENT_REQUEST_MARKER: &str = "## Current request";

/// Remove daemon-written context wrappers and restart recovery preambles from
/// persisted `asked` text so owner bubbles display only the user's message.
fn clean_asked_text(raw: &str) -> String {
    let mut text = raw.trim();
    loop {
        if text.starts_with(PRIOR_CONVERSATION_MARKER)
            && let Some(idx) = text.find(CURRENT_REQUEST_MARKER)
        {
            let after = text[idx + CURRENT_REQUEST_MARKER.len()..].trim();
            if !after.is_empty() {
                text = after;
                continue;
            }
        }
        if let Some(rest) = text.strip_prefix(INTERRUPTED_MARKER)
            && let Some(idx) = rest.find(INTERRUPTED_END)
        {
            let after = rest[idx + INTERRUPTED_END.len()..].trim();
            if !after.is_empty() {
                text = after;
                continue;
            }
        }
        break;
    }
    text.to_string()
}

/// What it was asked: the owner's prompt or follow-up as a right-aligned
/// bubble clearly attributed to "You"; aidaemon's delegation on the left in
/// a recessed inset block labeled "aidaemon asked".
fn asked(
    store: Store,
    composer: Composer,
    latest: Memo<Option<RunSummary>>,
    run: &RunSummary,
    this: bool,
    direct: bool,
) -> View {
    // A record from before runs kept what they were asked: the task the
    // opened run's record has, else its label.
    let task = match untracked(|| store.run_view()) {
        Read::Ready(detail) if this && !detail.task.is_empty() => Some(detail.task),
        _ => None,
    };
    let raw_words = run
        .asked
        .as_ref()
        .map(|asked| asked.text.clone())
        .unwrap_or_else(|| task.unwrap_or_else(|| run.label.clone()));
    let words = clean_asked_text(&raw_words);
    if is_owner_prompt(run, direct) {
        let open = signal(false);
        let long = words.chars().count() > 420 || words.lines().count() > 6;
        let short = words.clone();
        let actions = prompt_actions(store, composer, latest, words.clone());
        return view()
            .w_full()
            .items_end()
            .gap_1()
            .child(
                text("You")
                    .px_1()
                    .text_size(px(type_scale::XS))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(colors::FAINT),
            )
            .child(
                view()
                    .max_w(percentage(0.88))
                    .rounded(px(ui::radius::SURFACE))
                    .bg(colors::BUBBLE)
                    .px_3()
                    .py_2()
                    .gap_1()
                    .child(
                        when(move || open.get() || !long, {
                            let words = words.clone();
                            move || {
                                text(words.clone())
                                    .selectable(true)
                                    .text_size(px(ui::fit::body()))
                                    .text_color(colors::TEXT)
                            }
                        })
                        .otherwise(move || {
                            text(short.clone())
                                .number_of_lines(6)
                                .text_size(px(ui::fit::body()))
                                .text_color(colors::TEXT)
                        }),
                    )
                    .child(when(
                        move || long,
                        move || {
                            ui::quiet_disclosure(
                                move || if open.get() { "Less" } else { "All of it" },
                                open,
                            )
                        },
                    )),
            )
            .child(actions);
    }
    let open = signal(false);
    let long = words.chars().count() > 240 || words.lines().count() > 4;
    let short = words.clone();
    let actions = prompt_actions(store, composer, latest, words.clone());
    view()
        .w_full()
        .gap_1()
        .child(asked_box(words, short, open, long))
        .child(actions)
}

/// What aidaemon asked the agent, in its quiet box.
fn asked_box(words: String, short: String, open: Signal<bool>, long: bool) -> View {
    view()
        .w_full()
        .rounded(px(ui::radius::CONTROL))
        .bg(colors::INSET)
        .border_1()
        .border_color(colors::LINE)
        .px_3()
        .py_2()
        .gap_1()
        .child(
            text("aidaemon asked")
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::FAINT),
        )
        .child(
            when(move || open.get() || !long, {
                let words = words.clone();
                move || {
                    text(words.clone())
                        .selectable(true)
                        .text_size(px(type_scale::SM))
                        .text_color(colors::MUTED)
                }
            })
            .otherwise(move || {
                text(short.clone())
                    .number_of_lines(4)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED)
            }),
        )
        .child(when(
            move || long,
            move || {
                ui::quiet_disclosure(move || if open.get() { "Less" } else { "All of it" }, open)
            },
        ))
}

/// Most changed files listed directly under a reply before folding behind a
/// collapsible disclosure so large edits never flood the pane.
const INLINE_EDITED_MAX: usize = 3;

/// What the agent answered: working while it runs; once ended, its reply
/// first (previewed and collapsible when verbose), then Copy / Read aloud,
/// then its checks/warnings and changed files (folded when more than 3).
fn answer(
    store: Store,
    composer: Composer,
    latest: Memo<Option<RunSummary>>,
    run: &RunSummary,
    this: bool,
) -> View {
    let mut said = view().gap_2();
    if run.phase == RunPhase::Running {
        let step = if run.step.is_empty() {
            "Working".to_string()
        } else {
            run.step.clone()
        };
        return said.child(ui::working(step));
    }
    let reply = run
        .result
        .as_ref()
        .map(|result| result.text.clone())
        .filter(|text| !text.is_empty());
    let agent_name = run.agent.clone().unwrap_or_else(|| "Agent".to_string());
    match (&reply, run.phase) {
        (Some(reply), _) => {
            let budget = if this {
                FOCUSED_REPLY_BUDGET
            } else {
                OTHER_REPLY_BUDGET
            };
            let preview = markdown_preview(reply, budget);
            let is_verbose = preview.is_some();
            let expanded = signal(!is_verbose);
            // Agent attribution row above the reply, with a quick Collapse
            // button at the top when a verbose reply is expanded.
            said = said.child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_1p5()
                    .child(
                        ui::icon_inline(Glyph::Work)
                            .text_color(colors::ACCENT)
                            .accessibility_elements_hidden(true),
                    )
                    .child(
                        text(agent_name)
                            .flex_1()
                            .min_w(px(0.0))
                            .number_of_lines(1)
                            .text_size(px(ui::fit::small()))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(colors::MUTED),
                    )
                    .child(when(
                        move || is_verbose && expanded.get(),
                        move || {
                            ui::btn("Collapse", Variant::Ghost, Size::Sm, move || {
                                expanded.set(false)
                            })
                        },
                    )),
            );
            match preview {
                Some((short, total_lines)) => {
                    let full = reply.clone();
                    let expand_label = if total_lines >= 6 {
                        format!("Show full response ({total_lines} lines)")
                    } else {
                        "Show full response".to_string()
                    };
                    said = said
                        .child(dynamic(move || {
                            let text = if expanded.get() {
                                full.clone()
                            } else {
                                short.clone()
                            };
                            Some(untracked(move || {
                                super::conversation::compact_markdown(
                                    move || text.clone().into(),
                                    || false,
                                )
                            }))
                        }))
                        .child(view().flex_row().items_center().child(dynamic(move || {
                            if expanded.get() {
                                Some(untracked(move || {
                                    ui::btn(
                                        "Collapse response",
                                        Variant::Ghost,
                                        Size::Sm,
                                        move || expanded.set(false),
                                    )
                                }))
                            } else {
                                let label = expand_label.clone();
                                Some(untracked(move || {
                                    ui::btn(label.clone(), Variant::Outline, Size::Sm, move || {
                                        expanded.set(true)
                                    })
                                }))
                            }
                        })));
                }
                None => {
                    let shown = reply.clone();
                    said = said.child(super::conversation::compact_markdown(
                        move || shown.clone().into(),
                        || false,
                    ));
                }
            }
            said = said.child(reply_actions(store, reply.clone()));
        }
        (None, RunPhase::Failed | RunPhase::Stopped) => {
            let why = run.reason.clone().unwrap_or_else(|| match run.phase {
                RunPhase::Failed => "It failed.".into(),
                _ => "It was stopped.".into(),
            });
            said = said.child(line(Glyph::Warning, Tone::Danger, why, false));
        }
        _ => {
            said = said.child(note("It ended without a reply.".into()));
        }
    }
    said = said.child(try_again(store, composer, latest, run));
    let coding = store.is_coding(&run.session_id);
    let facts: Vec<(Glyph, Tone, String)> =
        run.result.as_ref().map(result_facts).unwrap_or_default();
    if !facts.is_empty() {
        let mut checks = view()
            .w_full()
            .rounded(px(ui::radius::CONTROL))
            .bg(colors::SURFACE)
            .border_1()
            .border_color(colors::BORDER)
            .px_3()
            .py_2()
            .gap_1();
        for (glyph, tone, words) in facts {
            checks = checks.child(line(glyph, tone, words, false));
        }
        said = said.child(checks);
    }
    if !run.edited.is_empty() {
        said = said.child(edited_files(store, run.edited.clone(), coding));
    }
    said
}

/// Changed files under a reply: shown inline up to `INLINE_EDITED_MAX`, or
/// folded behind a single `N files changed` disclosure when there are more.
pub(crate) fn edited_files(store: Store, paths: Vec<String>, coding: bool) -> View {
    if paths.len() <= INLINE_EDITED_MAX {
        let mut list = view().gap_1();
        for path in paths {
            list = list.child(changed_file(store, path, coding));
        }
        return list;
    }
    let count = paths.len();
    let open = signal(false);
    view()
        .gap_1()
        .child(ui::quiet_disclosure(format!("{count} files changed"), open))
        .child(ui::unfold(open, move || {
            let mut list = view().gap_1().pl(px(8.0));
            for path in paths.clone() {
                list = list.child(changed_file(store, path, coding));
            }
            list
        }))
}

/// Build a safe Markdown preview for verbose replies: returns `None` when
/// `text` fits within `budget`'s trigger limits; otherwise returns
/// `Some((preview, total_lines))` cut at a clean line/word boundary with any
/// open fenced code block closed so Markdown stays well-formed.
pub(crate) fn markdown_preview(text: &str, budget: CollapseBudget) -> Option<(String, usize)> {
    let trimmed = text.trim();
    let total_lines = trimmed.lines().count();
    if trimmed.chars().count() <= budget.trigger_chars && total_lines <= budget.trigger_lines {
        return None;
    }
    let mut out = String::new();
    let mut used_chars = 0usize;
    let mut open_fence: Option<&'static str> = None;

    for raw_line in trimmed.lines().take(budget.preview_lines) {
        let line_chars = raw_line.chars().count();
        let sep_chars = usize::from(!out.is_empty());
        if used_chars + sep_chars + line_chars > budget.preview_chars {
            if !out.is_empty() && used_chars >= budget.preview_chars / 3 {
                break;
            }
            let remaining = budget.preview_chars.saturating_sub(used_chars + sep_chars);
            if remaining == 0 {
                break;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            let cut = truncate_at_word(raw_line, remaining);
            out.push_str(cut);
            update_fence(&mut open_fence, cut);
            break;
        }
        if !out.is_empty() {
            out.push('\n');
            used_chars += 1;
        }
        out.push_str(raw_line);
        used_chars += line_chars;
        update_fence(&mut open_fence, raw_line);
    }

    let mut preview = out.trim_end().to_string();
    if let Some(fence) = open_fence {
        preview.push('\n');
        preview.push_str(fence);
    } else if !preview.ends_with('…') {
        preview.push('…');
    }
    Some((preview, total_lines))
}

/// Update open fenced code block tracker (` ``` ` or `~~~`) for one line.
fn update_fence(open_fence: &mut Option<&'static str>, line: &str) {
    let lead = line.trim_start();
    let marker = if lead.starts_with("```") {
        Some("```")
    } else if lead.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    };
    if let Some(m) = marker {
        match *open_fence {
            Some(current) if current == m => *open_fence = None,
            None => *open_fence = Some(m),
            Some(_) => {}
        }
    }
}

/// Slice `s` to at most `max_chars` characters, preferring the last whitespace
/// break in the second half so words stay intact.
fn truncate_at_word(s: &str, max_chars: usize) -> &str {
    let Some((byte_at, _)) = s.char_indices().nth(max_chars) else {
        return s;
    };
    let head = &s[..byte_at];
    let cut = head
        .rfind(char::is_whitespace)
        .filter(|&pos| pos > byte_at / 2)
        .unwrap_or(byte_at);
    head[..cut].trim_end()
}

/// Under the owner's prompt: Copy, and Edit while the agent takes a
/// follow-up, which puts the words in the composer to change and send
/// again. A coding agent cannot take back what it already did, so an edit
/// is sent as a new follow-up, never as a rewrite of the thread.
fn prompt_actions(
    store: Store,
    composer: Composer,
    latest: Memo<Option<RunSummary>>,
    words: String,
) -> View {
    let copied = words.clone();
    let can_follow_up = computed(move || {
        latest.with(|run| {
            run.as_ref()
                .is_some_and(|run| run.can_continue || run.phase == RunPhase::Running)
        })
    });
    view()
        .flex_row()
        .items_center()
        .gap_4()
        .child(ui::icon_btn(Glyph::Copy, "Copy", Size::Sm, move || {
            drop(set_clipboard(copied.clone()));
            store.say("Copied your message");
        }))
        .child(when(
            move || can_follow_up.get(),
            move || {
                let words = words.clone();
                ui::icon_btn(Glyph::Pen, "Edit and send again", Size::Sm, move || {
                    composer.draft.set(words.clone());
                })
            },
        ))
}

/// Under the latest exchange when it failed or was stopped and its agent
/// takes a follow-up: Try again sends the same words again (live
/// 2026-11-27: a stopped run left the owner to retype a long prompt).
fn try_again(
    store: Store,
    composer: Composer,
    latest: Memo<Option<RunSummary>>,
    run: &RunSummary,
) -> View {
    let id = run.id.clone();
    let words = run
        .asked
        .as_ref()
        .map(|asked| clean_asked_text(&asked.text))
        .unwrap_or_default();
    let shown = computed(move || {
        composer.sent.with(Option::is_none)
            && latest.with(|latest| {
                latest.as_ref().is_some_and(|latest| {
                    latest.id == id
                        && latest.can_continue
                        && matches!(latest.phase, RunPhase::Failed | RunPhase::Stopped)
                })
            })
    });
    let has_words = !words.trim().is_empty();
    view().child(when(
        move || has_words && shown.get(),
        move || {
            let words = words.clone();
            view().flex_row().child(ui::btn(
                "Try again",
                Variant::Outline,
                Size::Sm,
                move || {
                    if let Some(latest) = latest.get_untracked() {
                        composer.send(store, &latest, &words);
                    }
                },
            ))
        },
    ))
}

/// Under the agent's reply: Copy and Read aloud, as under a model's.
fn reply_actions(store: Store, reply: String) -> View {
    let copied = reply.clone();
    view()
        .flex_row()
        .items_center()
        .gap_4()
        .child(ui::icon_btn(Glyph::Copy, "Copy", Size::Sm, move || {
            drop(set_clipboard(copied.clone()));
            store.say("Copied the reply");
        }))
        .child(super::conversation::read_aloud(store, move || {
            reply.clone()
        }))
}

/// What a run's result says beyond its reply, a line each: each check it
/// reports, as its record bears it out (a check the record doesn't show
/// run, or shows failing, is one warning, not a pass and a warning), its
/// open questions, that it needs the owner, and each other claim the
/// record doesn't bear out.
pub(crate) fn result_facts(
    result: &aidaemon_client_protocol::RunResult,
) -> Vec<(Glyph, Tone, String)> {
    use aidaemon_client_protocol::RunMismatch;
    let mut facts = Vec::new();
    let checks = result
        .report
        .as_ref()
        .map(|report| report.checks.as_slice())
        .unwrap_or_default();
    let not_run = |command: &str| {
        result
            .mismatches
            .iter()
            .any(|m| matches!(m, RunMismatch::CheckNotRun { command: c } if c == command))
    };
    let failed = |command: &str| {
        result
            .mismatches
            .iter()
            .any(|m| matches!(m, RunMismatch::CheckFailed { command: c } if c == command))
    };
    for check in checks {
        let command = &check.command;
        facts.push(if failed(command) {
            (
                Glyph::Warning,
                Tone::Danger,
                format!("{command} failed (it said passed)"),
            )
        } else if not_run(command) {
            (
                Glyph::Warning,
                Tone::Warning,
                format!(
                    "{command}: said {}, but no run of it was seen",
                    if check.passed { "passed" } else { "failed" }
                ),
            )
        } else if check.passed {
            (Glyph::Check, Tone::Success, format!("{command} passed"))
        } else {
            (Glyph::Warning, Tone::Danger, format!("{command} failed"))
        });
    }
    if let Some(report) = &result.report {
        for question in &report.open_questions {
            facts.push((Glyph::Dot, Tone::Info, format!("Open: {question}")));
        }
        if report.needs_owner {
            facts.push((Glyph::Approval, Tone::Warning, "Needs you".into()));
        }
    }
    let reported = |command: &str| checks.iter().any(|check| check.command == command);
    for mismatch in &result.mismatches {
        let words = match mismatch {
            RunMismatch::ChangeNotSeen { path } => {
                format!(
                    "Says it changed {}, but no change was seen",
                    super::files::file_name(path)
                )
            }
            RunMismatch::ChangeNotClaimed { path } => {
                format!(
                    "Changed {} without saying so",
                    super::files::file_name(path)
                )
            }
            RunMismatch::CheckNotRun { command } | RunMismatch::CheckFailed { command }
                if reported(command) =>
            {
                continue;
            }
            RunMismatch::CheckNotRun { command } => {
                format!("Says it ran {command}, but no run was seen")
            }
            RunMismatch::CheckFailed { command } => {
                format!("Says {command} passed, but it failed")
            }
            RunMismatch::Unknown => continue,
        };
        facts.push((Glyph::Warning, Tone::Warning, words));
    }
    facts
}

/// A file a run changed; in a coding conversation it opens its diff.
fn changed_file(store: Store, path: String, coding: bool) -> View {
    let name = super::files::file_name(&path).to_string();
    let tint = if coding {
        colors::ACCENT
    } else {
        colors::MUTED
    };
    let row = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(ui::icon(Glyph::File).text_color(tint))
        .child(
            text(name.clone())
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .text_size(px(type_scale::SM))
                .text_color(tint),
        );
    if coding {
        view().child(
            button()
                .accessibility_label(format!("{name}, show what changed"))
                .on_press(move || store.show_change(&path))
                .child(row),
        )
    } else {
        row
    }
}

/// The steps of a run the view wasn't opened on: one quiet line that
/// opens them.
fn more(store: Store, run: &RunSummary) -> View {
    if run.phase == RunPhase::Running {
        return view();
    }
    let id = run.id.clone();
    view().child(
        button()
            .self_start()
            .accessibility_label("Show its steps")
            .on_press(move || store.open_run(&id))
            .child(
                text("Steps")
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            ),
    )
}

/// The steps, plan, permissions and cost of the run the view was opened
/// on, folded once it has a reply (open while working or when it ended
/// without a reply); nothing when it kept none.
fn steps(store: Store, run: &RunSummary, folds: Folds) -> View {
    let has_reply = run
        .result
        .as_ref()
        .is_some_and(|result| !result.text.trim().is_empty());
    let open = fold_state(
        folds,
        (run.id.clone(), Fold::Steps),
        run.phase == RunPhase::Running || !has_reply,
    );
    let coding = store.is_coding(&run.session_id);
    let run_id = run.id.clone();
    view().child(dynamic(move || {
        let Read::Ready(detail) = store.run_view() else {
            return None;
        };
        let has = !detail.steps.is_empty()
            || !detail.plan.is_empty()
            || !detail.permissions.is_empty()
            || detail.model.is_some()
            || detail.usage.is_some();
        let run_id = run_id.clone();
        has.then(|| {
            untracked(move || {
                let count = detail.steps.len();
                let label = if count == 0 {
                    "Details".to_string()
                } else if count == 1 {
                    "1 step".to_string()
                } else {
                    format!("{count} steps")
                };
                view()
                    .gap_1()
                    .child(ui::quiet_disclosure(label, open))
                    .child(ui::unfold(open, move || {
                        body(store, detail.clone(), coding, folds, run_id.clone())
                    }))
            })
        })
    }))
}

/// The runs of run `id`'s thread, oldest first: its first run and every
/// follow-up in the same agent session (`RunSummary::thread`, the
/// daemon's own record).
fn thread_of(runs: &[RunSummary], id: &str) -> Vec<RunSummary> {
    let key = |run: &RunSummary| run.thread.clone().unwrap_or_else(|| run.id.clone());
    let Some(thread) = runs.iter().find(|run| run.id == id).map(key) else {
        return Vec::new();
    };
    let mut thread: Vec<RunSummary> = runs
        .iter()
        .filter(|run| key(run) == thread)
        .cloned()
        .collect();
    thread.sort_by_key(|run| run.started_at_ms);
    thread
}

/// A follow-up at the view's foot, continuing the thread's latest run: only
/// when its agent kept its session, it ended, and nothing goes on from it
/// yet. Once sent, the message shows as sent (under "You") until the run
/// that takes it starts, and the view moves to that run.
fn follow_up(
    store: Store,
    composer: Composer,
    latest: Memo<Option<RunSummary>>,
    who: Memo<(String, String, bool)>,
    direct_agent: Memo<bool>,
) -> View {
    let draft = composer.draft;
    let sent_tick = signal(0u64);
    let dragging = signal(false);
    let touch = ui::touch();
    let terminal = idiom() == Idiom::Terminal;
    let height = signal(ui::fit::body_lh());
    let sent = composer.sent;
    let continued = computed(move || {
        let (_, earlier) = sent.get()?;
        store
            .all_runs()
            .into_iter()
            .filter(|run| run.continues.as_deref() == Some(earlier.as_str()))
            .max_by_key(|run| run.started_at_ms)
            .map(|run| run.id)
    });
    effect(move || {
        if let Some(next) = continued.get() {
            sent.set(None);
            untracked(|| store.open_run(&next));
        }
    });
    // On desktop in a direct coding-agent conversation, the main center
    // composer right next to this pane already talks to the same agent: skip
    // the duplicate footer input there while keeping it for delegated threads
    // and compact (mobile) full-screen sheets.
    let open = computed(move || {
        if direct_agent.get() && !store.is_compact() {
            return false;
        }
        latest.with(|run| {
            run.as_ref()
                .is_some_and(|run| run.can_continue || run.phase == RunPhase::Running)
        })
    });
    let name = computed(move || who.with(|(name, _, _)| name.clone()));
    let can_send = computed(move || {
        let has_text = draft.with(|d| !d.trim().is_empty());
        let files = store.draft_files();
        let uploading = files
            .iter()
            .any(|f| matches!(f.state, DraftFileState::Uploading));
        let has_ready_file = files
            .iter()
            .any(|f| matches!(f.state, DraftFileState::Ready(_)));
        !uploading && (has_text || has_ready_file)
    });
    let send = move || {
        let text = draft.get_untracked();
        let files = store.draft_files();
        let has_ready_file = files
            .iter()
            .any(|f| matches!(f.state, DraftFileState::Ready(_)));
        if text.trim().is_empty() && !has_ready_file {
            return;
        }
        let Some((session, run)) = latest.with_untracked(|run| {
            run.as_ref()
                .map(|run| (run.session_id.clone(), run.id.clone()))
        }) else {
            return;
        };
        let preview = if !text.trim().is_empty() {
            text.trim().to_string()
        } else {
            let names: Vec<String> = files
                .iter()
                .filter(|f| matches!(f.state, DraftFileState::Ready(_)))
                .map(|f| f.name.clone())
                .collect();
            format!("Sent {}", names.join(", "))
        };
        if store.follow_up(&session, &run, &text) {
            draft.set(String::new());
            height.set(ui::fit::body_lh());
            sent_tick.update(|n| *n = n.saturating_add(1));
            sent.set(Some((preview, run)));
        }
    };
    view()
        .child(when(
            move || sent.with(Option::is_some),
            move || {
                view()
                    .px_3()
                    .py_2()
                    .gap_1()
                    .items_end()
                    .border_t_1()
                    .border_color(colors::LINE)
                    .child(
                        text("You")
                            .px_1()
                            .text_size(px(type_scale::XS))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(colors::FAINT),
                    )
                    .child(
                        view()
                            .max_w(percentage(0.88))
                            .rounded(px(ui::radius::SURFACE))
                            .bg(colors::BUBBLE)
                            .px_3()
                            .py_2()
                            .child(
                                text(move || {
                                    sent.with(|sent| {
                                        sent.as_ref()
                                            .map(|(text, _)| text.clone())
                                            .unwrap_or_default()
                                    })
                                })
                                .text_size(px(ui::fit::body())),
                            ),
                    )
                    .child(
                        text(move || format!("Sending to {}…", name.get()))
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
            },
        ))
        .child(when(
            move || open.get() && sent.with(Option::is_none),
            move || {
                view()
                    .px_3()
                    .py_2()
                    .border_t_1()
                    .border_color(colors::LINE)
                    .child(
                        view()
                            .w_full()
                            .rounded(px(ui::radius::SURFACE))
                            .bg(colors::COMPOSER)
                            .border_1()
                            .border_color(move || {
                                if dragging.get() {
                                    colors::ACCENT
                                } else {
                                    colors::EDGE
                                }
                            })
                            .px_2()
                            .pt_1()
                            .pb_1()
                            .gap_1()
                            .on_file_drag_enter(move |_: &FileDragEvent| {
                                if store.takes_files() {
                                    dragging.set(true);
                                }
                            })
                            .on_file_drag_leave(move || dragging.set(false))
                            .on_file_drop(move |drop: &FileDropEvent| {
                                dragging.set(false);
                                if store.takes_files() {
                                    super::conversation::attach_dropped(
                                        store,
                                        drop.uris.clone(),
                                        drop.mime_types.clone(),
                                    );
                                }
                            })
                            .child(when(
                                move || dragging.get(),
                                || {
                                    text("Drop to attach")
                                        .px_1()
                                        .text_size(px(type_scale::XS))
                                        .text_color(colors::ACCENT)
                                },
                            ))
                            .child(super::conversation::draft_files(store))
                            .child(
                                input()
                                    .on_paste(move |paste: &PasteEvent| {
                                        if !store.takes_files() {
                                            return;
                                        }
                                        for file in &paste.files {
                                            store.attach_file(
                                                file.name.clone(),
                                                file.mime.clone(),
                                                &file.data,
                                            );
                                        }
                                    })
                                    .value(move || draft.get())
                                    .on_change(move |text: &str| draft.set(text.to_owned()))
                                    .multiline(true)
                                    .submit_behavior(if touch {
                                        SubmitBehavior::Newline
                                    } else {
                                        SubmitBehavior::Submit
                                    })
                                    .on_submit(move |_: &str| send())
                                    .placeholder(move || {
                                        let files = store.draft_files();
                                        if files.iter().any(|f| f.marked_from.is_some()) {
                                            "Say what to do with the marked part".to_owned()
                                        } else if !files.is_empty() {
                                            "Say what to do with it, or just send".to_owned()
                                        } else {
                                            let who = name.get();
                                            latest.with(|run| match run.as_ref() {
                                                Some(r)
                                                    if r.phase == RunPhase::Running
                                                        && r.accepts_mid_run =>
                                                {
                                                    format!("Added to {who}'s work")
                                                }
                                                Some(r) if r.phase == RunPhase::Running => {
                                                    format!("Sent when {who} finishes")
                                                }
                                                _ => "Reply".to_owned(),
                                            })
                                        }
                                    })
                                    .placeholder_color(|| color(colors::FAINT))
                                    .caret_color(|| color(colors::ACCENT))
                                    .selection_color(|| color(colors::ACCENT))
                                    .on_content_size_change(
                                        move |size: &nsh_core::geometry::Size<Points>| {
                                            height.set(
                                                size.height.0.clamp(ui::fit::body_lh(), 140.0),
                                            );
                                        },
                                    )
                                    .h(move || px(height.get() + if touch { 8.0 } else { 12.0 }))
                                    .px(px(6.0))
                                    .py(px(if touch { 4.0 } else { 6.0 }))
                                    .text_size(px(ui::fit::body()))
                                    .line_height(px(ui::fit::body_lh()))
                                    .text_color(colors::TEXT)
                                    .label("Follow-up for the agent"),
                            )
                            .child(
                                view()
                                    .flex_row()
                                    .items_center()
                                    .gap_1()
                                    .child(super::conversation::attach_button(store, terminal))
                                    .child(super::conversation::dictation_for(
                                        store,
                                        terminal,
                                        move || draft.get_untracked(),
                                        move |text| draft.set(text),
                                        move || sent_tick.get(),
                                    ))
                                    .child(view().flex_1())
                                    .child(
                                        ui::icon_btn_primary(
                                            Glyph::Send,
                                            "Send to the agent",
                                            send,
                                        )
                                        .disabled(move || !can_send.get()),
                                    ),
                            ),
                    )
            },
        ))
}

fn note(words: String) -> View {
    view().px_1().child(
        text(words)
            .text_size(px(type_scale::SM))
            .text_color(colors::FAINT),
    )
}

/// The run's steps, plan, permissions, and what it ran on and cost.
fn body(store: Store, detail: RunDetail, coding: bool, folds: Folds, run_id: String) -> View {
    let mut page = view().gap_3().pt_1().pl(px(4.0));
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
    if !detail.steps.is_empty() {
        let mut steps = view().gap_1();
        for (first, row) in rows(&detail.steps) {
            steps = steps.child(match row {
                Row::Work(group) => work(
                    store,
                    group,
                    coding,
                    fold_state(folds, (run_id.clone(), Fold::Work(first)), false),
                ),
                Row::HandOffs(group) => hand_offs(store, group),
                Row::Alone(step) => step_row(store, &step, coding),
            });
        }
        page = page.child(steps);
    }
    // Only what was not simply allowed is worth a line: a run that may do
    // anything has every action allowed, which its steps already show.
    let allowed = detail
        .permissions
        .iter()
        .filter(|permission| permission.outcome == RunPermissionOutcome::Allowed)
        .count();
    if allowed < detail.permissions.len() {
        let mut asked = view().gap_1();
        for permission in detail
            .permissions
            .iter()
            .filter(|permission| permission.outcome != RunPermissionOutcome::Allowed)
        {
            let (glyph, tone, outcome) = match permission.outcome {
                RunPermissionOutcome::Refused => (Glyph::Warning, Tone::Danger, "refused"),
                RunPermissionOutcome::Declined => (Glyph::Warning, Tone::Warning, "declined"),
                RunPermissionOutcome::Unanswered => (Glyph::Warning, Tone::Warning, "not answered"),
                _ => (Glyph::Dot, Tone::Neutral, ""),
            };
            let mut words = format!("{} · {outcome}", permission.label);
            if let Some(why) = &permission.reason {
                words.push_str(&format!(": {why}"));
            }
            asked = asked.child(line(glyph, tone, words, false));
        }
        if allowed > 0 {
            let words = match allowed {
                1 => "1 other action allowed".to_string(),
                n => format!("{n} other actions allowed"),
            };
            asked = asked.child(line(Glyph::Check, Tone::Neutral, words, false));
        }
        page = page.child(section("Permissions", asked));
    }
    let mut facts = Vec::new();
    if let Some(model) = &detail.model {
        facts.push(format!("Model: {model}"));
    }
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

/// A row of the steps list.
#[derive(Debug, Clone, PartialEq)]
enum Row {
    /// Finished work in a row, folded into one line that opens it.
    Work(Vec<RunStepItem>),
    /// Work handed to other agents in a row: a row per thing done, naming
    /// each agent, which opens its run.
    HandOffs(Vec<RunStepItem>),
    /// A step that stands out: still going, or a failure nothing came after.
    Alone(RunStepItem),
}

/// The steps as rows, by one rule: every step folds into the finished
/// work around it, except what the owner needs to see on its own:
/// - a step still going, listed last, where the agent is now (one started
///   earlier, a long build beside other work, no longer splits the folds);
/// - a failure nothing finished after (one the agent went on past folds,
///   its fold counting it: "· 1 failed");
/// - the first work handed to each other agent, a link to its run: when it
///   joined. Later follow-ups to it fold, counted, their links inside (a
///   lead messaging its helpers again and again chopped the list into
///   pieces, live 2026-11-27).
///
/// Nothing else, so a new kind of step folds without a change here (live
/// 2026-11-27: a rule listing what folds left each new shape of step, a
/// retried edit, a coordination call, splitting the list).
fn rows(steps: &[RunStepItem]) -> Vec<(usize, Row)> {
    let last_done = steps
        .iter()
        .rposition(|step| step.state == RunStepState::Done);
    let mut rows: Vec<(usize, Row)> = Vec::new();
    let mut going: Vec<(usize, RunStepItem)> = Vec::new();
    let mut joined: Vec<String> = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        let step = step.clone();
        if step.state == RunStepState::Started {
            going.push((index, step));
            continue;
        }
        let unrecovered =
            step.state == RunStepState::Failed && !last_done.is_some_and(|done| done > index);
        let hand_off = step.kind == RunStepKind::HandOff
            && step.state == RunStepState::Done
            && step.run.is_some()
            && step
                .agent
                .as_ref()
                .is_some_and(|agent| !joined.contains(agent));
        if hand_off && let Some(agent) = &step.agent {
            joined.push(agent.clone());
        }
        match rows.last_mut() {
            _ if unrecovered => rows.push((index, Row::Alone(step))),
            Some((_, Row::HandOffs(group))) if hand_off => group.push(step),
            _ if hand_off => rows.push((index, Row::HandOffs(vec![step]))),
            Some((_, Row::Work(group))) => group.push(step),
            _ => rows.push((index, Row::Work(vec![step]))),
        }
    }
    rows.extend(
        going
            .into_iter()
            .map(|(index, step)| (index, Row::Alone(step))),
    );
    rows
}

/// A fold in a run's view the owner can open or close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fold {
    /// Its "N steps" disclosure.
    Steps,
    /// A fold of finished work, by its first step: steps added after it
    /// leave it the same fold.
    Work(usize),
}

/// The folds the owner set in this view, by run, each with whether it is
/// open; a fold not listed is as it starts.
type Folds = Signal<Vec<((String, Fold), bool)>>;

/// A fold's open state for this drawing: as the owner last set it, else
/// `default`; what the owner sets is kept in `folds` for the next drawing.
fn fold_state(folds: Folds, key: (String, Fold), default: bool) -> Signal<bool> {
    let set = folds.with_untracked(|folds| {
        folds
            .iter()
            .find(|(seen, _)| *seen == key)
            .map(|(_, open)| *open)
    });
    let open = signal(set.unwrap_or(default));
    let started = std::cell::Cell::new(false);
    effect(move || {
        let now = open.get();
        // Its first run is its start, not the owner's press.
        if !started.replace(true) {
            return;
        }
        let key = key.clone();
        folds.update(
            |folds| match folds.iter_mut().find(|(seen, _)| *seen == key) {
                Some((_, was)) => *was = now,
                None => folds.push((key, now)),
            },
        );
    });
    open
}

/// Hand-offs in a row: one row per thing done, "Handed work to Claude Code
/// and Codex", "Sent a follow-up to Codex", each name opening the run it
/// started. The words before the agent's name are the daemon's own
/// ("Handed work to", "Sent a follow-up to"), so rows group by them.
fn hand_offs(store: Store, steps: Vec<RunStepItem>) -> View {
    let mut leads: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for step in &steps {
        let (Some(run), Some(agent)) = (step.run.clone(), step.agent.clone()) else {
            continue;
        };
        // Machine format: the daemon writes these words and ends them with
        // the agent's name.
        let lead = step
            .words
            .strip_suffix(agent.as_str())
            .unwrap_or(&step.words)
            .trim_end()
            .to_string();
        match leads.last_mut() {
            Some((seen, runs)) if *seen == lead => {
                if !runs.iter().any(|(known, _)| *known == run) {
                    runs.push((run, agent));
                }
            }
            _ => leads.push((lead, vec![(run, agent)])),
        }
    }
    let mut rows = view().gap_1();
    for (lead, runs) in leads {
        rows = rows.child(hand_off_row(store, lead, runs));
    }
    rows
}

fn hand_off_row(store: Store, lead: String, runs: Vec<(String, String)>) -> View {
    let mut row = view()
        .flex_row()
        .flex_wrap()
        .items_center()
        .gap_1()
        .child(
            view()
                .pr_1()
                .child(ui::icon(Glyph::Check).text_color(Tone::Success.color())),
        )
        .child(text(lead).text_size(px(type_scale::SM)));
    let count = runs.len();
    for (index, (run, agent)) in runs.into_iter().enumerate() {
        if index > 0 && index + 1 == count {
            row = row.child(text("and").text_size(px(type_scale::SM)));
        }
        // "Claude Code, Codex and Gemini": a comma after each but the last two.
        let agent = if index + 2 < count {
            format!("{agent},")
        } else {
            agent
        };
        row = row.child(
            button()
                .accessibility_label(format!("{}, open its run", agent.trim_end_matches(',')))
                .on_press(move || store.open_run(&run))
                .child(
                    text(agent)
                        .text_size(px(type_scale::SM))
                        .text_color(colors::ACCENT),
                ),
        );
    }
    row
}

/// One step's row; a file it changed opens its diff in Changes.
fn step_row(store: Store, step: &RunStepItem, coding: bool) -> View {
    let (glyph, tone) = match step.state {
        RunStepState::Done => (Glyph::Check, Tone::Success),
        RunStepState::Failed => (Glyph::Warning, Tone::Danger),
        _ => (Glyph::Work, Tone::Accent),
    };
    let row = line(glyph, tone, step.words.clone(), step.by_helper);
    match (&step.path, step.kind) {
        (Some(path), RunStepKind::Edit) if coding => {
            let path = path.clone();
            view().child(
                button()
                    .accessibility_label(format!("{}, show what changed", step.words))
                    .on_press(move || store.show_change(&path))
                    .child(row),
            )
        }
        _ => row,
    }
}

/// Finished work in a row: one step is its own line; more are one line
/// ("Changed 3 files · read 6 files · ran 2 commands") that opens them,
/// the changed files first (one row per file, with how many edits it took,
/// each opening its diff), then the rest in order.
fn work(store: Store, steps: Vec<RunStepItem>, coding: bool, open: Signal<bool>) -> View {
    let files = files_changed(&steps);
    let file_row = move |(path, count): &(String, usize)| {
        let mut words = format!("Editing {}", super::files::file_name(path));
        if *count > 1 {
            words.push_str(&format!(" · {count} edits"));
        }
        let row = line(Glyph::Check, Tone::Success, words.clone(), false);
        if !coding {
            return row;
        }
        let path = path.clone();
        view().child(
            button()
                .accessibility_label(format!("{words}, show what changed"))
                .on_press(move || store.show_change(&path))
                .child(row),
        )
    };
    let other = move |step: &RunStepItem| {
        if let (Some(run), Some(agent), RunStepState::Done) =
            (step.run.clone(), step.agent.clone(), step.state)
        {
            return hand_offs(
                store,
                vec![RunStepItem {
                    run: Some(run),
                    agent: Some(agent),
                    ..step.clone()
                }],
            );
        }
        let (glyph, tone) = if step.state == RunStepState::Failed {
            (Glyph::Warning, Tone::Danger)
        } else {
            (Glyph::Check, Tone::Success)
        };
        line(glyph, tone, step.words.clone(), step.by_helper)
    };
    let rest = rest_of_work(&steps);
    match (files.as_slice(), rest.as_slice()) {
        ([file], []) => return file_row(file),
        ([], [step]) => return other(step),
        _ => {}
    }
    view()
        .gap_1()
        .child(ui::quiet_disclosure(work_summary(&steps), open))
        .child(ui::unfold(open, move || {
            let mut list = view().gap_1().pl(px(8.0));
            for file in &files {
                list = list.child(file_row(file));
            }
            for step in &rest {
                list = list.child(other(step));
            }
            list
        }))
}

/// A fold's steps other than the edits [`files_changed`] counts: each is
/// listed as its own line, so every step of a fold is shown once.
fn rest_of_work(steps: &[RunStepItem]) -> Vec<RunStepItem> {
    steps
        .iter()
        .filter(|step| !is_file_change(step))
        .cloned()
        .collect()
}

/// An edit that changed a file it names: listed under that file.
fn is_file_change(step: &RunStepItem) -> bool {
    step.kind == RunStepKind::Edit && step.state != RunStepState::Failed && step.path.is_some()
}

/// The files edits changed, each once in the order first changed, with how
/// many edits each took.
fn files_changed(steps: &[RunStepItem]) -> Vec<(String, usize)> {
    let mut files: Vec<(String, usize)> = Vec::new();
    for path in steps
        .iter()
        .filter(|step| is_file_change(step))
        .filter_map(|step| step.path.clone())
    {
        match files.iter_mut().find(|(seen, _)| *seen == path) {
            Some((_, count)) => *count += 1,
            None => files.push((path, 1)),
        }
    }
    files
}

/// What a fold of finished work did: the files it changed first, then the
/// rest by kind in their first order.
fn work_summary(steps: &[RunStepItem]) -> String {
    let plural = |n: usize, one: &str, many: &str| {
        if n == 1 {
            one.to_string()
        } else {
            format!("{n} {many}")
        }
    };
    let mut parts = Vec::new();
    let changed = files_changed(steps).len();
    if changed > 0 {
        parts.push(format!("changed {}", plural(changed, "a file", "files")));
    }
    // Work handed to an agent, by agent: "handed Codex work 2 times".
    let mut handed: Vec<(String, usize)> = Vec::new();
    for agent in steps
        .iter()
        .filter(|step| step.kind == RunStepKind::HandOff && step.state != RunStepState::Failed)
        .filter_map(|step| step.agent.clone())
    {
        match handed.iter_mut().find(|(seen, _)| *seen == agent) {
            Some((_, count)) => *count += 1,
            None => handed.push((agent, 1)),
        }
    }
    let mut counts: Vec<(RunStepKind, usize)> = Vec::new();
    for step in steps.iter().filter(|step| {
        !is_file_change(step)
            && step.state != RunStepState::Failed
            && !(step.kind == RunStepKind::HandOff && step.agent.is_some())
    }) {
        match counts.iter_mut().find(|(kind, _)| *kind == step.kind) {
            Some((_, count)) => *count += 1,
            None => counts.push((step.kind, 1)),
        }
    }
    parts.extend(counts.into_iter().map(|(kind, n)| match kind {
        RunStepKind::Read => format!("read {}", plural(n, "a file", "files")),
        RunStepKind::Search => format!("searched {}", plural(n, "once", "times")),
        RunStepKind::Run => format!("ran {}", plural(n, "a command", "commands")),
        RunStepKind::Fetch => format!("looked up {}", plural(n, "a page", "pages")),
        RunStepKind::Plan => "updated its plan".to_string(),
        RunStepKind::Coordinate => format!("coordinated {}", plural(n, "once", "times")),
        RunStepKind::Edit => format!("made {}", plural(n, "an edit", "edits")),
        RunStepKind::Delegate => format!("asked {}", plural(n, "a helper", "helpers")),
        RunStepKind::HandOff => format!("handed off work {}", plural(n, "once", "times")),
        _ => format!("used {}", plural(n, "another tool", "other tools")),
    }));
    parts.extend(
        handed
            .into_iter()
            .map(|(agent, n)| format!("handed {agent} work {}", plural(n, "once", "times"))),
    );
    let failed = steps
        .iter()
        .filter(|step| step.state == RunStepState::Failed)
        .count();
    if failed > 0 {
        parts.push(format!("{failed} failed"));
    }
    let words = parts.join(" · ");
    let mut chars = words.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => words,
    }
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

#[cfg(test)]
mod tests {
    use super::{
        FOCUSED_REPLY_BUDGET, Row, clean_asked_text, files_changed, is_owner_prompt,
        markdown_preview, rest_of_work, rows, work_summary,
    };
    use aidaemon_client_protocol::{RunStepItem, RunStepKind, RunStepState};

    fn step(kind: RunStepKind, state: RunStepState) -> RunStepItem {
        RunStepItem {
            words: String::new(),
            state,
            at_ms: None,
            by_helper: false,
            kind,
            path: None,
            run: None,
            agent: None,
        }
    }

    #[test]
    fn finished_work_is_summed_up_changes_first_then_by_kind() {
        use RunStepKind::*;
        let steps: Vec<RunStepItem> = [Run, Read, Read, Run, Other, Search, Coordinate, Coordinate]
            .into_iter()
            .map(|kind| step(kind, RunStepState::Done))
            .collect();
        assert_eq!(
            work_summary(&steps),
            "Ran 2 commands · read 2 files · used another tool · searched once · coordinated 2 times"
        );
        // Edits count the files they changed, each once, and come first.
        let mut with_edits = steps.clone();
        for path in ["/w/a.rs", "/w/b.rs", "/w/a.rs"] {
            let mut edit = step(Edit, RunStepState::Done);
            edit.path = Some(path.into());
            with_edits.push(edit);
        }
        assert!(
            work_summary(&with_edits).starts_with("Changed 2 files · ran 2 commands"),
            "{}",
            work_summary(&with_edits)
        );
    }

    /// Live 2026-11-27: a retried edit's failure and a build started
    /// earlier split 172 steps into many small folds.
    #[test]
    fn failures_gone_past_fold_and_steps_still_going_come_last() {
        use RunStepKind::*;
        let edit = |path: &str, state| {
            let mut edit = step(Edit, state);
            edit.path = Some(path.into());
            edit
        };
        let steps = vec![
            step(Read, RunStepState::Done),
            step(Run, RunStepState::Started),
            edit("/w/client.rs", RunStepState::Failed),
            edit("/w/client.rs", RunStepState::Done),
            step(Read, RunStepState::Done),
            edit("/w/mod.rs", RunStepState::Failed),
        ];
        let laid = rows(&steps);
        assert_eq!(laid.len(), 3, "{laid:?}");
        assert_eq!(laid[0].0, 0);
        let Row::Work(group) = &laid[0].1 else {
            panic!("{laid:?}")
        };
        assert_eq!(group.len(), 4);
        assert_eq!(
            work_summary(group),
            "Changed a file · read 2 files · 1 failed"
        );
        // Nothing finished after it: it stands out.
        assert!(matches!(&laid[1].1, Row::Alone(step) if step.state == RunStepState::Failed));
        // Still going: last, where the agent is now.
        assert!(matches!(&laid[2].1, Row::Alone(step) if step.state == RunStepState::Started));
        // A fold is known by its first step: steps added after it leave it
        // the same fold, so an opened one stays open.
        assert_eq!(laid[2].0, 1);
    }

    /// Every kind of step in every state, with and without the file or run
    /// it may name: the shapes a run's steps come in.
    fn every_shape() -> Vec<RunStepItem> {
        use RunStepKind::*;
        let mut shapes = Vec::new();
        for kind in [
            Read, Search, Edit, Run, Fetch, Delegate, Plan, Coordinate, HandOff, Other, Unknown,
        ] {
            for state in [
                RunStepState::Started,
                RunStepState::Done,
                RunStepState::Failed,
                RunStepState::Unknown,
            ] {
                let plain = step(kind, state);
                shapes.push(plain.clone());
                if matches!(kind, Read | Edit) {
                    let mut named = plain.clone();
                    named.path = Some("/w/synthetic/slug.py".into());
                    shapes.push(named);
                }
                if kind == HandOff {
                    let mut linked = plain.clone();
                    linked.run = Some("cli-agent-5e6f".into());
                    linked.agent = Some("Codex".into());
                    linked.words = "Handed work to Codex".into();
                    shapes.push(linked.clone());
                    linked.agent = Some("Gemini".into());
                    linked.words = "Sent a follow-up to Gemini".into();
                    shapes.push(linked);
                }
            }
        }
        // Each step tells itself apart, so "shown once" can be checked.
        shapes
    }

    /// The one rule's promises, for any steps: each step is shown once, in
    /// its order (those still going last); no two folds or two hand-off
    /// rows sit side by side; a row on its own is a step still going or a
    /// failure nothing finished after; a fold is known by its first step.
    fn check(steps: &[RunStepItem]) {
        let steps: Vec<RunStepItem> = steps
            .iter()
            .enumerate()
            .map(|(index, step)| {
                let mut step = step.clone();
                step.at_ms = Some(index as u64);
                step
            })
            .collect();
        let laid = rows(&steps);
        let mut shown: Vec<RunStepItem> = Vec::new();
        for (index, (first, row)) in laid.iter().enumerate() {
            let group = match row {
                Row::Work(group) | Row::HandOffs(group) => group.clone(),
                Row::Alone(step) => vec![step.clone()],
            };
            assert!(!group.is_empty(), "{laid:?}");
            assert_eq!(steps[*first], group[0], "a row is known by its first step");
            if let Some((_, before)) = index.checked_sub(1).map(|at| &laid[at]) {
                let same = matches!(
                    (before, row),
                    (Row::Work(_), Row::Work(_)) | (Row::HandOffs(_), Row::HandOffs(_))
                );
                assert!(!same, "two folds side by side: {laid:?}");
            }
            match row {
                Row::Alone(step) if step.state == RunStepState::Started => {}
                Row::Alone(step) => {
                    let at = step.at_ms.unwrap_or_default() as usize;
                    assert_eq!(step.state, RunStepState::Failed, "{laid:?}");
                    assert!(
                        !steps[at + 1..]
                            .iter()
                            .any(|later| later.state == RunStepState::Done),
                        "a failure the agent went on past stands alone: {laid:?}"
                    );
                }
                Row::Work(group) => {
                    assert!(group.iter().all(|step| step.state != RunStepState::Started));
                    // Each of a fold's steps is listed once when it opens.
                    let edits: usize = files_changed(group).iter().map(|(_, n)| n).sum();
                    assert_eq!(edits + rest_of_work(group).len(), group.len(), "{group:?}");
                    assert!(!work_summary(group).is_empty(), "{group:?}");
                }
                Row::HandOffs(group) => {
                    assert!(group.iter().all(|step| step.run.is_some()), "{group:?}");
                    // Only an agent's first hand-off stands out.
                    for step in group {
                        let at = step.at_ms.unwrap_or_default() as usize;
                        assert!(
                            !steps[..at]
                                .iter()
                                .any(|earlier| earlier.kind == RunStepKind::HandOff
                                    && earlier.state == RunStepState::Done
                                    && earlier.run.is_some()
                                    && earlier.agent == step.agent),
                            "{laid:?}"
                        );
                    }
                }
            }
            shown.extend(group);
        }
        let mut expected: Vec<RunStepItem> = steps
            .iter()
            .filter(|step| step.state != RunStepState::Started)
            .cloned()
            .collect();
        expected.extend(
            steps
                .iter()
                .filter(|step| step.state == RunStepState::Started)
                .cloned(),
        );
        assert_eq!(
            shown, expected,
            "each step once, in order, those going last"
        );
    }

    #[test]
    fn any_steps_keep_the_one_grouping_rule() {
        let shapes = every_shape();
        // Every list of up to three steps.
        check(&[]);
        for a in &shapes {
            check(std::slice::from_ref(a));
            for b in &shapes {
                check(&[a.clone(), b.clone()]);
                for c in &shapes {
                    check(&[a.clone(), b.clone(), c.clone()]);
                }
            }
        }
        // And long ones, as a run's hundreds of steps come.
        let mut seed: u64 = 0x5eed;
        for _ in 0..2_000 {
            let steps: Vec<RunStepItem> = (0..40)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    shapes[(seed >> 33) as usize % shapes.len()].clone()
                })
                .collect();
            check(&steps);
        }
    }

    /// Real runs' steps hold the rule too: `RUN_SHAPES` names a file of
    /// runs, one a line, each step `kind:state:names_file:has_run` (kinds
    /// and states only, exported from a daemon's records). Run by hand:
    /// `RUN_SHAPES=<file> cargo test --lib real_runs -- --ignored`.
    #[test]
    #[ignore]
    fn real_runs_keep_the_one_grouping_rule() {
        use RunStepKind::*;
        let path = std::env::var("RUN_SHAPES").expect("RUN_SHAPES names a file");
        let text = std::fs::read_to_string(path).expect("RUN_SHAPES is readable");
        let (mut runs, mut steps_seen, mut rows_shown, mut alone) = (0, 0, 0, 0);
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let steps: Vec<RunStepItem> = line
                .split_whitespace()
                .map(|token| {
                    let parts: Vec<&str> = token.split(':').collect();
                    let kind = match parts[0] {
                        "read" => Read,
                        "search" => Search,
                        "edit" => Edit,
                        "run" => Run,
                        "fetch" => Fetch,
                        "delegate" => Delegate,
                        "plan" => Plan,
                        "coordinate" => Coordinate,
                        "hand_off" => HandOff,
                        "other" => Other,
                        _ => Unknown,
                    };
                    let state = match parts[1] {
                        "started" => RunStepState::Started,
                        "done" => RunStepState::Done,
                        "failed" => RunStepState::Failed,
                        _ => RunStepState::Unknown,
                    };
                    let mut item = step(kind, state);
                    if parts[2] == "1" {
                        item.path = Some("/w/synthetic/file.rs".into());
                    }
                    if parts[3] == "1" {
                        item.run = Some("cli-agent-synthetic".into());
                        item.agent = Some("Codex".into());
                    }
                    item
                })
                .collect();
            check(&steps);
            let laid = rows(&steps);
            runs += 1;
            steps_seen += steps.len();
            rows_shown += laid.len();
            alone += laid
                .iter()
                .filter(|(_, row)| matches!(row, Row::Alone(_)))
                .count();
        }
        eprintln!("{runs} runs, {steps_seen} steps, {rows_shown} rows, {alone} standing alone");
    }

    /// Live 2026-11-27, kinds and states only: hand-offs with their run
    /// read as one row; the steps around them fold.
    #[test]
    fn hand_offs_in_a_row_are_one_row_between_folds() {
        use RunStepKind::*;
        let hand_off = |agent: &str| {
            let mut step = step(HandOff, RunStepState::Done);
            step.run = Some(format!("cli-agent-{agent}"));
            step.agent = Some(agent.into());
            step
        };
        let laid = rows(&[
            step(Read, RunStepState::Done),
            hand_off("Codex"),
            hand_off("Gemini"),
            step(Coordinate, RunStepState::Done),
            step(Unknown, RunStepState::Done),
            // A follow-up to an agent already at work folds, counted.
            hand_off("Codex"),
            step(Delegate, RunStepState::Done),
        ]);
        assert!(
            matches!(laid.as_slice(), [(_, Row::Work(_)), (_, Row::HandOffs(two)), (_, Row::Work(four))]
                if two.len() == 2 && four.len() == 4),
            "{laid:?}"
        );
        let Row::Work(folded) = &laid[2].1 else {
            unreachable!()
        };
        assert_eq!(
            work_summary(folded),
            "Coordinated once · used another tool · asked a helper · handed Codex work once"
        );
    }
    use aidaemon_client_protocol::{RunAsked, RunPhase, RunSummary};

    fn sample_run(asked: Option<RunAsked>) -> RunSummary {
        RunSummary {
            id: "cli-agent-1".into(),
            session_id: "app:synthetic-1".into(),
            label: "Review the codebase".into(),
            agent: Some("Antigravity CLI".into()),
            phase: RunPhase::Succeeded,
            background: true,
            step: String::new(),
            started_at_ms: 1000,
            ended_at_ms: Some(2000),
            took: None,
            reason: None,
            stopped_by_owner: false,
            out_of_usage: None,
            group: None,
            edited: Vec::new(),
            continues: None,
            thread: None,
            can_continue: true,
            accepts_mid_run: false,
            keeps_session: None,
            result: None,
            asked,
        }
    }

    #[test]
    fn owner_prompt_covers_direct_agent_and_owner_follow_ups() {
        let delegated = sample_run(Some(RunAsked {
            text: "Check the logs".into(),
            by_owner: false,
        }));
        assert!(!is_owner_prompt(&delegated, false));
        assert!(is_owner_prompt(&delegated, true));

        let follow_up = sample_run(Some(RunAsked {
            text: "Also add tests".into(),
            by_owner: true,
        }));
        assert!(is_owner_prompt(&follow_up, false));
        assert!(is_owner_prompt(&follow_up, true));

        let legacy = sample_run(None);
        assert!(!is_owner_prompt(&legacy, false));
        assert!(is_owner_prompt(&legacy, true));
    }

    #[test]
    fn short_reply_is_not_collapsed() {
        assert!(markdown_preview("Fixed the slug in `slug.py`.", FOCUSED_REPLY_BUDGET).is_none());
    }

    #[test]
    fn verbose_reply_collapses_and_closes_open_code_fence() {
        let mut text = String::from("Summary line.\n\n```rust\n");
        for i in 0..30 {
            text.push_str(&format!("let step_{i} = {i};\n"));
        }
        text.push_str("```\n\nClosing remarks.");
        let (preview, lines) =
            markdown_preview(&text, FOCUSED_REPLY_BUDGET).expect("verbose reply collapses");
        assert!(lines > 30);
        assert!(preview.starts_with("Summary line."));
        assert_eq!(
            preview.matches("```").count() % 2,
            0,
            "code fence closed: {preview}"
        );
    }

    #[test]
    fn clean_asked_text_strips_daemon_interrupted_recovery_note() {
        let raw = "[Where your last turn stopped] Your previous turn in this session (run cli-agent-1) did not finish: a daemon restart stopped it part way.\n- Steps finished: 5.\nCheck the current state of those files before redoing anything, then go on from there with the message below.\n\nContinue";
        assert_eq!(clean_asked_text(raw), "Continue");
        assert_eq!(clean_asked_text("Fix the bug"), "Fix the bug");
    }

    #[test]
    fn clean_asked_text_strips_prior_conversation_wrapper() {
        let raw = "## Prior conversation\nuser: Yes, fix it\nassistant: Done.\n\n## Current request\nThe ux is not that great on desktop here.";
        assert_eq!(
            clean_asked_text(raw),
            "The ux is not that great on desktop here."
        );

        let nested = "## Prior conversation\nuser: Check it\n\n## Current request\n[Where your last turn stopped] Partial run.\nCheck the current state of those files before redoing anything, then go on from there with the message below.\n\nContinue";
        assert_eq!(clean_asked_text(nested), "Continue");
    }
}
