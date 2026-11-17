//! The app's looks over `nsh_base`'s larger views: a diff, program output
//! in a terminal, a code editor, charts, a file tree, a searchable field
//! and a hover preview. Features use these, never the views directly.

use nsh_base::{DiffLayout, DiffView, DiffViewApi, diff_view};
use nsh_core::prelude::*;

use super::theme::{MONO, text as type_scale};

/// A unified diff (`git diff`'s section for one or more files), as code
/// review tools show it: line numbers, the words changed within lines,
/// unchanged lines collapsed into separators that expand, highlighted by
/// `language` (a file name works). Only the rows in view are made, so a
/// long diff needs no "Show all". `split` lays it side by side.
pub fn diff(
    patch: impl Into<SharedString>,
    language: impl Into<SharedString>,
    split: impl Fn() -> bool + 'static,
    handle: &DiffRef,
) -> View {
    let patch: SharedString = patch.into();
    let language: SharedString = language.into();
    view().flex_1().min_h(px(0.0)).child(
        diff_view()
            .patch(patch)
            .language(language)
            .layout(move || {
                if split() {
                    DiffLayout::Split
                } else {
                    DiffLayout::Unified
                }
            })
            .soft_wrap(true)
            .font_family(SharedString::from(MONO))
            .font_size(px(type_scale::XS))
            .line_height(px(18.0))
            .component_ref(handle)
            .size_full(),
    )
}

/// The ref of a [`diff`]'s view, for its next and previous change.
pub type DiffRef = ComponentRef<DiffView>;

/// Moves `diff` to its next (`forward`) or previous change.
pub fn diff_step(diff: &DiffRef, forward: bool) {
    if let Some(api) = diff.api::<DiffViewApi>() {
        if forward {
            api.next_hunk();
        } else {
            api.previous_hunk();
        }
    }
}

/// A program's output (a command's, a tool's) in a terminal: its colours
/// and cursor moves drawn as a terminal would, scrolled back like a list
/// and following the end while the owner is there. `content` is the whole
/// output so far; what it adds is written as it arrives, and anything else
/// starts the terminal over. `rows` is how many lines show.
pub fn output(
    content: impl Fn() -> String + 'static,
    rows: impl Fn() -> u32 + 'static,
    label: impl Into<SharedString>,
) -> View {
    use std::{cell::RefCell, rc::Rc};

    use nsh_base::{Terminal, TerminalApi, terminal};

    // In a terminal the output is its own text: emulating a terminal in a
    // few cells of one adds nothing, and nsh 0.5.0's emulator panics on a
    // double-width character in a one-column screen (live 2026-11-15, cargo
    // output in the Steps pane of `aidaemon-app --tui`).
    if idiom() == nsh_core::platform::Idiom::Terminal {
        return view().w_full().child(
            text(move || {
                let plain = aidaemon_client_core::text::plain_output(&content());
                let lines: Vec<&str> = plain.lines().collect();
                let keep = (rows() as usize).max(1);
                lines[lines.len().saturating_sub(keep)..].join("\n")
            })
            .selectable(true)
            .font_family(MONO)
            .text_size(px(type_scale::XS)),
        );
    }
    const LINE: f32 = 16.0;
    let term = component_ref::<Terminal>();
    let written: Rc<RefCell<String>> = Rc::default();
    let content = Rc::new(content);
    // Writes what is new since the last write; a terminal not yet laid out
    // is caught up when it is (its first resize).
    let sync = {
        let (written, content) = (written.clone(), content.clone());
        move || {
            let now = untracked(|| content());
            let Some(api) = term.api::<TerminalApi>() else {
                return;
            };
            let mut done = written.borrow_mut();
            if let Some(added) = now.strip_prefix(done.as_str()) {
                if !added.is_empty() {
                    api.write(for_terminal(added));
                }
            } else {
                api.reset();
                api.write(for_terminal(&now));
            }
            *done = now;
        }
    };
    {
        let (sync, content) = (sync.clone(), content.clone());
        effect(move || {
            content();
            sync();
        });
    }
    view()
        .w_full()
        // Never a screen narrow enough to break the emulator.
        .min_w(px(120.0))
        .h(move || Length::from(px(rows() as f32 * LINE + 8.0)))
        .child(
            terminal()
                .font_family(SharedString::from(MONO))
                .font_size(px(type_scale::XS))
                .line_height(px(LINE))
                .scrollback(5000)
                .accessibility_label(label.into())
                .on_resize(move |_: &nsh_base::TerminalEvent| sync())
                .component_ref(&term)
                .size_full(),
        )
}

/// Program output on an inset well: its last 8 lines in a terminal, or
/// 24 with Show more (the terminal scrolls back through the rest), with
/// Copy. `label` names it for screen readers.
pub fn output_panel(detail: impl Fn() -> String + Clone + 'static, label: &'static str) -> View {
    use super::{Size, Variant, btn, colors, radius};
    let all = signal(false);
    let lines = {
        let detail = detail.clone();
        move || detail().lines().count()
    };
    let more = {
        let lines = lines.clone();
        move || lines() > 8
    };
    let rows = {
        let lines = lines.clone();
        move || (lines().max(1) as u32).min(if all.get() { 24 } else { 8 })
    };
    let copy = detail.clone();
    view()
        .gap_1()
        .rounded(px(radius::SMALL))
        .bg(colors::INSET)
        .px_2()
        .py_1()
        .child(output(detail, rows, label))
        .child(
            view()
                .flex_row()
                .gap_1()
                .child(when(more, move || {
                    view().child(dynamic(move || {
                        let label = if all.get() { "Show less" } else { "Show more" };
                        Some(untracked(move || {
                            btn(label, Variant::Ghost, Size::Sm, move || {
                                all.update(|a| *a = !*a)
                            })
                        }))
                    }))
                }))
                .child(btn("Copy", Variant::Ghost, Size::Sm, move || {
                    set_clipboard(copy())
                })),
        )
}

/// Output for a terminal: lines end in a carriage return and a line feed,
/// as a program's own terminal would send them.
fn for_terminal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 32);
    let mut last = '\0';
    for c in text.chars() {
        if c == '\n' && last != '\r' {
            out.push('\r');
        }
        out.push(c);
        last = c;
    }
    out
}

/// A code editor over `value` (TOML, a script, a prompt): highlighted by
/// `language`, brackets closed and lines indented as an editor does,
/// growing from one row to `max_rows` before it scrolls. Line numbers show
/// once it holds more than one line.
pub fn code_editor(
    value: Signal<String>,
    language: &'static str,
    label: impl Into<SharedString>,
    max_rows: u32,
) -> View {
    use nsh_base::editor;

    use super::theme::{colors, radius};

    view()
        .w_full()
        .rounded(px(radius::CONTROL))
        .border_1()
        .border_color(colors::BORDER)
        .bg(colors::INSET)
        .focus_within(|s| s.border_color(colors::ACCENT))
        .py_1()
        .child(
            editor()
                .value(move || SharedString::from(value.get()))
                .language(SharedString::from(language))
                .on_change(move |text: &SharedString| value.set(text.to_string()))
                .line_numbers(move || value.with(|v| v.contains('\n')))
                .folding(false)
                .indent_guides(false)
                .active_line(false)
                .tab_size(2u32)
                .min_rows(1u32)
                .max_rows(max_rows)
                .font_family(SharedString::from(MONO))
                .font_size(px(type_scale::SM))
                .line_height(px(20.0))
                .accessibility_label(label.into())
                .w_full(),
        )
}

/// Two numbers per day (what went in, what came out), as stacked bars,
/// oldest at the left. `days` are the bars' names, in order; the bar under
/// the pointer names itself through `hovered`.
pub fn day_bars(
    label: impl Into<SharedString>,
    days: usize,
    first: Vec<f64>,
    second: Vec<f64>,
    hovered: Signal<Option<u32>>,
) -> View {
    use nsh_base::plot::{BarCornerRadii, bar, plot};
    use nsh_core::element::{RowAccessor, Rows};

    use super::theme::colors;

    let rows = Rows::from_columns(days, [("first".into(), first), ("second".into(), second)]);
    view().w_full().h(px(96.0)).child(
        plot()
            .data(rows)
            .y(RowAccessor::key("first"))
            .accessibility_label(label.into())
            .on_hover_change(move |index: &Option<u32>| hovered.set(*index))
            .size_full()
            .child(
                bar()
                    .value(RowAccessor::key("first"))
                    .stack(SharedString::from("day"))
                    .fill_color(colors::MUTED)
                    .inactive_opacity(Percentage(0.5)),
            )
            .child(
                bar()
                    .value(RowAccessor::key("second"))
                    .stack(SharedString::from("day"))
                    .corner_radii(BarCornerRadii::from(px(2.0)))
                    .fill_color(colors::ACCENT)
                    .inactive_opacity(Percentage(0.5)),
            ),
    )
}

/// A file of a [`file_tree`]: its path (`/` between folders) and what its
/// row shows beside its name.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeFile {
    pub path: String,
    /// A letter for how it changed (M, A, D...), in `tone`.
    pub letter: &'static str,
    pub tone: super::theme::Tone,
    /// `+3 −1`, or empty.
    pub stats: String,
    /// Marked with an accent dot (written in this conversation).
    pub marked: bool,
    /// What screen readers say for its row.
    pub label: String,
}

/// Files under their folders, folders open: a folder of one folder shows
/// as one row (`src/ui`). Pressing a file calls `open` with its path.
pub fn file_tree(
    label: &'static str,
    files: Vec<TreeFile>,
    open: impl Fn(String) + 'static,
) -> View {
    use std::rc::Rc;

    use nsh_base::{TreeEntry, tree};

    use super::controls::{Glyph, icon};
    use super::theme::{colors, fit};

    let items = tree_items(&files);
    let mut folders = Vec::new();
    folder_ids(&items, &mut folders);
    let expanded = signal(folders);
    let items_for_count = items.clone();
    let rows = move || visible_rows(&items_for_count, &expanded.get()) as f32;
    let files = Rc::new(files);
    let row_h = fit::row();
    view()
        .w_full()
        .h(move || Length::from(px(rows() * row_h)))
        .child(
            tree()
                .items(items)
                .accessibility_label(SharedString::from(label))
                .expanded(move || Some(expanded.get()))
                .on_expanded_change(move |ids: &Vec<SharedString>| expanded.set(ids.clone()))
                // The keyboard's row shows, as every other control's does.
                .item_class("rounded-md focus-visible:ring-2 focus-visible:ring-ring")
                .selected(nsh_base::SelectValue::Controlled(None))
                .on_selected_change(move |id: &SharedString| {
                    if let Some(path) = id.strip_prefix("f:") {
                        open(path.to_owned())
                    }
                })
                .render_item(move |e: &TreeEntry| {
                    let indent = 4.0 + e.depth as f32 * 12.0;
                    let mut row = view()
                        .h(px(row_h))
                        .w_full()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .pl(px(indent))
                        .pr_2()
                        .rounded(px(super::theme::radius::CONTROL))
                        .hover(|s| s.bg(colors::HOVER));
                    if e.folder {
                        return row
                            .child(icon(Glyph::Folder).text_color(colors::FAINT))
                            .child(
                                text(e.label.clone())
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .number_of_lines(1)
                                    .font_family(MONO)
                                    .text_size(px(type_scale::XS))
                                    .text_color(colors::MUTED),
                            );
                    }
                    let path = e.id.strip_prefix("f:").unwrap_or(&e.id);
                    let Some(file) = files.iter().find(|f| f.path == path) else {
                        return row.child(text(e.label.clone()));
                    };
                    let name = path.rsplit('/').next().unwrap_or(path).to_owned();
                    row = row
                        .child(
                            text(file.letter)
                                .w(px(14.0))
                                .font_family(MONO)
                                .text_size(px(type_scale::XS))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(file.tone.color()),
                        )
                        .child(
                            text(name)
                                .flex_1()
                                .min_w(px(0.0))
                                .number_of_lines(1)
                                .font_family(MONO)
                                .text_size(px(type_scale::XS))
                                .text_color(if file.marked {
                                    colors::TEXT
                                } else {
                                    colors::MUTED
                                }),
                        );
                    if !file.stats.is_empty() {
                        row = row.child(
                            text(file.stats.clone())
                                .font_family(MONO)
                                .text_size(px(type_scale::XS))
                                .text_color(colors::FAINT),
                        );
                    }
                    if file.marked {
                        row = row.child(icon(Glyph::Dot).text_color(colors::ACCENT));
                    }
                    row
                })
                .size_full(),
        )
}

/// The tree of `files`' paths: folders (`d:` ids) before files (`f:`
/// ids), each in path order, a folder holding only a folder joined to it.
fn tree_items(files: &[TreeFile]) -> Vec<nsh_base::TreeItem> {
    use std::collections::BTreeMap;

    use nsh_base::TreeItem;

    #[derive(Default)]
    struct Node {
        folders: BTreeMap<String, Node>,
        files: Vec<(String, String)>,
    }
    let mut root = Node::default();
    for file in files {
        let parts: Vec<&str> = file.path.split('/').filter(|p| !p.is_empty()).collect();
        let Some((_, dirs)) = parts.split_last() else {
            continue;
        };
        let mut node = &mut root;
        for dir in dirs {
            node = node.folders.entry((*dir).to_owned()).or_default();
        }
        node.files.push((file.label.clone(), file.path.clone()));
    }
    fn build(node: Node, prefix: &str) -> Vec<TreeItem> {
        let mut items = Vec::new();
        for (name, mut child) in node.folders {
            let mut name = name;
            // A folder holding one folder and nothing else is one row.
            while child.files.is_empty() && child.folders.len() == 1 {
                let (inner, grandchild) = child.folders.pop_first().expect("one folder");
                name = format!("{name}/{inner}");
                child = grandchild;
            }
            let path = format!("{prefix}{name}/");
            items.push(TreeItem::new(format!("d:{path}"), name).children(build(child, &path)));
        }
        let mut files = node.files;
        files.sort_by(|a, b| a.1.cmp(&b.1));
        // A file's label is what screen readers say; its row shows its name.
        for (label, path) in files {
            items.push(TreeItem::new(format!("f:{path}"), label));
        }
        items
    }
    build(root, "")
}

fn folder_ids(items: &[nsh_base::TreeItem], out: &mut Vec<SharedString>) {
    for item in items {
        if !item.children.is_empty() {
            out.push(item.id.clone());
            folder_ids(&item.children, out);
        }
    }
}

/// How many rows show with `expanded` open.
fn visible_rows(items: &[nsh_base::TreeItem], expanded: &[SharedString]) -> usize {
    items
        .iter()
        .map(|item| {
            1 + if !item.children.is_empty() && expanded.contains(&item.id) {
                visible_rows(&item.children, expanded)
            } else {
                0
            }
        })
        .sum()
}

#[cfg(test)]
mod tree_tests {
    use super::{TreeFile, tree_items, visible_rows};
    use crate::ui::Tone;

    fn file(path: &str) -> TreeFile {
        TreeFile {
            path: path.into(),
            letter: "M",
            tone: Tone::Warning,
            stats: String::new(),
            marked: false,
            label: path.into(),
        }
    }

    #[test]
    fn folders_of_one_folder_join_and_files_follow_folders() {
        let items = tree_items(&[
            file("src/ui/rich.rs"),
            file("src/ui/kit.rs"),
            file("README.md"),
            file("crates/core/src/lib.rs"),
        ]);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_ref()).collect();
        assert_eq!(labels, ["crates/core/src", "src/ui", "README.md"]);
        assert_eq!(items[1].children.len(), 2);
        assert_eq!(items[1].children[0].id.as_ref(), "f:src/ui/kit.rs");
        assert_eq!(visible_rows(&items, &[]), 3);
        assert_eq!(visible_rows(&items, &[items[1].id.clone()]), 5);
    }
}

/// A text field that suggests from `suggestions` as the owner types (a
/// setting's key, a name): the list filters by what is typed, a pick fills
/// the field, and anything else typed stays as typed.
pub fn suggest_field(
    label: impl Into<SharedString>,
    placeholder: &'static str,
    value: Signal<String>,
    suggestions: Vec<String>,
) -> View {
    use nsh_base::{SelectEntry, SelectItem, combobox};

    use super::theme::{colors, fit, radius};

    view().w_full().child(
        combobox()
            .items(
                suggestions
                    .into_iter()
                    .map(|s| SelectItem::new(s.clone(), s))
                    .collect::<Vec<_>>(),
            )
            .placeholder(placeholder)
            .accessibility_label(label.into())
            .query(move || Some(SharedString::from(value.get())))
            .on_query_change(move |q: &SharedString| value.set(q.to_string()))
            .value(move || Some(Vec::new()))
            .on_change(move |picked: &Vec<SharedString>| {
                if let Some(first) = picked.first() {
                    value.set(first.to_string())
                }
            })
            .input_class(if super::touch() {
                "flex-1 w-full h-12 px-3 text-base rounded-md border border-border bg-card focus:border-ring"
            } else {
                "flex-1 w-full h-8 px-3 text-sm rounded-md border border-border bg-card focus:border-ring"
            })
            .popup_style(|s| {
                s.bg(colors::SURFACE)
                    .border_1()
                    .border_color(colors::EDGE)
                    .rounded(px(radius::CONTROL))
                    .shadow_lg()
                    .p_1()
                    .max_h(px(280.0))
            })
            .item_class("rounded-sm focus:bg-accent")
            .render_item(|e: &SelectEntry| {
                text(e.title.clone())
                    .px_2()
                    .py(px(5.0))
                    .number_of_lines(1)
                    .font_family(MONO)
                    .text_size(px(fit::small()))
                    .text_color(colors::TEXT)
            })
            .w_full(),
    )
}

/// `trigger` with a card of more about it shown while the pointer rests on
/// it (or it has the keyboard focus). Touch screens and terminals have no
/// hover: there, a press keeps meaning what it meant, and only `trigger`
/// shows.
pub fn with_preview(trigger: impl IntoElement + 'static, card: impl FnOnce() -> View) -> View {
    use nsh_base::hover_card;

    use super::theme::{colors, radius};

    if super::touch() || idiom() == nsh_core::platform::Idiom::Terminal {
        return view().child(trigger);
    }
    view().child(
        hover_card()
            .open_delay(600.0)
            .close_delay(150.0)
            .offset(px(6.0))
            .placement(Placement::Right)
            .popup_style(|s| {
                s.bg(colors::SURFACE)
                    .border_1()
                    .border_color(colors::EDGE)
                    .rounded(px(radius::CONTROL))
                    .shadow_lg()
                    .p_3()
                    .w(px(280.0))
            })
            .popup_starting_class("opacity-0")
            .popup_ending_class("opacity-0")
            .trigger(trigger)
            .child(card()),
    )
}

#[cfg(test)]
mod tests {
    use super::for_terminal;

    #[test]
    fn line_feeds_return_the_carriage() {
        assert_eq!(for_terminal("a\nb\r\nc"), "a\r\nb\r\nc");
    }
}
