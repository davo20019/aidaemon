//! The inspector's Code tab: the uncommitted changes in the git
//! repositories the selected conversation worked in. Each repository lists
//! its changed files (those this conversation wrote first); pressing one
//! shows its diff. Read when the tab opens, on Refresh, and after a tool
//! that can change files finishes.

use aidaemon_client_protocol::{FileChange, FileStatus, RepoChanges};
use nsh_core::prelude::*;

use crate::store::{ChangesRead, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, radius, section_label, type_scale};

/// A file's diff shows this many lines until Show all.
const DIFF_LINES: usize = 400;

pub fn code(store: Store) -> View {
    // Read on opening, and whenever another conversation is selected.
    effect(move || {
        store.selected();
        untracked(move || store.refresh_changes_if_stale());
    });
    // The file whose diff is open: (repository root, path).
    let open: Signal<Option<(String, String)>> = signal(None);
    effect(move || {
        store.selected();
        open.set(None);
    });
    view().flex_1().child(
        scroll()
            .flex_1()
            .child(view().px_3().pt_2().pb_4().gap_2().child(dynamic(move || {
                let read = store.changes();
                Some(untracked(move || body(store, read, open)))
            }))),
    )
}

fn body(store: Store, read: ChangesRead, open: Signal<Option<(String, String)>>) -> View {
    match read {
        ChangesRead::NotYet | ChangesRead::Loading => view().child(ui::working("Reading changes…")),
        ChangesRead::Failed(reason) => view().child(ui::banner(reason, Tone::Danger).child(
            ui::btn("Try again", Variant::Secondary, Size::Sm, move || {
                store.refresh_changes()
            }),
        )),
        ChangesRead::Ready(changes) if changes.repos.is_empty() => view()
            .gap_2()
            .child(ui::empty(
                "No code changes",
                "When this conversation works in a git repository, its uncommitted \
                 changes show here.",
            ))
            .child(refresh(store)),
        ChangesRead::Ready(changes) => {
            let mut list = view().gap_3();
            for repo in changes.repos {
                list = list.child(repo_view(store, repo, open));
            }
            list
        }
    }
}

fn refresh(store: Store) -> View {
    view().flex_row().justify_center().child(ui::btn(
        "Refresh",
        Variant::Ghost,
        Size::Sm,
        move || store.refresh_changes(),
    ))
}

fn repo_view(store: Store, repo: RepoChanges, open: Signal<Option<(String, String)>>) -> View {
    let name = repo
        .root
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or(repo.root.as_str())
        .to_string();
    let (added, removed) = repo
        .files
        .iter()
        .fold((0u32, 0u32), |(a, r), f| (a + f.added, r + f.removed));
    let summary = match repo.files.len() {
        0 => "No uncommitted changes".to_string(),
        1 => format!("1 file · +{added} −{removed}"),
        n => format!("{n} files · +{added} −{removed}"),
    };
    let root = repo.root.clone();
    let mut header = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            text(name)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Sm, move || {
            store.refresh_changes()
        }));
    if let Some(branch) = &repo.branch {
        header = header.child(ui::badge(branch.clone(), Tone::Neutral));
    }
    let patch = repo.patch.clone();
    let files = repo.files.clone();
    view()
        .gap_1()
        .child(header)
        .child(
            text(summary)
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        )
        .child(when(
            move || repo.truncated,
            || {
                ui::banner(
                    "This diff is too large to show in full; commit or stash part of it to see the rest.",
                    Tone::Warning,
                )
            },
        ))
        .child(dynamic(move || {
            let shown = open.get().filter(|(r, _)| *r == root);
            let (root, files, patch) = (root.clone(), files.clone(), patch.clone());
            Some(untracked(move || match shown {
                Some((_, path)) => file_diff(&path, &patch, open),
                None => file_list(root, files, open),
            }))
        }))
}

fn file_list(root: String, files: Vec<FileChange>, open: Signal<Option<(String, String)>>) -> View {
    let mut list = view().gap(px(2.0));
    if files.iter().any(|f| f.by_session) {
        list = list.child(section_label("Changed files"));
    }
    for file in files {
        let target = (root.clone(), file.path.clone());
        let (letter, tone) = status(file.status);
        let mut row = ui::row(|| false, move || open.set(Some(target.clone())))
            .accessibility_label(format!(
                "{}, {}{}",
                file.path,
                status_word(file.status),
                if file.by_session {
                    ", written in this conversation"
                } else {
                    ""
                }
            ))
            .child(
                text(letter)
                    .w(px(14.0))
                    .font_family(ui::theme::MONO)
                    .text_size(px(type_scale::XS))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(tone.color()),
            )
            .child(
                text(file.path.clone())
                    .flex_1()
                    .min_w(px(0.0))
                    .number_of_lines(1)
                    .font_family(ui::theme::MONO)
                    .text_size(px(type_scale::XS))
                    .text_color(if file.by_session {
                        colors::TEXT
                    } else {
                        colors::MUTED
                    }),
            );
        if file.added > 0 || file.removed > 0 {
            row = row.child(
                text(format!("+{} −{}", file.added, file.removed))
                    .font_family(ui::theme::MONO)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            );
        }
        if file.by_session {
            row = row.child(ui::icon(Glyph::Dot).text_color(colors::ACCENT));
        }
        list = list.child(row);
    }
    list
}

fn file_diff(path: &str, patch: &str, open: Signal<Option<(String, String)>>) -> View {
    let lines = diff_of(patch, path);
    let total = lines.len();
    let all = signal(false);
    let copy = lines.join("\n");
    view()
        .gap_1()
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .child(ui::btn("All files", Variant::Ghost, Size::Sm, move || {
                    open.set(None)
                }))
                .child(
                    text(path.to_string())
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                )
                .child(ui::btn("Copy", Variant::Ghost, Size::Sm, move || {
                    set_clipboard(copy.clone())
                })),
        )
        .child(dynamic(move || {
            let every = all.get();
            let lines = lines.clone();
            Some(untracked(move || {
                let mut block = view().rounded(px(radius::SMALL)).bg(colors::INSET).py_1();
                if lines.is_empty() {
                    block = block.child(
                        text("No line changes to show (a binary file, a rename or a mode change).")
                            .px_2()
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    );
                }
                let shown = if every { total } else { total.min(DIFF_LINES) };
                for line in lines.into_iter().take(shown) {
                    block = block.child(diff_line(line));
                }
                block
            }))
        }))
        .child(when(
            move || total > DIFF_LINES && !all.get(),
            move || {
                ui::btn(
                    format!("Show all {total} lines"),
                    Variant::Ghost,
                    Size::Sm,
                    move || all.set(true),
                )
            },
        ))
}

fn diff_line(line: String) -> View {
    let tone = if line.starts_with("@@") {
        Some(Tone::Info)
    } else if line.starts_with('+') {
        Some(Tone::Success)
    } else if line.starts_with('-') {
        Some(Tone::Danger)
    } else {
        None
    };
    let row = view().px_2();
    let row = match tone {
        Some(Tone::Success) | Some(Tone::Danger) => row.bg(tone.unwrap().tint()),
        _ => row,
    };
    row.child(
        text(if line.is_empty() {
            " ".to_string()
        } else {
            line
        })
        .selectable(true)
        .font_family(ui::theme::MONO)
        .text_size(px(type_scale::XS))
        .text_color(tone.map_or(colors::MUTED, Tone::color)),
    )
}

/// The hunk lines of `path`'s section of a unified diff, without its
/// `diff --git`, `index`, `---` and `+++` header lines.
pub fn diff_of(patch: &str, path: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut inside = false;
    let mut in_hunks = false;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            inside = rest.ends_with(&format!(" b/{path}"));
            in_hunks = false;
            continue;
        }
        if !inside {
            continue;
        }
        if line.starts_with("@@") {
            in_hunks = true;
        }
        if in_hunks || line.starts_with("Binary") {
            lines.push(line.to_string());
        }
    }
    lines
}

fn status(status: FileStatus) -> (&'static str, Tone) {
    match status {
        FileStatus::Added => ("A", Tone::Success),
        FileStatus::Untracked => ("U", Tone::Success),
        FileStatus::Deleted => ("D", Tone::Danger),
        FileStatus::Renamed => ("R", Tone::Info),
        FileStatus::Modified => ("M", Tone::Warning),
        FileStatus::Unknown => ("?", Tone::Neutral),
    }
}

fn status_word(status: FileStatus) -> &'static str {
    match status {
        FileStatus::Added => "added",
        FileStatus::Untracked => "new, not yet in git",
        FileStatus::Deleted => "deleted",
        FileStatus::Renamed => "renamed",
        FileStatus::Modified => "modified",
        FileStatus::Unknown => "changed",
    }
}

#[cfg(test)]
mod tests {
    use super::diff_of;

    #[test]
    fn diff_of_keeps_one_files_hunks() {
        let patch = "diff --git a/a.rs b/a.rs\nindex 1..2 100644\n--- a/a.rs\n+++ b/a.rs\n\
                     @@ -1 +1 @@\n-old\n+new\n\
                     diff --git a/b.md b/b.md\nnew file\n--- /dev/null\n+++ b/b.md\n@@ -0,0 +1 @@\n+note\n";
        assert_eq!(diff_of(patch, "a.rs"), ["@@ -1 +1 @@", "-old", "+new"]);
        assert_eq!(diff_of(patch, "b.md"), ["@@ -0,0 +1 @@", "+note"]);
        assert!(diff_of(patch, "c.rs").is_empty());
    }
}
