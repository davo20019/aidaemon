//! The details pane's Changes tab: whether the last change was checked
//! (the last command run after it, and how it ended), then the uncommitted
//! changes in the git repositories the selected conversation worked in.
//! Each repository lists its changed files (those this conversation wrote
//! first); pressing one shows its diff. Read when the tab opens, on
//! Refresh, and after a tool that can change files finishes.

use aidaemon_client_protocol::{CheckRun, FileChange, FileStatus, RepoChanges};
use nsh_core::prelude::*;

use crate::store::{ChangesRead, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, radius, section_label, type_scale};

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
    // A changed file asked for (a run's "Changed" list): its diff, once the
    // changes are read, when it is among them. Each ask once per view.
    let handled = std::cell::Cell::new(0u64);
    effect(move || {
        let ChangesRead::Ready(changes) = store.changes() else {
            return;
        };
        let Some((number, path)) = store.change_to_open() else {
            return;
        };
        if number <= handled.get() {
            return;
        }
        handled.set(number);
        if let Some(found) = changed_file_named(&changes.repos, &path) {
            open.set(Some(found));
        }
    });
    view().flex_1().child(dynamic(move || {
        let shown = open.get();
        Some(untracked(move || match shown {
            Some((root, path)) => file_diff(store, root, path, open),
            None => view().flex_1().child(ui::refreshable(
                scroll()
                    .flex_1()
                    .child(view().px_3().pt_2().pb_4().gap_2().child(dynamic(move || {
                        let read = store.changes();
                        Some(untracked(move || body(store, read, open)))
                    }))),
                move || store.pull_refreshing(),
                move || store.pull_refresh(),
            )),
        }))
    }))
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
            let wrote = changes
                .repos
                .iter()
                .any(|repo| repo.files.iter().any(|f| f.by_session));
            let mut list = view().gap_3().child(check(changes.last_check, wrote));
            for repo in changes.repos {
                list = list.child(repo_view(store, repo, open));
            }
            list
        }
    }
}

/// Whether the conversation's last change was checked: the command, and
/// how it ended. Nothing when it wrote nothing that is still uncommitted.
fn check(run: Option<CheckRun>, wrote: bool) -> View {
    let (glyph, tone, line) = match run {
        Some(run) if run.passed => (
            Glyph::Check,
            Tone::Success,
            format!("Passed: {}", run.command),
        ),
        Some(run) => (
            Glyph::Close,
            Tone::Danger,
            match run.exit_code {
                Some(code) => format!("Failed (exit {code}): {}", run.command),
                None => format!("Failed: {}", run.command),
            },
        ),
        None if wrote => (
            Glyph::Dot,
            Tone::Warning,
            "Not checked: no command ran after the last change".to_owned(),
        ),
        None => return view(),
    };
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .px_2()
        .py_2()
        .rounded(px(radius::SMALL))
        .bg(colors::INSET)
        .child(ui::icon(glyph).text_color(tone.color()))
        .child(
            text(line)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(2)
                .font_family(ui::theme::MONO)
                .text_size(px(type_scale::XS))
                .text_color(colors::TEXT),
        )
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
        .child(file_list(root, files, open))
}

fn file_list(root: String, files: Vec<FileChange>, open: Signal<Option<(String, String)>>) -> View {
    let mut list = view().gap(px(2.0));
    if files.iter().any(|f| f.by_session) {
        list = list.child(section_label("Changed files"));
    }
    let rows = files
        .into_iter()
        .map(|file| {
            let (letter, tone) = super::files::status_letter(file.status);
            ui::TreeFile {
                label: format!(
                    "{}, {}{}",
                    file.path,
                    super::files::status_word(file.status),
                    if file.by_session {
                        ", written in this conversation"
                    } else {
                        ""
                    }
                ),
                stats: if file.added > 0 || file.removed > 0 {
                    format!("+{} −{}", file.added, file.removed)
                } else {
                    String::new()
                },
                path: file.path,
                letter,
                tone,
                marked: file.by_session,
            }
        })
        .collect();
    list.child(ui::file_tree("Changed files", rows, move |path| {
        open.set(Some((root.clone(), path)))
    }))
}

/// One file's diff, filling the pane: back to the list, its path, the
/// previous and next change, side by side or one over the other, Copy.
/// It reads the file's section again whenever the changes are read again.
fn file_diff(
    store: Store,
    root: String,
    path: String,
    open: Signal<Option<(String, String)>>,
) -> View {
    let split = signal(false);
    let section = {
        let (root, path) = (root.clone(), path.clone());
        computed(move || match store.changes() {
            ChangesRead::Ready(changes) => changes
                .repos
                .iter()
                .find(|repo| repo.root == root)
                .map(|repo| file_patch(&repo.patch, &path))
                .unwrap_or_default(),
            _ => String::new(),
        })
    };
    let diff: ui::DiffRef = component_ref();
    let label = path.clone();
    view()
        .flex_1()
        .min_h(px(0.0))
        .gap_1()
        .pt_1()
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .px_2()
                .child(ui::btn("All files", Variant::Ghost, Size::Sm, move || {
                    open.set(None)
                }))
                .child(
                    text(path.clone())
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                )
                .child(ui::icon_btn(
                    Glyph::Up,
                    "Previous change",
                    Size::Sm,
                    move || ui::diff_step(&diff, false),
                ))
                .child(ui::icon_btn(
                    Glyph::Down,
                    "Next change",
                    Size::Sm,
                    move || ui::diff_step(&diff, true),
                ))
                .child(ui::icon_btn(
                    Glyph::Columns,
                    "Side by side",
                    Size::Sm,
                    move || split.update(|s| *s = !*s),
                ))
                .child(ui::btn("Copy", Variant::Ghost, Size::Sm, move || {
                    drop(set_clipboard(section.get_untracked()));
                })),
        )
        .child(dynamic(move || {
            let patch = section.get();
            let label = label.clone();
            Some(untracked(move || {
                if has_lines(&patch) {
                    ui::diff(patch, label, move || split.get(), &diff)
                } else {
                    view().px_3().child(
                        text("No line changes to show (a binary file, a rename or a mode change).")
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
                }
            }))
        }))
        // Undoing sits apart, at the foot of the pane, away from the
        // header's controls; it asks before it changes anything.
        .child(
            view()
                .flex_row()
                .justify_end()
                .px_3()
                .py_2()
                .border_t_1()
                .border_color(colors::LINE)
                .child({
                    let (root, path) = (root.clone(), path.clone());
                    ui::btn("Undo changes…", Variant::Danger, Size::Sm, move || {
                        let new = file_status(store, &root, &path).is_some_and(|s| {
                            matches!(s, FileStatus::Untracked | FileStatus::Added)
                        });
                        let (root, path) = (root.clone(), path.clone());
                        ui::confirm(
                            format!("Undo the changes to {path}?"),
                            if new {
                                "It is a new file: undoing deletes it. This cannot be undone here."
                            } else {
                                "It goes back to its last commit. The uncommitted changes are lost."
                            },
                            if new {
                                "Delete the file"
                            } else {
                                "Undo changes"
                            },
                            move || {
                                store.revert_file(&root, &path);
                                open.set(None);
                            },
                        );
                    })
                    .disabled(move || store.reverting())
                }),
        )
}

/// Whether a file's section of a diff has any hunk.
fn has_lines(section: &str) -> bool {
    section.lines().any(|line| line.starts_with("@@"))
}

/// `path`'s section of a unified diff: from its `diff --git` line to the
/// next file's.
pub fn file_patch(patch: &str, path: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            inside = rest.ends_with(&format!(" b/{path}"));
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// How `path` in repository `root` changed, as last read.
fn file_status(store: Store, root: &str, path: &str) -> Option<FileStatus> {
    match untracked(|| store.changes()) {
        ChangesRead::Ready(changes) => changes
            .repos
            .iter()
            .find(|r| r.root == root)?
            .files
            .iter()
            .find(|f| f.path == path)
            .map(|f| f.status),
        _ => None,
    }
}

/// The repository root and repository-relative path of changed file `path`:
/// a full path under a repository's root, or (a run's report names files
/// from where it worked) a path relative to one.
fn changed_file_named(
    repos: &[aidaemon_client_protocol::RepoChanges],
    path: &str,
) -> Option<(String, String)> {
    repos.iter().find_map(|repo| {
        let root = repo.root.trim_end_matches('/');
        let relative = path
            .strip_prefix(&format!("{root}/"))
            .unwrap_or(path.trim_start_matches("./"));
        repo.files
            .iter()
            .any(|file| file.path == relative)
            .then(|| (repo.root.clone(), relative.to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::{changed_file_named, file_patch, has_lines};

    #[test]
    fn file_patch_keeps_one_files_section() {
        let patch = "diff --git a/a.rs b/a.rs\nindex 1..2 100644\n--- a/a.rs\n+++ b/a.rs\n\
                     @@ -1 +1 @@\n-old\n+new\n\
                     diff --git a/b.md b/b.md\nnew file\n--- /dev/null\n+++ b/b.md\n@@ -0,0 +1 @@\n+note\n";
        assert_eq!(
            file_patch(patch, "a.rs"),
            "diff --git a/a.rs b/a.rs\nindex 1..2 100644\n--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-old\n+new\n"
        );
        assert!(file_patch(patch, "b.md").ends_with("@@ -0,0 +1 @@\n+note\n"));
        assert!(file_patch(patch, "c.rs").is_empty());
        assert!(!has_lines("diff --git a/x b/x\nBinary files differ\n"));
    }

    #[test]
    fn a_changed_file_is_found_by_full_or_relative_path() {
        let repos: Vec<aidaemon_client_protocol::RepoChanges> = serde_json::from_str(
            r#"[{"root":"/work/acme","files":[{"path":"tests/test_slug.py","status":"modified","added":1,"removed":0}]}]"#,
        )
        .unwrap();
        let found = Some(("/work/acme".to_string(), "tests/test_slug.py".to_string()));
        assert_eq!(
            changed_file_named(&repos, "/work/acme/tests/test_slug.py"),
            found
        );
        assert_eq!(changed_file_named(&repos, "tests/test_slug.py"), found);
        assert_eq!(changed_file_named(&repos, "./tests/test_slug.py"), found);
        assert_eq!(
            changed_file_named(&repos, "/elsewhere/tests/test_slug.py"),
            None
        );
        assert_eq!(changed_file_named(&repos, "test_slug.py"), None);
    }
}
