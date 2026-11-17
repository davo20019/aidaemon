//! The details pane's Files tab: the folder of the conversation's project,
//! read-only, one folder at a time (a phone has no room for a deep tree).
//! A path at the top goes back up; a folder opens in place; a file opens
//! filling the pane, highlighted, with Back to its folder. Files git
//! ignores are hidden until asked for; a file holding secrets shows by
//! name only. Read when the tab opens, on another conversation's project,
//! and on Refresh.

use aidaemon_client_protocol::{EntryKind, FileStatus, ProjectEntry, ProjectSummary, Withheld};
use nsh_core::prelude::*;

use crate::store::{FileRead, FolderRead, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, type_scale};

pub fn files(store: Store) -> View {
    let project = computed(move || store.files_project());
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let project = project.get();
        Some(untracked(move || match project {
            Some(project) => browser(store, project),
            None => view().child(ui::empty(
                "No project folder",
                "Files show for a conversation in a project with a folder.",
            )),
        }))
    }))
}

/// One project's files: the folder open, or the file open in it.
fn browser(store: Store, project: ProjectSummary) -> View {
    let id = project.id.clone();
    // The folder open (relative, "" for the project's own), whether
    // ignored files show, and the file open.
    let folder = signal(String::new());
    let ignored = signal(false);
    let open: Signal<Option<String>> = signal(None);
    {
        let id = id.clone();
        effect(move || {
            let (path, all) = (folder.get(), ignored.get());
            let id = id.clone();
            untracked(move || store.list_project_files(&id, &path, all));
        });
    }
    {
        let id = id.clone();
        effect(move || match open.get() {
            Some(path) => untracked(|| store.read_project_file(&id, &path)),
            None => untracked(|| store.close_project_file()),
        });
    }
    let name = project.name.clone();
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let shown = open.get();
        let name = name.clone();
        let id = id.clone();
        Some(untracked(move || match shown {
            Some(path) => file_view(store, path, open),
            None => folder_view(store, id, name, folder, ignored, open),
        }))
    }))
}

fn folder_view(
    store: Store,
    id: String,
    name: String,
    folder: Signal<String>,
    ignored: Signal<bool>,
    open: Signal<Option<String>>,
) -> View {
    view()
        .flex_1()
        .min_h(px(0.0))
        .child(crumbs(store, id, name, folder, ignored))
        .child(
            scroll()
                .flex_1()
                .child(view().px_2().pb_4().gap(px(2.0)).child(dynamic(move || {
                    let read = store.folder();
                    Some(untracked(move || listing(read, folder, open)))
                }))),
        )
}

/// Where in the project: its name, then each folder down to the open one,
/// each pressable; Refresh, and whether ignored files show.
fn crumbs(
    store: Store,
    id: String,
    name: String,
    folder: Signal<String>,
    ignored: Signal<bool>,
) -> View {
    let path = dynamic(move || {
        let current = folder.get();
        let name = name.clone();
        Some(untracked(move || {
            let mut row = view().flex_row().flex_wrap().items_center().gap_1();
            row = row.child(ui::btn(name, Variant::Ghost, Size::Sm, move || {
                folder.set(String::new())
            }));
            let mut so_far = String::new();
            for part in current.split('/').filter(|p| !p.is_empty()) {
                if !so_far.is_empty() {
                    so_far.push('/');
                }
                so_far.push_str(part);
                let target = so_far.clone();
                row = row
                    .child(
                        text("/")
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
                    .child(ui::btn(
                        part.to_owned(),
                        Variant::Ghost,
                        Size::Sm,
                        move || folder.set(target.clone()),
                    ));
            }
            row
        }))
    });
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .py_1()
        .border_b_1()
        .border_color(colors::LINE)
        .child(view().flex_1().min_w(px(0.0)).child(path))
        .child(dynamic(move || {
            let branch = store.folder().ready().and_then(|d| d.branch.clone());
            Some(untracked(move || match branch {
                Some(branch) => ui::badge(branch, Tone::Neutral),
                None => view(),
            }))
        }))
        .child(dynamic(move || {
            let label = if ignored.get() {
                "Hide ignored"
            } else {
                "Show ignored"
            };
            Some(untracked(move || {
                view().child(ui::btn(label, Variant::Ghost, Size::Sm, move || {
                    ignored.update(|i| *i = !*i)
                }))
            }))
        }))
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Sm, move || {
            store.list_project_files(&id, &folder.get_untracked(), ignored.get_untracked())
        }))
}

fn listing(read: FolderRead, folder: Signal<String>, open: Signal<Option<String>>) -> View {
    match read {
        FolderRead::NotYet | FolderRead::Loading => {
            view().p_2().child(ui::working("Reading files…"))
        }
        FolderRead::Failed(reason) => view().p_2().child(ui::banner(reason, Tone::Danger)),
        FolderRead::Ready(dir) if dir.entries.is_empty() => view().child(ui::empty(
            "Empty folder",
            "Nothing here, or only files git ignores.",
        )),
        FolderRead::Ready(dir) => {
            let mut list = view().gap(px(2.0));
            if !dir.path.is_empty() {
                let up = dir
                    .path
                    .rsplit_once('/')
                    .map(|(up, _)| up.to_owned())
                    .unwrap_or_default();
                list = list.child(
                    ui::row(|| false, move || folder.set(up.clone()))
                        .accessibility_label("Up one folder")
                        .child(ui::icon(Glyph::Back).text_color(colors::FAINT))
                        .child(
                            text("..")
                                .font_family(ui::theme::MONO)
                                .text_size(px(type_scale::SM))
                                .text_color(colors::MUTED),
                        ),
                );
            }
            for entry in dir.entries {
                let path = if dir.path.is_empty() {
                    entry.name.clone()
                } else {
                    format!("{}/{}", dir.path, entry.name)
                };
                list = list.child(entry_row(entry, path, folder, open));
            }
            if dir.truncated {
                list = list.child(ui::banner(
                    "This folder holds more files than are shown.",
                    Tone::Warning,
                ));
            }
            list
        }
    }
}

fn entry_row(
    entry: ProjectEntry,
    path: String,
    folder: Signal<String>,
    open: Signal<Option<String>>,
) -> View {
    let is_folder = entry.kind == EntryKind::Folder;
    let mut label = entry.name.clone();
    if is_folder {
        label.push_str(", folder");
    }
    if let Some(status) = entry.status {
        label.push_str(", ");
        label.push_str(status_word(status));
    }
    if entry.changed {
        label.push_str(", has changes");
    }
    if entry.secret {
        label.push_str(", holds secrets");
    }
    let press = move || {
        if is_folder {
            folder.set(path.clone())
        } else {
            open.set(Some(path.clone()))
        }
    };
    let ink = if entry.ignored {
        colors::FAINT
    } else {
        colors::TEXT
    };
    let mut row = ui::row(|| false, press)
        .accessibility_label(label)
        .child(
            ui::icon(if is_folder {
                Glyph::Folder
            } else {
                Glyph::File
            })
            .text_color(colors::FAINT),
        )
        .child(
            text(entry.name)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .font_family(ui::theme::MONO)
                .text_size(px(type_scale::SM))
                .text_color(ink),
        );
    if entry.secret {
        row = row.child(ui::badge("secret", Tone::Warning));
    }
    if entry.changed {
        row = row.child(ui::icon(Glyph::Dot).text_color(Tone::Warning.color()));
    }
    if let Some(status) = entry.status {
        let (letter, tone) = status_letter(status);
        row = row.child(
            text(letter)
                .font_family(ui::theme::MONO)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(tone.color()),
        );
    }
    if let Some(size) = entry.size_bytes {
        row = row.child(
            text(size_words(size))
                .font_family(ui::theme::MONO)
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        );
    }
    view().child(row)
}

/// The open file, filling the pane: Back, its path, Copy, then its text.
fn file_view(store: Store, path: String, open: Signal<Option<String>>) -> View {
    let back = path
        .rsplit_once('/')
        .map(|(folder, _)| folder.rsplit('/').next().unwrap_or(folder).to_owned())
        .unwrap_or_else(|| "All files".to_owned());
    view()
        .flex_1()
        .min_h(px(0.0))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .px_2()
                .py_1()
                .border_b_1()
                .border_color(colors::LINE)
                .child(ui::btn_icon(
                    Glyph::Back,
                    back,
                    Variant::Ghost,
                    Size::Sm,
                    move || open.set(None),
                ))
                .child(
                    text(path.clone())
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                )
                .child(ui::btn("Copy path", Variant::Ghost, Size::Sm, {
                    let path = path.clone();
                    move || set_clipboard(path.clone())
                })),
        )
        .child(dynamic(move || {
            let read = store.open_file();
            Some(untracked(move || file_body(read)))
        }))
}

fn file_body(read: FileRead) -> View {
    match read {
        FileRead::NotYet | FileRead::Loading => {
            view().p_3().child(ui::working("Reading the file…"))
        }
        FileRead::Failed(reason) => view().p_3().child(ui::banner(reason, Tone::Danger)),
        FileRead::Ready(file) => match (file.text, file.withheld) {
            (Some(text), _) => {
                let name = file
                    .path
                    .rsplit('/')
                    .next()
                    .unwrap_or(&file.path)
                    .to_owned();
                let mut body = view().flex_1().min_h(px(0.0));
                if file.truncated {
                    body = body.child(view().p_2().child(ui::banner(
                        format!(
                            "Only the start of this file shows: it is {}.",
                            size_words(file.size_bytes)
                        ),
                        Tone::Warning,
                    )));
                }
                body.child(ui::code_viewer(text, name.clone(), name))
            }
            (None, Some(Withheld::Secret)) => view().child(ui::empty(
                "Not shown",
                "This file holds secrets (keys, credentials or settings), so its text \
                 never leaves the computer it is on.",
            )),
            (None, _) => view().child(ui::empty(
                "Not a text file",
                format!(
                    "{}; open it on the computer it is on.",
                    size_words(file.size_bytes)
                ),
            )),
        },
    }
}

fn status_letter(status: FileStatus) -> (&'static str, Tone) {
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

/// `1.2 KB`, `340 B`.
fn size_words(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut size = bytes as f64 / 1024.0;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if size < 10.0 {
        format!("{size:.1} {}", UNITS[unit])
    } else {
        format!("{size:.0} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::size_words;

    #[test]
    fn sizes_read_short() {
        assert_eq!(size_words(340), "340 B");
        assert_eq!(size_words(1_229), "1.2 KB");
        assert_eq!(size_words(52_000), "51 KB");
        assert_eq!(size_words(3 * 1024 * 1024), "3.0 MB");
    }
}
