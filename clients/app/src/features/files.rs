//! A project's files, two ways in: the details pane's Files tab (the
//! conversation's project) and a page of their own from the project's
//! sidebar page (no conversation needed). Both are this browser: the folder of the conversation's project,
//! read-only, one folder at a time (a phone has no room for a deep tree).
//! A path at the top goes back up; a folder opens in place; a file opens
//! filling the pane, highlighted, with Back to its folder. Files git
//! ignores are hidden until asked for; a file holding secrets shows by
//! name only. Read when the tab opens, on another conversation's project,
//! and on Refresh.

use aidaemon_client_protocol::{
    AttachmentRef, EntryKind, FileStatus, ProjectEntry, ProjectSummary, Withheld,
};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::store::{FileRead, FolderRead, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, type_scale};

/// The details pane's Files tab.
pub fn files(store: Store) -> View {
    shown(store, false)
}

/// The browser for the project in view; `in_page` on the Files page.
fn shown(store: Store, in_page: bool) -> View {
    let project = computed(move || {
        if in_page {
            store.files_project()
        } else {
            store.tab_files_project()
        }
    });
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let project = project.get();
        Some(untracked(move || match project {
            Some(project) => browser(store, project, in_page),
            None => view().child(ui::empty(
                "No project folder",
                "Files show for a conversation in a project with a folder.",
            )),
        }))
    }))
}

/// The Files page: a bar like the other pages', then the browser.
pub fn page(store: Store) -> View {
    let title = computed(move || {
        store
            .files_project()
            .map(|p| p.name)
            .unwrap_or_else(|| "Files".to_owned())
    });
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(
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
                    text(move || format!("{} files", title.get()))
                        .flex_1()
                        .px_2()
                        .number_of_lines(1)
                        .role(AccessibilityRole::Header)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD),
                ),
        )
        .child(shown(store, true))
}

/// One project's files: the folder open, or the file open in it.
fn browser(store: Store, project: ProjectSummary, in_page: bool) -> View {
    let id = project.id.clone();
    // The folder open (relative, "" for the project's own), whether
    // ignored files show, and the file open: a file brought from full
    // screen or into it opens again, in its folder.
    let start = store.files_start(&id);
    let in_folder = start
        .as_deref()
        .and_then(|p| p.rsplit_once('/'))
        .map(|(f, _)| f.to_owned())
        .unwrap_or_default();
    let folder = signal(in_folder);
    let ignored = signal(false);
    let open: Signal<Option<String>> = signal(start);
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
        effect(move || {
            let shown = open.get();
            untracked(|| {
                store.moved_off_files_start(shown.as_deref());
                match shown {
                    Some(path) => store.read_project_file(&id, &path),
                    None => store.close_project_file(),
                }
            });
        });
    }
    let name = project.name.clone();
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let shown = open.get();
        let name = name.clone();
        let id = id.clone();
        Some(untracked(move || match shown {
            Some(path) => file_view(store, id, path, open, in_page),
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
/// each pressable; the git branch checked out, whether git-ignored files
/// show, and Refresh.
fn crumbs(
    store: Store,
    id: String,
    name: String,
    folder: Signal<String>,
    ignored: Signal<bool>,
) -> View {
    // One line, never wrapping: the project, then the folders down to the
    // open one; deep, the middle folds into "…" (pressed, the deepest of
    // them). The open folder reads as the place, the others as the way back.
    let path = dynamic(move || {
        let current = folder.get();
        let name = name.clone();
        Some(untracked(move || {
            let parts: Vec<&str> = current.split('/').filter(|p| !p.is_empty()).collect();
            let up_to = |n: usize| parts[..n].join("/");
            let step = |label: String, target: String| {
                ui::btn(label, Variant::Ghost, Size::Sm, move || {
                    folder.set(target.clone())
                })
                .flex_shrink(1.0)
                .min_w(px(0.0))
            };
            let slash = || {
                text("/")
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT)
            };
            let mut row = view().flex_row().items_center().gap_1().min_w(px(0.0));
            let Some((here, above)) = parts.split_last() else {
                return row.child(place(name));
            };
            row = row.child(step(name, String::new())).child(slash());
            if above.len() > 1 {
                row = row
                    .child(step("…".to_owned(), up_to(above.len() - 1)))
                    .child(slash());
            }
            if let Some(parent) = above.last() {
                row = row
                    .child(step((*parent).to_owned(), up_to(above.len())))
                    .child(slash());
            }
            row.child(place((*here).to_owned()))
        }))
    });
    view()
        .gap_1()
        .px_2()
        .py_1()
        .border_b_1()
        .border_color(colors::LINE)
        .child(path)
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .flex_row()
                        .child(dynamic(move || {
                            let branch = store.folder().ready().and_then(|d| d.branch.clone());
                            // Named, so the code chip reads as the git
                            // branch checked out here, not a commit or a name.
                            Some(untracked(move || match branch {
                                Some(branch) => view()
                                    .flex_row()
                                    .items_center()
                                    .gap_1()
                                    .min_w(px(0.0))
                                    .child(
                                        text("Branch")
                                            .text_size(px(type_scale::XS))
                                            .text_color(colors::MUTED),
                                    )
                                    .child(ui::badge(branch, Tone::Neutral)),
                                None => view(),
                            }))
                        })),
                )
                // A switch, not a button whose words flip: it says which
                // files it adds (the ones .gitignore hides) and shows its state.
                .child(ui::toggle(
                    "Git-ignored",
                    move || ignored.get(),
                    move |next| ignored.set(next),
                ))
                .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Sm, move || {
                    store.list_project_files(&id, &folder.get_untracked(), ignored.get_untracked())
                })),
        )
}

/// The open folder in the path: where the owner is, not a way back.
fn place(name: String) -> View {
    view().flex_shrink(1.0).min_w(px(0.0)).px_2().child(
        text(name)
            .number_of_lines(1)
            .text_size(px(type_scale::SM))
            .font_weight(FontWeight::SEMIBOLD)
            .role(AccessibilityRole::Header),
    )
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
fn file_view(
    store: Store,
    id: String,
    path: String,
    open: Signal<Option<String>>,
    in_page: bool,
) -> View {
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
                    move || drop(set_clipboard(path.clone()))
                }))
                .child(full_screen(store, id, path.clone(), in_page)),
        )
        .child(dynamic(move || {
            let read = store.open_file();
            Some(untracked(move || file_body(store, read)))
        }))
}

/// Full screen from the details pane; back to it from the Files page when
/// the file came from there (a page opened from the sidebar has nowhere
/// to go back to but the sidebar).
fn full_screen(store: Store, id: String, path: String, in_page: bool) -> View {
    if !in_page {
        return ui::icon_btn(Glyph::Expand, "Full screen", Size::Sm, move || {
            store.files_full_screen(&id, &path)
        });
    }
    view().child(when(
        move || store.files_from_conversation(),
        move || {
            let path = path.clone();
            ui::icon_btn(Glyph::Shrink, "Exit full screen", Size::Sm, move || {
                store.files_leave_full_screen(Some(path.clone()))
            })
        },
    ))
}

fn file_body(store: Store, read: FileRead) -> View {
    match read {
        FileRead::NotYet | FileRead::Loading => {
            view().p_3().child(ui::working("Reading the file…"))
        }
        FileRead::Failed(reason) => view().p_3().child(ui::banner(reason, Tone::Danger)),
        FileRead::Ready(file) => match (file.text, file.withheld, file.link) {
            (Some(text), _, _) => {
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
            (None, Some(Withheld::Secret), _) => view().child(ui::empty(
                "Not shown",
                "This file holds secrets (keys, credentials or settings), so its text \
                 never leaves the computer it is on.",
            )),
            (None, Some(Withheld::Binary), Some(link)) => shown_file(store, link, file.size_bytes),
            (None, _, _) => view().child(ui::empty(
                "Not a text file",
                format!(
                    "{}; open it on the computer it is on.",
                    size_words(file.size_bytes)
                ),
            )),
        },
    }
}

/// A picture shown in place (tap for the viewer); a PDF, or a picture in
/// the terminal, opened outside the app by its link.
fn shown_file(store: Store, link: AttachmentRef, size_bytes: u64) -> View {
    let Some(url) = store.file_url(&link) else {
        return view().child(ui::empty("Not a text file", size_words(size_bytes)));
    };
    if link.is_image() && idiom() != Idiom::Terminal {
        let pixels = link.width.zip(link.height);
        let viewed = ui::Viewed {
            id: link.id.clone(),
            url: url.clone(),
            name: link.name.clone(),
            pixels,
        };
        // Zooms in place; Full screen for the viewer (share, save, mark up).
        return view()
            .flex_1()
            .min_h(px(0.0))
            .flex_col()
            .child(ui::zoomable_picture(
                url,
                format!("Picture: {}", link.name),
                pixels,
                // Its link may have expired: read the file again.
                move || store.project_file_failed(),
            ))
            .child(view().flex_row().justify_center().p_2().child(ui::btn(
                "Full screen",
                Variant::Ghost,
                Size::Sm,
                move || store.view_picture(viewed.clone()),
            )));
    }
    let what = if link.is_image() {
        "A picture"
    } else {
        "A PDF"
    };
    view()
        .flex_1()
        .items_center()
        .justify_center()
        .gap_3()
        .child(ui::empty(
            if link.is_image() { "Picture" } else { "PDF" },
            format!("{what}, {}.", size_words(size_bytes)),
        ))
        .child(ui::btn("Open", Variant::Secondary, Size::Md, move || {
            crate::link::open_outside(store, url.clone(), link.name.clone());
        }))
}

/// A changed file's one-letter mark and its tone.
pub(crate) fn status_letter(status: FileStatus) -> (&'static str, Tone) {
    match status {
        FileStatus::Added => ("A", Tone::Success),
        FileStatus::Untracked => ("U", Tone::Success),
        FileStatus::Deleted => ("D", Tone::Danger),
        FileStatus::Renamed => ("R", Tone::Info),
        FileStatus::Modified => ("M", Tone::Warning),
        FileStatus::Unknown => ("?", Tone::Neutral),
    }
}

/// How a file changed, in words.
pub(crate) fn status_word(status: FileStatus) -> &'static str {
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

/// The last part of a path (either separator).
pub(crate) fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
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
