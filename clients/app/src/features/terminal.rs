//! The details' Terminal tab: the conversation's shell, which the owner and
//! the agent share. It starts in the conversation's project folder the
//! first time it is shown and keeps running on the daemon while the owner
//! is in another conversation; coming back shows what it wrote meanwhile.

use std::cell::Cell;
use std::rc::Rc;

use aidaemon_client_core::shells::ShellEvent;
use nsh_base::{Terminal, TerminalApi, TerminalEvent};
use nsh_core::prelude::*;

use crate::store::{ShellSink, Store};
use crate::ui::{self, Size, Variant, btn, colors, type_scale};

pub fn terminal_tab(store: Store) -> View {
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let session = store.selected()?;
        Some(untracked(move || shell_view(store, session)))
    }))
}

fn shell_view(store: Store, session: String) -> View {
    // A terminal in a terminal adds nothing: the TUI build points at the
    // others.
    if idiom() == nsh_core::platform::Idiom::Terminal {
        return view().p_4().child(
            text("The conversation's terminal is in the desktop and web apps.")
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
        );
    }
    let term = component_ref::<Terminal>();
    // The shell ended (its exit code), or the socket closed (why).
    let ended = signal(None::<Option<u32>>);
    let dropped = signal(None::<String>);
    let attached = Rc::new(Cell::new(false));
    let size = Rc::new(Cell::new((0u32, 0u32)));

    let sink: ShellSink = Rc::new(move |event| {
        let api = term.api::<TerminalApi>();
        match event {
            ShellEvent::Ready => dropped.set(None),
            ShellEvent::Output(data) => {
                if let Some(api) = api {
                    api.write(data);
                }
            }
            ShellEvent::Reset => {
                if let Some(api) = api {
                    api.reset();
                }
            }
            ShellEvent::Ended(code) => ended.set(Some(code)),
            ShellEvent::Closed(why) => dropped.set(Some(why)),
        }
    });

    // Attach once laid out (the daemon starts it at this size); later
    // sizes resize it.
    let attach = {
        let (session, sink, attached, size) =
            (session.clone(), sink.clone(), attached.clone(), size.clone());
        move || {
            let (cols, rows) = size.get();
            if cols < 2 || rows < 1 {
                return;
            }
            attached.set(true);
            ended.set(None);
            dropped.set(None);
            store.attach_shell(&session, cols, rows, sink.clone());
        }
    };
    let on_event = {
        let (attach, attached, size) = (attach.clone(), attached.clone(), size.clone());
        move |event: &TerminalEvent| match event {
            TerminalEvent::Data(data) => store.shell_input(data),
            TerminalEvent::Resize(cols, rows) => {
                size.set((*cols, *rows));
                if attached.get() {
                    store.shell_resize(*cols, *rows);
                } else {
                    attach();
                }
            }
            _ => {}
        }
    };
    {
        let session = session.clone();
        cleanup(move || store.detach_shell_of(&session));
    }

    let folder = {
        let session = session.clone();
        move || {
            store
                .shell_of(&session)
                .map(|shell| {
                    if shell.title.is_empty() {
                        shell.folder
                    } else {
                        shell.title
                    }
                })
                .unwrap_or_else(|| "Starting the shell…".to_owned())
        }
    };
    let error = {
        let session = session.clone();
        move || store.shell_error(&session)
    };
    let restart = {
        let attach = attach.clone();
        move || {
            if let Some(api) = term.api::<TerminalApi>() {
                api.reset();
            }
            attach();
        }
    };
    let close = {
        let session = session.clone();
        move || store.close_shell(&session)
    };
    view()
        .flex_1()
        .min_h(px(0.0))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .px_3()
                .py_1()
                .child(
                    text(folder)
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                )
                .child(when(
                    move || ended.get().is_some() || dropped.get().is_some(),
                    move || {
                        let restart = restart.clone();
                        btn("Restart", Variant::Ghost, Size::Sm, restart)
                    },
                ))
                .child(when(
                    move || ended.get().is_none() && dropped.get().is_none(),
                    move || {
                        let close = close.clone();
                        btn("End", Variant::Ghost, Size::Sm, close)
                    },
                )),
        )
        .child(dynamic(move || {
            let line = match (error(), ended.get(), dropped.get()) {
                (Some(why), ..) => why,
                (None, Some(Some(code)), _) => format!("The shell exited ({code})."),
                (None, Some(None), _) => "The shell exited.".to_owned(),
                (None, None, Some(why)) => format!("Disconnected: {why}."),
                (None, None, None) => return None,
            };
            Some(untracked(move || {
                text(line)
                    .px_3()
                    .pb_1()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::MUTED)
                    .into_element()
            }))
        }))
        .child(ui::interactive_terminal(
            term,
            on_event,
            "The conversation's terminal",
        ))
}
