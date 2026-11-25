//! The terminal on screen, attached to its conversation's shell (protocol
//! 1.42). One terminal is attached at a time: the one in the details pane.
//! Moving to another conversation detaches it and attaches that
//! conversation's own; each shell keeps running on the daemon meanwhile,
//! and its output so far comes back when it is attached again.
//!
//! The shell's output never enters a signal: it goes straight to the
//! terminal view's sink, so a busy shell does not rebuild pages.

use std::rc::Rc;

use aidaemon_client_core::shells::{self, ShellEvent};
use aidaemon_client_protocol::{ShellInfo, ShellTransport};

use super::Store;

/// A shell socket as the link opened it.
pub type ShellLink = super::SocketLink;

/// Where a terminal view takes its shell's events.
pub type ShellSink = Rc<dyn Fn(ShellEvent)>;

/// The terminal attached now.
pub(super) struct Attached {
    session_id: String,
    sink: ShellSink,
    link: Option<ShellLink>,
    epoch: u64,
}

/// The socket address of a shell: `transport.url` (a path) on the gateway
/// at `http_base`.
fn socket_url(http_base: &str, transport: &ShellTransport) -> String {
    aidaemon_client_core::surfaces::path_socket_url(http_base, &transport.url)
}

impl Store {
    /// A terminal for conversation `session_id` is laid out at `cols` by
    /// `rows` cells: attach it to the conversation's shell (started in its
    /// folder when none runs). Its events go to `sink`.
    pub fn attach_shell(&self, session_id: &str, cols: u32, rows: u32, sink: ShellSink) {
        self.detach_shell();
        let epoch = self.shell_epoch.with(|e| *e) + 1;
        self.shell_epoch.replace(epoch);
        self.shell.replace(Some(Attached {
            session_id: session_id.to_owned(),
            sink,
            link: None,
            epoch,
        }));
        let (cols, rows) = shells::cells(cols, rows);
        self.act(|c| c.open_shell(session_id, cols, rows));
    }

    /// The terminal went away (another conversation, another tab).
    pub fn detach_shell(&self) {
        if let Some(attached) = self.shell.replace(None)
            && let Some(link) = attached.link
        {
            (link.close)();
        }
    }

    /// The terminal of conversation `session_id` went away: detach it,
    /// unless another conversation's has taken its place already.
    pub fn detach_shell_of(&self, session_id: &str) {
        let mine = self
            .shell
            .with(|a| a.as_ref().is_some_and(|a| a.session_id == session_id));
        if mine {
            self.detach_shell();
        }
    }

    /// The daemon opened the shell: connect the attached terminal to it.
    pub(super) fn connect_shell(&self, session_id: &str, transport: &ShellTransport) {
        let epoch = self.shell.with(|attached| {
            attached
                .as_ref()
                .filter(|a| a.session_id == session_id && a.link.is_none())
                .map(|a| a.epoch)
        });
        let Some(epoch) = epoch else {
            return;
        };
        let base = self.gateway_base.with(|b| b.clone());
        let url = socket_url(&base, transport);
        crate::link::open_shell(*self, epoch, url, transport.ticket.clone());
    }

    /// The link opened the socket of opening `epoch`.
    pub fn shell_attached(&self, epoch: u64, link: ShellLink) {
        let stale = self.shell.with(|attached| match attached {
            Some(a) if a.epoch == epoch => {
                if let Some(old) = a.link.replace(link) {
                    (old.close)();
                }
                None
            }
            _ => Some(link),
        });
        if let Some(link) = stale {
            (link.close)();
        }
    }

    /// Something came on the socket of opening `epoch`.
    pub fn shell_event(&self, epoch: u64, event: ShellEvent) {
        let sink = self.shell.with(|attached| {
            let a = attached.as_mut().filter(|a| a.epoch == epoch)?;
            if matches!(event, ShellEvent::Closed(_)) {
                a.link = None;
            }
            Some(a.sink.clone())
        });
        if let Some(sink) = sink {
            sink(event);
        }
    }

    fn shell_send(&self, text: String) {
        self.shell.with(|attached| {
            if let Some(link) = attached.as_ref().and_then(|a| a.link.as_ref()) {
                (link.send)(text);
            }
        });
    }

    /// What the owner typed into the attached terminal.
    pub fn shell_input(&self, data: &str) {
        self.shell_send(shells::input(data));
    }

    /// The attached terminal's size changed.
    pub fn shell_resize(&self, cols: u32, rows: u32) {
        let (cols, rows) = shells::cells(cols, rows);
        self.shell_send(shells::resize(cols, rows));
    }

    /// End conversation `session_id`'s shell.
    pub fn close_shell(&self, session_id: &str) {
        self.act(|c| c.close_shell(session_id));
    }

    /// Conversation `session_id`'s shell, running or ended (tracked).
    pub fn shell_of(&self, session_id: &str) -> Option<ShellInfo> {
        self.read(|c| c.state.shells.get(session_id).cloned())
    }

    /// Why conversation `session_id`'s shell could not open (tracked).
    pub fn shell_error(&self, session_id: &str) -> Option<String> {
        self.read(|c| c.shell_errors.get(session_id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shell_socket_is_on_the_gateway_s_origin() {
        let transport = ShellTransport {
            url: "/client/v1/shells/sh_1/ws".into(),
            ticket: "tkt".into(),
        };
        assert_eq!(
            socket_url("https://daemon.example.invalid/", &transport),
            "wss://daemon.example.invalid/client/v1/shells/sh_1/ws"
        );
        assert_eq!(
            socket_url("http://127.0.0.1:8788", &transport),
            "ws://127.0.0.1:8788/client/v1/shells/sh_1/ws"
        );
    }
}
