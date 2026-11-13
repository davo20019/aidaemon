//! The screen: sessions on the left, the chat in the middle (with the
//! oldest waiting approval as a card under it), running work on the right,
//! the message box and a status line below. Keys turn into protocol
//! commands; nothing here talks to the network.

use aidaemon_client_protocol::{
    Approval, ApprovalDecision, AskReason, CardAction, Command, Impact, Risk,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::state::{Line, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Sessions,
    Chat,
    Approval,
    Tasks,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Send(Command),
    Quit,
}

#[derive(Debug)]
pub struct App {
    pub state: State,
    pub focus: Focus,
    /// The session the chat shows; `None`: the next message starts one.
    pub selected: Option<String>,
    pub task_index: usize,
    pub input: String,
    pub connected: bool,
    /// Sessions whose history was asked for.
    fetched: std::collections::HashSet<String>,
    /// Waiting for the session the next message starts.
    pub awaiting_new: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: State::default(),
            focus: Focus::Chat,
            selected: None,
            task_index: 0,
            input: String::new(),
            connected: false,
            fetched: Default::default(),
            awaiting_new: false,
        }
    }
}

fn key_of(decision: ApprovalDecision) -> (char, &'static str) {
    match decision {
        ApprovalDecision::ApproveOnce => ('y', "allow once"),
        ApprovalDecision::ApproveSession => ('s', "allow this session"),
        ApprovalDecision::ApproveAlways => ('a', "always allow"),
        ApprovalDecision::Deny => ('n', "deny"),
    }
}

impl App {
    /// After a frame: follow a session this client just started.
    pub fn after_frame(&mut self, frame: &aidaemon_client_protocol::ServerFrame) {
        use aidaemon_client_protocol::{CommandResult, ServerFrame};
        if let ServerFrame::Ack {
            result:
                CommandResult::MessageAccepted {
                    session_id,
                    new_session: true,
                },
            ..
        } = frame
        {
            if self.awaiting_new {
                self.awaiting_new = false;
                self.selected = Some(session_id.clone());
                self.fetched.insert(session_id.clone());
            }
        }
        if self.focus == Focus::Approval && self.state.next_approval().is_none() {
            self.focus = Focus::Chat;
        }
    }

    fn select_session(&mut self, delta: isize) -> Option<Action> {
        let order = &self.state.order;
        if order.is_empty() {
            return None;
        }
        let at = self
            .selected
            .as_ref()
            .and_then(|id| order.iter().position(|o| o == id));
        let next = match at {
            None => 0,
            Some(at) => (at as isize + delta).clamp(0, order.len() as isize - 1) as usize,
        };
        let id = order[next].clone();
        self.selected = Some(id.clone());
        // History is fetched once per session.
        if !self.fetched.insert(id.clone()) {
            return None;
        }
        Some(Action::Send(Command::GetHistory {
            session_id: id,
            before: None,
            limit: 100,
        }))
    }

    fn cycle_focus(&mut self) {
        let card = self.state.next_approval().is_some();
        self.focus = match self.focus {
            Focus::Sessions => Focus::Chat,
            Focus::Chat if card => Focus::Approval,
            Focus::Chat | Focus::Approval => Focus::Tasks,
            Focus::Tasks => Focus::Sessions,
        };
    }

    /// What a key does.
    pub fn key(&mut self, key: KeyEvent) -> Option<Action> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => return Some(Action::Quit),
            KeyCode::Char('n') if ctrl => {
                self.selected = None;
                self.focus = Focus::Chat;
                return None;
            }
            KeyCode::Tab => {
                self.cycle_focus();
                return None;
            }
            KeyCode::Esc => {
                self.focus = Focus::Chat;
                return None;
            }
            _ => {}
        }
        match self.focus {
            Focus::Sessions => match key.code {
                KeyCode::Up => self.select_session(-1),
                KeyCode::Down => self.select_session(1),
                KeyCode::Enter => {
                    self.focus = Focus::Chat;
                    None
                }
                _ => None,
            },
            Focus::Tasks => {
                let running = self.state.running();
                match key.code {
                    KeyCode::Up => {
                        self.task_index = self.task_index.saturating_sub(1);
                        None
                    }
                    KeyCode::Down => {
                        self.task_index =
                            (self.task_index + 1).min(running.len().saturating_sub(1));
                        None
                    }
                    KeyCode::Char('c') | KeyCode::Delete => {
                        running.get(self.task_index).map(|task| {
                            Action::Send(Command::CancelTask {
                                task_id: task.id.clone(),
                            })
                        })
                    }
                    _ => None,
                }
            }
            Focus::Approval => {
                let approval = self.state.next_approval()?;
                let KeyCode::Char(pressed) = key.code else {
                    return None;
                };
                approval
                    .options
                    .iter()
                    .find(|decision| key_of(**decision).0 == pressed)
                    .map(|decision| {
                        Action::Send(Command::ResolveApproval {
                            approval_id: approval.id.clone(),
                            decision: *decision,
                        })
                    })
            }
            Focus::Chat => match key.code {
                KeyCode::Enter => self.submit(),
                KeyCode::Backspace => {
                    self.input.pop();
                    None
                }
                KeyCode::Char(c) => {
                    self.input.push(c);
                    None
                }
                _ => None,
            },
        }
    }

    fn submit(&mut self) -> Option<Action> {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return None;
        }
        // `/stop` stops the session's work; other slash commands run as
        // they would in the chat.
        if let Some(command) = text.strip_prefix('/') {
            let session_id = self.selected.clone()?;
            self.input.clear();
            let (name, args) = command.split_once(' ').unwrap_or((command, ""));
            return Some(Action::Send(match name {
                "stop" => Command::StopTurn { session_id },
                _ => Command::RunSlash {
                    session_id,
                    command: name.to_string(),
                    args: args.to_string(),
                },
            }));
        }
        let writable = self.selected.as_ref().is_none_or(|id| {
            self.state
                .sessions
                .get(id)
                .and_then(|v| v.summary.as_ref())
                .is_none_or(|s| s.writable)
        });
        if !writable {
            self.state.notice = Some(
                "That session belongs to another chat: reply there, or Ctrl-N for a new one."
                    .into(),
            );
            return None;
        }
        self.input.clear();
        self.awaiting_new = self.selected.is_none();
        Some(Action::Send(Command::SendMessage {
            session_id: self.selected.clone(),
            text,
            attachments: Vec::new(),
        }))
    }

    pub fn draw(&self, frame: &mut Frame) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(frame.area());
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(24),
                Constraint::Min(30),
                Constraint::Percentage(26),
            ])
            .split(rows[0]);
        self.draw_sessions(frame, columns[0]);
        match self.state.next_approval() {
            Some(approval) => {
                let middle = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(5), Constraint::Length(11)])
                    .split(columns[1]);
                self.draw_chat(frame, middle[0]);
                self.draw_card(frame, middle[1], approval);
            }
            None => self.draw_chat(frame, columns[1]),
        }
        self.draw_tasks(frame, columns[2]);
        let input = Paragraph::new(self.input.as_str()).block(self.block(
            Focus::Chat,
            if self.selected.is_some() {
                "Message"
            } else {
                "Message (new session)"
            },
        ));
        frame.render_widget(input, rows[1]);
        let status = match (&self.state.notice, self.connected) {
            (Some(notice), true) => notice.clone(),
            (_, false) => "Reconnecting...".to_string(),
            (None, true) => "Tab: switch pane  Ctrl-N: new session  /stop  Ctrl-C: quit".into(),
        };
        frame.render_widget(
            Paragraph::new(status).style(Style::default().fg(Color::DarkGray)),
            rows[2],
        );
    }

    fn block(&self, focus: Focus, title: &str) -> Block<'static> {
        let style = if self.focus == focus {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        };
        Block::default()
            .borders(Borders::ALL)
            .border_style(style)
            .title(title.to_string())
    }

    fn draw_sessions(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .state
            .order
            .iter()
            .map(|id| {
                let busy = self
                    .state
                    .sessions
                    .get(id)
                    .is_some_and(|v| v.live.is_some());
                ListItem::new(format!(
                    "{}{}",
                    if busy { "● " } else { "  " },
                    self.state.title(id)
                ))
            })
            .collect();
        let mut list_state = ListState::default();
        list_state.select(
            self.selected
                .as_ref()
                .and_then(|id| self.state.order.iter().position(|o| o == id)),
        );
        let list = List::new(items)
            .block(self.block(Focus::Sessions, "Sessions"))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, area, &mut list_state);
    }

    fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<TextLine> = Vec::new();
        if let Some(view) = self
            .selected
            .as_ref()
            .and_then(|id| self.state.sessions.get(id))
        {
            for line in &view.lines {
                match line {
                    Line::User { text, origin } => {
                        let who = if origin.is_empty() || origin == "app" {
                            "You".to_string()
                        } else {
                            format!("You ({origin})")
                        };
                        lines.push(TextLine::from(Span::styled(
                            format!("{who}: "),
                            Style::default().fg(Color::Green),
                        )));
                        lines.extend(text.lines().map(|l| TextLine::from(l.to_string())));
                    }
                    Line::Assistant { text } => {
                        lines.push(TextLine::from(Span::styled(
                            "Assistant:",
                            Style::default().fg(Color::Magenta),
                        )));
                        lines.extend(text.lines().map(|l| TextLine::from(l.to_string())));
                    }
                    Line::Note { text } => lines.push(TextLine::from(Span::styled(
                        text.clone(),
                        Style::default().fg(Color::Yellow),
                    ))),
                }
                lines.push(TextLine::from(""));
            }
            if let Some(live) = &view.live {
                if !live.streamed.is_empty() {
                    lines.extend(live.streamed.lines().map(|l| TextLine::from(l.to_string())));
                }
                lines.push(TextLine::from(Span::styled(
                    live.status.clone().unwrap_or_else(|| "Working...".into()),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        } else {
            lines.push(TextLine::from("Type a message to start a new session."));
        }
        // Keep the newest lines in view.
        let height = area.height.saturating_sub(2) as usize;
        let skip = lines.len().saturating_sub(height);
        let title = self
            .selected
            .as_ref()
            .map(|id| self.state.title(id))
            .unwrap_or_else(|| "New session".into());
        let chat = Paragraph::new(lines.split_off(skip))
            .block(self.block(Focus::Chat, &title))
            .wrap(Wrap { trim: false });
        frame.render_widget(chat, area);
    }

    fn draw_card(&self, frame: &mut Frame, area: Rect, approval: &Approval) {
        let risk_style = match approval.risk {
            Risk::Low => Style::default().fg(Color::Blue),
            Risk::Medium => Style::default().fg(Color::Yellow),
            Risk::High | Risk::Critical => Style::default().fg(Color::Red),
        };
        let mut lines = vec![TextLine::from(Span::styled(
            approval.summary.clone(),
            risk_style.add_modifier(Modifier::BOLD),
        ))];
        if let Some(card) = &approval.card {
            if let Some(impact) = &card.impact {
                let warning = match impact {
                    Impact::Deletes { targets } if !targets.is_empty() => {
                        format!("⚠ Deletes {}", targets.join(", "))
                    }
                    Impact::Deletes { .. } => "⚠ Deletes files".into(),
                    Impact::Sends => "⚠ Sends something to another service".into(),
                    Impact::Spends => "⚠ Spends money".into(),
                    Impact::WritesOutsideWorkArea => "⚠ Writes outside the work area".into(),
                    Impact::Unknown => String::new(),
                };
                if !warning.is_empty() {
                    lines.push(TextLine::from(Span::styled(warning, risk_style)));
                }
            }
            match &card.reason {
                Some(AskReason::ReadOutsideContent { check_says }) => {
                    lines.push(TextLine::from(check_says.clone().unwrap_or_else(|| {
                        "It read outside content, so this step needs your OK.".into()
                    })))
                }
                Some(AskReason::AfterADecline) => lines.push(TextLine::from(
                    "You said no to an earlier step, so this one asks first.",
                )),
                _ => {}
            }
            if let CardAction::Command { looks_inside, .. } = &card.action {
                if !looks_inside.is_empty() {
                    lines.push(TextLine::from(format!(
                        "Only looks inside: {}",
                        looks_inside.join(", ")
                    )));
                }
            }
        }
        for warning in &approval.warnings {
            lines.push(TextLine::from(format!("• {warning}")));
        }
        if let Some(detail) = &approval.detail {
            lines.push(TextLine::from(Span::styled(
                detail.lines().next().unwrap_or_default().to_string(),
                Style::default().fg(Color::DarkGray),
            )));
        }
        let keys: Vec<String> = approval
            .options
            .iter()
            .map(|decision| {
                let (key, label) = key_of(*decision);
                format!("[{key}] {label}")
            })
            .collect();
        lines.push(TextLine::from(keys.join("   ")));
        let mut title = format!("Approval needed ({:?} risk)", approval.risk);
        if let Some(channel) = &approval.channel {
            title.push_str(&format!(", also in {channel}"));
        }
        if self.state.approvals.len() > 1 {
            title.push_str(&format!(", {} waiting", self.state.approvals.len()));
        }
        let card = Paragraph::new(lines)
            .block(self.block(Focus::Approval, &title))
            .wrap(Wrap { trim: true });
        frame.render_widget(card, area);
    }

    fn draw_tasks(&self, frame: &mut Frame, area: Rect) {
        let running = self.state.running();
        let items: Vec<ListItem> = running
            .iter()
            .map(|task| {
                ListItem::new(format!(
                    "#{} {} ({})",
                    task.number,
                    task.description,
                    self.state.title(&task.session_id)
                ))
            })
            .collect();
        let mut list_state = ListState::default();
        if !running.is_empty() {
            list_state.select(Some(self.task_index.min(running.len() - 1)));
        }
        let list = List::new(items)
            .block(self.block(Focus::Tasks, "Running (c: cancel)"))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, area, &mut list_state);
    }
}
