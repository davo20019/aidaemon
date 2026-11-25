//! `shared_terminal` (feature `tool`): the agent's side of a conversation's
//! shell, the terminal the owner has open in their app.
//!
//! - `read` returns what the shell shows (its screen and lines above it).
//!   What is on it was written by whatever ran there, so it is untrusted.
//! - `type` types a line into it. The shell is the owner's, unconfined, so
//!   every line is the owner's to allow through the agent kernel: the card
//!   shows the exact text, Allow runs it once, a decline binds the run.
//!   The call returns what the shell wrote once it went quiet (or after a
//!   per-call wait; the program keeps running and `read` shows the rest).
//!
//! The agent never opens a shell: the owner does, from the Terminal tab.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::owner_answers::OwnerAnsweredCalls;
use aidaemon_tools::semantics::{
    ApprovalCall, ApprovalResponse, OwnerApprovalNeed, ToolCallContext, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolOutcomeStatus, ToolSemantics,
    TrustRoots, UserRole,
};

use crate::{ShellInfo, ShellRegistry};

/// The effect every typed line declares: a decline binds the run.
const TYPE_EFFECT: &str = "shared_terminal";
/// The longest line the agent may type at once.
const MAX_TYPED_CHARS: usize = 2000;
/// Lines above the screen a read returns by default, and at most.
const DEFAULT_ABOVE: usize = 40;
const MAX_ABOVE: usize = 400;
/// The shell is done answering once it writes nothing for this long.
const QUIET: Duration = Duration::from_millis(1500);
/// The most a typed line waits for the shell (per call: the program keeps
/// running after).
const MAX_WAIT: Duration = Duration::from_secs(30);
/// The most output a typed line returns.
const MAX_OUTPUT_CHARS: usize = 16 * 1024;

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum Args {
    Read {
        #[serde(default)]
        lines_above: Option<usize>,
    },
    Type {
        text: String,
        /// Press Enter after the text (default true).
        #[serde(default = "yes")]
        enter: bool,
    },
}

fn yes() -> bool {
    true
}

/// `shared_terminal`.
pub struct SharedTerminalTool {
    shells: ShellRegistry,
    owner_answers: OwnerAnsweredCalls,
}

impl SharedTerminalTool {
    pub fn new(shells: ShellRegistry) -> Self {
        Self {
            shells,
            owner_answers: OwnerAnsweredCalls::default(),
        }
    }

    fn running_shell(&self, session: &str) -> anyhow::Result<ShellInfo> {
        let shell = self.shells.for_session(session).context(
            "This conversation has no terminal open. The owner opens it from the Terminal tab of the conversation's details in their app; until then use the terminal tool.",
        )?;
        anyhow::ensure!(
            shell.running(),
            "This conversation's terminal has exited. The owner can restart it from the Terminal tab."
        );
        Ok(shell)
    }

    /// The subject an allowed line is kept under until its call runs: the
    /// exact text, so an answer covers only what the card showed.
    fn subject(text: &str, enter: bool) -> String {
        format!("{TYPE_EFFECT} {enter} {text}")
    }

    fn screen_json(&self, shell: &ShellInfo, above: usize) -> anyhow::Result<Value> {
        let screen = self.shells.screen(&shell.id, above)?;
        Ok(json!({
            "folder": shell.folder.display().to_string(),
            "title": shell.title,
            "lines_above": screen.above.join("\n"),
            "screen": screen.screen.join("\n"),
            "cursor": {"row": screen.cursor.0, "column": screen.cursor.1},
            "full_screen_program": screen.full_screen,
        }))
    }
}

/// What a shell writes until it goes quiet for [`QUIET`], at most
/// [`MAX_WAIT`] or [`MAX_OUTPUT_CHARS`].
async fn collect_until_quiet(output: &mut broadcast::Receiver<Arc<str>>) -> (String, bool) {
    let mut seen = String::new();
    let deadline = tokio::time::Instant::now() + MAX_WAIT;
    loop {
        let wait = QUIET.min(deadline.saturating_duration_since(tokio::time::Instant::now()));
        if wait.is_zero() {
            return (seen, false);
        }
        match tokio::time::timeout(wait, output.recv()).await {
            Ok(Ok(text)) => {
                seen.push_str(&text);
                if seen.len() >= MAX_OUTPUT_CHARS {
                    return (seen, false);
                }
            }
            Ok(Err(broadcast::error::RecvError::Lagged(_))) => {}
            Ok(Err(broadcast::error::RecvError::Closed)) => return (seen, true),
            // Quiet (or the wait is over).
            Err(_) => return (seen, tokio::time::Instant::now() < deadline),
        }
    }
}

/// Terminal output as plain text: escape sequences and carriage-return
/// overwrites dropped, kept to its last [`MAX_OUTPUT_CHARS`].
fn plain(raw: &str) -> String {
    let mut parser = vt100::Parser::new(
        std::num::NonZeroU16::new(500).unwrap_or(std::num::NonZeroU16::MIN),
        std::num::NonZeroU16::new(240).unwrap_or(std::num::NonZeroU16::MIN),
        0,
    );
    // Lines end as a terminal's do, so each starts at the left edge.
    parser.process(raw.replace('\n', "\r\n").as_bytes());
    let text = parser.screen().contents();
    let text = text.trim_end();
    let cut = text.len().saturating_sub(MAX_OUTPUT_CHARS);
    let start = text.ceil_char_boundary(cut);
    text[start..].to_string()
}

#[async_trait]
impl Tool for SharedTerminalTool {
    fn name(&self) -> &str {
        "shared_terminal"
    }

    fn description(&self) -> &str {
        "Read the terminal the owner has open in this conversation, or type a command into it once the owner allows."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "shared_terminal",
            "description": "The terminal the owner has open in this conversation's details in their app (a shell started in the conversation's project folder, which you share with the owner). `read` shows what is on it: use it when the owner refers to something in their terminal (an error, a command's output). `type` types a line into it, for something the owner wants run in their own shell (its environment, a program they are running); the owner allows each line first, and the call returns what the shell wrote. For your own commands use the terminal tool, not this one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "type"],
                        "description": "read: what the terminal shows. type: type `text` into it."
                    },
                    "lines_above": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_ABOVE,
                        "description": "read: lines above the screen to include (default 40)."
                    },
                    "text": {
                        "type": "string",
                        "description": "type: the text to type, e.g. a command line."
                    },
                    "enter": {
                        "type": "boolean",
                        "description": "type: press Enter after the text (default true). False to answer a prompt in parts."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn record_owner_answer(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
        response: Option<ApprovalResponse>,
    ) {
        let Some(session) = facts.session() else {
            return;
        };
        let Ok(Args::Type { text, enter }) = serde_json::from_value::<Args>(call.arguments.clone())
        else {
            return;
        };
        if matches!(
            response,
            Some(
                ApprovalResponse::AllowOnce
                    | ApprovalResponse::AllowSession
                    | ApprovalResponse::AllowAlways
            )
        ) {
            self.owner_answers
                .approve(session, &Self::subject(&text, enter));
        }
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args = serde_json::from_str(arguments)
            .context("shared_terminal arguments must be valid JSON with an action")?;
        anyhow::ensure!(
            facts.user_role == Some(UserRole::Owner),
            "Only the owner's own conversations have a shared terminal"
        );
        let session = facts
            .session()
            .context("shared_terminal needs a conversation")?
            .to_string();
        let shell = self.running_shell(&session)?;
        let output = match args {
            Args::Read { lines_above } => {
                let above = lines_above.unwrap_or(DEFAULT_ABOVE).min(MAX_ABOVE);
                self.screen_json(&shell, above)?
            }
            Args::Type { text, enter } => {
                anyhow::ensure!(
                    self.owner_answers
                        .take(&session, &Self::subject(&text, enter)),
                    "Typing into the owner's terminal needs their answer through the agent kernel"
                );
                let mut attached = self.shells.attach(&shell.id)?;
                let mut typed = text.clone();
                if enter {
                    typed.push('\r');
                }
                self.shells.write(&shell.id, &typed)?;
                let (raw, quiet) = collect_until_quiet(&mut attached.output).await;
                json!({
                    "typed": text,
                    "output": plain(&raw),
                    "interpretation": if quiet {
                        "The shell went quiet: this is what it wrote. Read the terminal for the prompt or the rest of the screen."
                    } else {
                        "The program is still writing (or waiting): this is what it wrote so far. It keeps running in the owner's terminal; read it again later."
                    },
                })
            }
        };
        Ok(ToolCallOutcome {
            output: serde_json::to_string_pretty(&output)?,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

#[async_trait]
impl ToolSemantics for SharedTerminalTool {
    /// Each typed line is the owner's to allow; their answer comes back to
    /// the tool, never as a session grant.
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            keeps_own_grants: true,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::EXTERNAL_HIGH_IMPACT
    }

    /// A typed line is asked every time, showing its exact text; a read
    /// asks nothing. A call that could not type (no shell, not the owner's
    /// chat) shows no card; execution says why.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if facts.user_role != Some(UserRole::Owner) {
            return OwnerApprovalNeed::NotNeeded;
        }
        let Some(session) = facts.session() else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let Ok(Args::Type { text, enter }) = serde_json::from_value::<Args>(call.arguments.clone())
        else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if text.chars().count() > MAX_TYPED_CHARS || self.running_shell(session).is_err() {
            return OwnerApprovalNeed::NotNeeded;
        }
        let shown: String = text.chars().filter(|c| !c.is_control()).collect();
        let summary = if enter {
            format!("Run in this conversation's terminal: {shown}")
        } else {
            format!("Type into this conversation's terminal (no Enter): {shown}")
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            effects: vec![aidaemon_core::EffectKey::named(
                aidaemon_core::EffectClass::Call,
                TYPE_EFFECT,
            )],
            summary,
            subject: Self::subject(&text, enter),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["It runs in your own shell, outside the agent's sandbox.".to_string()],
            card: None,
        })
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match serde_json::from_str::<Args>(arguments) {
            Ok(Args::Read { .. }) => aidaemon_core::CallEffects::none(),
            _ => aidaemon_core::CallEffects {
                keys: vec![aidaemon_core::EffectKey::named(
                    aidaemon_core::EffectClass::Call,
                    TYPE_EFFECT,
                )],
                complete: true,
                understood: true,
            },
        }
    }

    /// What a terminal shows was written by whatever ran there.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
    }

    /// A read looks; a typed line runs a program in the owner's shell,
    /// which may do anything there.
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        match serde_json::from_str::<Args>(arguments) {
            Ok(Args::Read { .. }) => ToolCallSemantics::observation(),
            _ => ToolCallSemantics::observation_and_mutation_with(
                aidaemon_tools::semantics::ToolMutationEffects::PROCESS_STATE,
            ),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::ShellsConfig;

    fn owner_facts(session: &str) -> ToolCallContext {
        let mut facts = aidaemon_tools::testing::call_facts(session);
        facts.user_role = Some(UserRole::Owner);
        facts
    }

    #[test]
    fn output_reads_as_plain_text() {
        assert_eq!(
            // A progress line overwrites itself after a carriage return.
            plain("\u{1b}[31mred\u{1b}[0m\n 10%\r100%"),
            "red\n100%"
        );
    }

    #[tokio::test]
    async fn reading_needs_no_card_and_typing_needs_the_owner() {
        let folder = tempfile::tempdir().unwrap();
        let shells = ShellRegistry::new(ShellsConfig::default());
        let tool = SharedTerminalTool::new(shells.clone());
        let facts = owner_facts("app:s1");

        // No shell yet: says where the owner opens it, asks nothing.
        let read = json!({"action": "read"});
        let error = tool.execute(&read.to_string(), &facts).await.unwrap_err();
        assert!(error.to_string().contains("Terminal tab"), "{error}");

        let shell = shells.open("app:s1", folder.path(), 80, 24).unwrap();
        // As in the owner's app: the shell is up (its prompt shown) first.
        let mut started = shells.attach(&shell.id).unwrap();
        let mut seen = started.backlog.clone();
        while seen.is_empty() {
            seen.push_str(&started.output.recv().await.unwrap());
        }
        let _ = collect_until_quiet(&mut started.output).await;
        let typed = json!({"action": "type", "text": "printf 'sum:%s\\n' $((40+2))"});
        let call = ApprovalCall {
            arguments: &typed,
            as_sent: &typed,
            refused: false,
        };
        assert!(matches!(
            tool.owner_approval_request(&call, &facts).await,
            OwnerApprovalNeed::Ask(ref request) if request.summary.contains("$((40+2))")
        ));
        assert!(matches!(
            tool.owner_approval_request(
                &ApprovalCall {
                    arguments: &read,
                    as_sent: &read,
                    refused: false,
                },
                &facts
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        ));
        // Not allowed: refused at execution.
        assert!(tool.execute(&typed.to_string(), &facts).await.is_err());

        tool.record_owner_answer(&call, &facts, Some(ApprovalResponse::AllowOnce))
            .await;
        let out = tool.execute(&typed.to_string(), &facts).await.unwrap();
        assert!(out.output.contains("sum:42"), "{}", out.output);
        // The answer covered that one call.
        assert!(tool.execute(&typed.to_string(), &facts).await.is_err());

        let screen = tool.execute(&read.to_string(), &facts).await.unwrap();
        assert!(screen.output.contains("sum:42"), "{}", screen.output);
    }
}
