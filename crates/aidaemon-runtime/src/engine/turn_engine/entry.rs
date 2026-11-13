//! The agent's front door: an inbound message with its attachments, and the
//! one-shot interpretation of a finished background command.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use aidaemon_channels::context::ChannelContext;
use aidaemon_events::conversation::MessageAttachment;
use aidaemon_tools::backend::BackendKind;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use aidaemon_work::scheduled_runs::{
    clear_scheduled_run_state, goal_has_scheduled_provenance, task_has_scheduled_provenance,
};
use serde_json::json;
use tokio::sync::mpsc;

use super::TurnEngine;
use crate::llm::side_calls::SideCall;

/// Whether an agent returning from a scheduled run clears the run's shared
/// budget and state. Background scheduled cycles are owned by the background
/// task lead, which keeps one shared budget alive across planning and every
/// executor; a direct or resumed run without a background task identity owns
/// its cleanup.
pub fn agent_return_owns_scheduled_run_cleanup(
    depth: usize,
    has_task_id: bool,
    is_root_scheduled_run: bool,
) -> bool {
    is_root_scheduled_run && (depth == 0 || !has_task_id)
}

/// `filename` with everything but ASCII letters, digits, `.`, `-` and `_`
/// replaced, for a path in the execution workspace.
fn safe_attachment_name(filename: &str) -> String {
    filename
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

impl TurnEngine<'_> {
    /// Run one inbound message: attachments are imported into a remote
    /// execution workspace first, and a root scheduled run's budget and state
    /// are cleared when it returns. The reply is the turn's finalized text.
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_inbound(
        &self,
        session_id: &str,
        user_text: &str,
        attachments: &[MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
        ingress_timing: Option<aidaemon_channels::ports::InboundMessageTiming>,
    ) -> anyhow::Result<String> {
        let scheduled_goal_to_clear = match self.goal_id() {
            Some(goal_id) => {
                let is_root_scheduled_run = match self.task_id() {
                    None => goal_has_scheduled_provenance(self.work(), goal_id, None).await,
                    task_id => task_has_scheduled_provenance(self.work(), task_id).await,
                };
                agent_return_owns_scheduled_run_cleanup(
                    self.depth(),
                    self.task_id().is_some(),
                    is_root_scheduled_run,
                )
                .then(|| goal_id.to_string())
            }
            None => None,
        };

        let backend = self.execution_backend_handle();
        let mut execution_user_text = user_text.to_string();
        if backend.kind() != BackendKind::Local {
            for attachment in attachments {
                let destination = backend
                    .workspace_root()
                    .join(".aidaemon")
                    .join("inbox")
                    .join(format!(
                        "{}-{}",
                        uuid::Uuid::new_v4(),
                        safe_attachment_name(&attachment.filename)
                    ));
                let imported = backend
                    .import_local_file(std::path::Path::new(&attachment.local_path), &destination)
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "Failed to import attachment {:?} into the {} execution workspace: {}",
                            attachment.filename,
                            backend.kind().as_str(),
                            error
                        )
                    })?;
                execution_user_text =
                    execution_user_text.replace(&attachment.local_path, imported.as_str());
            }
        }

        let reply = self
            .handle_message_impl(
                session_id,
                &execution_user_text,
                attachments,
                status_tx,
                user_role,
                channel_ctx,
                heartbeat,
                ingress_timing,
                false,
                None,
                None,
                None,
            )
            .await;

        if let Some(goal_id) = scheduled_goal_to_clear.as_deref() {
            if let Some(registry) = self.goal_tokens() {
                registry.clear_run_budget(goal_id).await;
            }
            clear_scheduled_run_state(self.work(), goal_id).await;
        }

        // The loop's response-finalization phase is the sole owner of all
        // user-facing transforms and persists that exact finalized text.
        reply
    }

    /// One-shot, tool-less interpretation of a finished background command's
    /// result: a bare value (a `wc -l` count, a path, a status line) becomes
    /// one short plain-language sentence without re-entering the agent loop.
    /// The side-call gateway sends no tools, so the model can only reply in
    /// text and cannot re-run the command. `None` on any failure or an empty
    /// reply, so the caller delivers the raw result instead.
    pub async fn interpret_background_result(&self, command: &str, output: &str) -> Option<String> {
        let system = "You translate the result of a shell command into ONE short, \
            plain-language sentence for a non-technical user. Say what the result \
            represents and flag any obvious caveat (for example, a raw text-match \
            count is not the same as a count of files). Never suggest running \
            anything, never use code formatting, and reply with the sentence only.";
        let user = format!(
            "Command that was run:\n{command}\n\nIts output:\n{output}\n\n\
             In one sentence, tell the user what this result means."
        );
        let messages = vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user}),
        ];

        match self
            .side_calls()
            .ask(
                SideCall::BackgroundInterpretation,
                &messages,
                &Default::default(),
                None,
            )
            .await
        {
            Ok(answer) => {
                let trimmed = answer.text.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            Err(e) => {
                tracing::warn!(error = %e, "Background-result interpretation LLM call failed");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_scheduled_agents_do_not_clear_the_shared_run_budget() {
        assert!(agent_return_owns_scheduled_run_cleanup(0, true, true));
        assert!(!agent_return_owns_scheduled_run_cleanup(1, true, true));
        assert!(!agent_return_owns_scheduled_run_cleanup(2, true, true));
        assert!(agent_return_owns_scheduled_run_cleanup(1, false, true));
        assert!(!agent_return_owns_scheduled_run_cleanup(0, true, false));
    }

    #[test]
    fn attachment_names_keep_only_safe_characters() {
        assert_eq!(safe_attachment_name("my photo (1).jpg"), "my_photo__1_.jpg");
    }
}
