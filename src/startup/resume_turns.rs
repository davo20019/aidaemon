//! Turns a process stop cut off are finished after the restart.
//!
//! A chat turn the daemon was answering when it stopped (a deploy, a crash,
//! a restart) never replied: the owner saw "Thinking…" and then nothing.
//! At boot each such turn gets one internal continuation in its chat, under
//! the same serializer and delivery as every background continuation. Its
//! message says, from the run's own record, which calls were running and
//! may or may not have taken effect; the turn start settles the old run
//! without running any of them again (`recovery::settle`).
//!
//! Which turns: top-level runs that never ended and were last checkpointed
//! after the previous boot, each still its session's latest run, in a chat
//! the hub can reach. A turn this pass continued that was cut off again is
//! not continued a second time (a turn that crashes the daemon would loop):
//! the chat is told instead.
//!
//! Background sub-agents the stop cut off are started again once, through
//! their chat (whose history holds the launch), told to pick up from what
//! the stopped run left. A job whose run was in the middle of a call that
//! reaches outside the machine (sending, publishing, deploying) or whose
//! effects are unknown is not started again: repeating it could do that
//! twice, so the chat is only told. A job started again that is cut off
//! again is not started a third time.

use std::collections::HashSet;
use std::sync::{Arc, Weak};

use crate::channels::SessionMap;
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_runtime::engine::recovery;
use aidaemon_runtime::spawn::{ChildAgentRuntime, RunsAnswer};
use aidaemon_store_sqlite::SqliteStore;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// The previous boot's mark and the chats it continued.
const BOOT_KEY: &str = "runtime.resume_cut_off_turns";

#[derive(Default, Serialize, Deserialize)]
struct BootRecord {
    /// The run store's clock when that boot started.
    boot_at: String,
    /// Chats that boot continued; cut off again, they are not continued.
    continued: Vec<String>,
    /// Chats whose background job that boot started again.
    #[serde(default)]
    restarted_jobs: Vec<String>,
}

/// The previous boot's mark, the chats it continued and the chats whose
/// background jobs it started again.
pub struct PreviousBoot {
    since: String,
    continued: Vec<String>,
    restarted_jobs: Vec<String>,
}

pub struct ResumeDeps {
    pub store: SqliteStore,
    pub state: Arc<dyn crate::traits::StateStore>,
    pub event_store: Arc<crate::events::EventStore>,
    pub sessions: SessionMap,
    pub hub: Weak<dyn OutboundRouter>,
    pub agent: Arc<dyn ChildAgentRuntime>,
    /// The background sub-agents the last stop cut off, as recorded at
    /// their launch (taken from the store at boot).
    pub interrupted_jobs: Vec<crate::traits::BackgroundSpawn>,
    /// The agent's tools, whose declared effects say whether a call the
    /// stop cut off is safe to repeat.
    pub tools: Vec<Arc<dyn aidaemon_tools::OutcomeTool>>,
}

/// Reads the previous boot's record and writes this boot's mark. Called
/// before anything this boot can checkpoint a run, so the next boot's list
/// holds only runs this process started or touched.
pub async fn mark_boot(
    store: &SqliteStore,
    state: &dyn crate::traits::StateStore,
) -> Option<PreviousBoot> {
    let previous = match state.get_setting(BOOT_KEY).await {
        Ok(Some(raw)) => serde_json::from_str::<BootRecord>(&raw).ok(),
        Ok(None) => None,
        Err(error) => {
            warn!(%error, "Could not read the previous boot's mark; no cut-off turn is resumed");
            None
        }
    };
    let boot_at = match store.clock().await {
        Ok(now) => now,
        Err(error) => {
            warn!(%error, "Could not read the run store's clock; no cut-off turn is resumed");
            return None;
        }
    };
    write_record(
        state,
        &BootRecord {
            boot_at,
            ..BootRecord::default()
        },
    )
    .await;
    previous.map(|record| PreviousBoot {
        since: record.boot_at,
        continued: record.continued,
        restarted_jobs: record.restarted_jobs,
    })
}

/// Read-modify-write of this boot's record, one at a time.
async fn update_record(
    state: &dyn crate::traits::StateStore,
    change: impl FnOnce(&mut BootRecord),
) {
    static RECORD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _held = RECORD.lock().await;
    let Ok(Some(raw)) = state.get_setting(BOOT_KEY).await else {
        return;
    };
    let Ok(mut record) = serde_json::from_str::<BootRecord>(&raw) else {
        return;
    };
    change(&mut record);
    write_record(state, &record).await;
}

async fn write_record(state: &dyn crate::traits::StateStore, record: &BootRecord) {
    let Ok(raw) = serde_json::to_string(record) else {
        return;
    };
    if let Err(error) = state.set_setting(BOOT_KEY, &raw).await {
        warn!(%error, "Could not record this boot's mark");
    }
}

/// Continues each chat turn the last stop cut off (see the module docs).
pub async fn resume_cut_off_turns(mut deps: ResumeDeps, previous: Option<PreviousBoot>) {
    let jobs = std::mem::take(&mut deps.interrupted_jobs);
    let Some(PreviousBoot {
        since,
        continued: continued_last_boot,
        restarted_jobs,
    }) = previous
    else {
        for job in &jobs {
            notify_job(&deps, job, JobFate::Lost).await;
        }
        return;
    };
    let mut job_notes = plan_jobs(&deps, jobs, &since, &restarted_jobs).await;
    let runs = match deps.store.unfinished_runs_since(&since).await {
        Ok(runs) => runs,
        Err(error) => {
            warn!(%error, "Could not list the turns the last stop cut off");
            return;
        }
    };
    let continued_last_boot: HashSet<String> = continued_last_boot.into_iter().collect();
    let Some(hub) = deps.hub.upgrade() else {
        return;
    };
    if !job_notes.is_empty() {
        let chats: Vec<String> = job_notes.keys().cloned().collect();
        update_record(&*deps.state, |record| record.restarted_jobs = chats).await;
    }
    let mut to_continue = Vec::new();
    for run in runs {
        let chat = run.session_id.clone();
        // Only chats the hub can answer in: a background or goal session has
        // its own recovery.
        if !deps.sessions.read().await.contains_key(&chat) {
            continue;
        }
        if continued_last_boot.contains(&chat) {
            info!(session_id = %chat, "A continued turn was cut off again; telling the chat");
            let note = "\u{26a0}\u{fe0f} The daemon restarted again while I was finishing your \
                        last request, so I stopped retrying it. Send it again, or say continue, \
                        to pick it up.";
            if let Err(error) = hub.send_text(&chat, note).await {
                warn!(session_id = %chat, %error, "Could not tell the chat its turn was cut off");
            }
            continue;
        }
        to_continue.push(run);
    }
    // A chat with both a cut-off turn and jobs to start again gets one
    // continuation, the jobs' note after the turn's.
    let mut continuations: Vec<(String, Option<String>, String)> = Vec::new();
    for run in &to_continue {
        let mut followup = recovery::resume_note(run);
        if let Some(note) = job_notes.remove(&run.session_id) {
            followup.push_str("\n\n");
            followup.push_str(&note);
        }
        let parent_task_id = deps
            .event_store
            .get_active_task(&run.session_id)
            .await
            .ok()
            .flatten()
            .and_then(|event| event.task_id);
        info!(session_id = %run.session_id, run_id = %run.run_id, "Continuing a turn the last stop cut off");
        continuations.push((run.session_id.clone(), parent_task_id, followup));
    }
    for (chat, note) in job_notes {
        info!(session_id = %chat, "Starting again a background job the last stop cut off");
        continuations.push((chat, None, note));
    }
    if continuations.is_empty() {
        return;
    }
    // Recorded before the turns run: if one takes the daemon down, the next
    // boot does not run it again.
    update_record(&*deps.state, |record| {
        record.continued = to_continue
            .iter()
            .map(|run| run.session_id.clone())
            .collect();
    })
    .await;
    for (chat, parent_task_id, followup) in continuations {
        let agent = deps.agent.clone();
        let hub = Arc::downgrade(&hub);
        let state = deps.state.clone();
        // Chats are continued side by side, each on its own task.
        tokio::spawn(async move {
            let answered = agent
                .answer_after_runs(RunsAnswer {
                    hub: Some(hub),
                    chat: chat.clone(),
                    parent_task_id,
                    followup,
                    clock: None,
                })
                .await;
            // The continuation ended on its own: a later cut-off in this
            // chat is a new turn's, and is continued again.
            update_record(&*state, |record| {
                record.continued.retain(|continued| continued != &chat);
            })
            .await;
            match answered {
                Ok(true) => info!(session_id = %chat, "Answered a turn the last stop cut off"),
                Ok(false) => warn!(session_id = %chat, "The continued turn gave no reply"),
                Err(error) => warn!(session_id = %chat, %error, "Continuing a cut-off turn failed"),
            }
        });
    }
}

/// What happens to a background job the last stop cut off.
enum JobFate {
    /// Started again through its chat.
    Restarted,
    /// Not started again: its run was in the middle of these calls, which
    /// reach outside the machine or whose effects are unknown.
    RiskyToRepeat(Vec<String>),
    /// Started again last boot and cut off again.
    CutOffAgain,
    /// Its chat cannot be reached, or the boot has nothing to compare with.
    Lost,
}

/// Tell the job's chat what became of it, through the notification queue
/// (held through quiet hours unless the owner was just there).
async fn notify_job(deps: &ResumeDeps, job: &crate::traits::BackgroundSpawn, fate: JobFate) {
    let mission = job.mission.trim();
    let message = match fate {
        JobFate::Restarted => format!(
            "\u{26a0}\u{fe0f} A restart stopped this background job before it finished: \
             {mission}\n\nStarting it again from where it stopped."
        ),
        JobFate::RiskyToRepeat(calls) => format!(
            "\u{26a0}\u{fe0f} A restart stopped this background job while it was running {}: \
             {mission}\n\nThat step may have gone through, so I did not start the job again. \
             Check what it did, then say continue to pick it up.",
            calls.join(", ")
        ),
        JobFate::CutOffAgain => format!(
            "\u{26a0}\u{fe0f} The daemon restarted again while this background job was running \
             after its restart, so I stopped restarting it: {mission}\n\nSay continue to \
             pick it up."
        ),
        JobFate::Lost => format!(
            "\u{26a0}\u{fe0f} A restart stopped this background job before it finished: \
             {mission}\n\nIt may have done part of the work. Check before running it again."
        ),
    };
    let queue = deps.state.clone() as Arc<dyn crate::traits::NotificationStore>;
    aidaemon_runtime::helper_run::delivery::enqueue_background_notification(
        Some(&queue),
        "global",
        &job.session_id,
        "failed",
        &message,
        "interrupted background sub-agent notifier",
        true,
    )
    .await;
}

/// The calls a cut-off run was in the middle of that are not safe to run
/// twice: a call whose declared effects reach a host outside this machine,
/// or a call whose tool declares nothing about what it does.
fn risky_in_flight_calls(
    run: &aidaemon_core::RunState,
    tools: &[Arc<dyn aidaemon_tools::OutcomeTool>],
) -> Vec<String> {
    let Some(in_flight) = &run.in_flight else {
        return Vec::new();
    };
    in_flight
        .calls
        .iter()
        .filter(|call| in_flight.started.contains(&call.id) && !in_flight.done.contains(&call.id))
        .filter(|call| {
            let Some(tool) = tools.iter().find(|tool| tool.name() == call.name) else {
                return true;
            };
            let effects = tool.call_effects(&call.arguments.to_string());
            let undeclared = effects.keys.is_empty() && !effects.complete;
            undeclared
                || effects.keys.iter().any(|key| {
                    key.class == aidaemon_core::EffectClass::Egress && !loopback_target(&key.target)
                })
        })
        .map(|call| format!("`{}`", call.name))
        .collect()
}

/// Whether an egress target (`host:port`, a bare host or a URL host) names
/// this machine.
fn loopback_target(target: &str) -> bool {
    // Machine format: `host:port`, `[v6]:port` or a bare host.
    let host = target
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| target.rsplit_once(':').map_or(target, |(host, _)| host));
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Each interrupted job's fate, the chat told for each one not started
/// again; returns the note that starts the rest again, per chat.
async fn plan_jobs(
    deps: &ResumeDeps,
    mut jobs: Vec<crate::traits::BackgroundSpawn>,
    since: &str,
    restarted_last_boot: &[String],
) -> std::collections::HashMap<String, String> {
    let mut notes: std::collections::HashMap<String, String> = Default::default();
    if jobs.is_empty() {
        return notes;
    }
    // The cut-off sub-agent runs, by the chat whose run started them, oldest
    // first: paired with the chat's jobs in launch order.
    let mut runs_by_chat: std::collections::HashMap<String, Vec<aidaemon_core::RunState>> =
        Default::default();
    match deps.store.unfinished_background_runs_since(since).await {
        Ok(runs) => {
            for run in runs {
                let Some(parent) = run.parent_run else {
                    continue;
                };
                if let Ok(Some(parent)) = deps.store.state(parent).await {
                    runs_by_chat.entry(parent.session_id).or_default().push(run);
                }
            }
        }
        Err(error) => warn!(%error, "Could not list the background runs the last stop cut off"),
    }
    jobs.sort_by(|a, b| a.started_at.cmp(&b.started_at));
    for job in jobs {
        let run = runs_by_chat
            .get_mut(&job.session_id)
            .and_then(|runs| (!runs.is_empty()).then(|| runs.remove(0)));
        let risky = run
            .as_ref()
            .map(|run| risky_in_flight_calls(run, &deps.tools))
            .unwrap_or_default();
        let reachable = deps.sessions.read().await.contains_key(&job.session_id);
        let fate = if !reachable {
            JobFate::Lost
        } else if restarted_last_boot.contains(&job.session_id) {
            JobFate::CutOffAgain
        } else if !risky.is_empty() {
            JobFate::RiskyToRepeat(risky)
        } else {
            JobFate::Restarted
        };
        if matches!(fate, JobFate::Restarted) {
            let note = restart_note(&job, run.as_ref());
            notes
                .entry(job.session_id.clone())
                .and_modify(|existing| {
                    existing.push_str("\n\n");
                    existing.push_str(&note);
                })
                .or_insert(note);
        }
        notify_job(deps, &job, fate).await;
    }
    notes
}

/// The internal message that starts a cut-off background job again in its
/// chat.
fn restart_note(
    job: &crate::traits::BackgroundSpawn,
    run: Option<&aidaemon_core::RunState>,
) -> String {
    let mut note = format!(
        "The daemon restarted while a background job you started in this conversation was \
         still running, and the restart stopped it: \"{}\". Start it again once, the same \
         way as its launch above, and add to its task that an earlier run was stopped by a \
         restart: it must first check what that run already did (files, installs, settings) \
         and continue from there instead of redoing it. Programs that run left going \
         (servers, tunnels, watchers) stopped with the restart and need starting again. \
         The owner was already told the job is being started again; reply only if \
         something needs their attention.",
        job.mission.trim()
    );
    if let Some(in_flight) = run.and_then(|run| run.in_flight.as_ref()) {
        let running: Vec<String> = in_flight
            .calls
            .iter()
            .filter(|call| {
                in_flight.started.contains(&call.id) && !in_flight.done.contains(&call.id)
            })
            .map(|call| format!("`{}`", call.name))
            .collect();
        if !running.is_empty() {
            note.push_str(&format!(
                " It was running {} when it stopped; that step may or may not have finished.",
                running.join(", ")
            ));
        }
    }
    note
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStore;
    use crate::events::EventStoreForTests as _;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use aidaemon_runtime::spawn::{ChildAgentRequest, ChildAgentRun};
    use aidaemon_tools::semantics::AgentRole;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Answers(
        StdMutex<Vec<RunsAnswer>>,
        Arc<aidaemon_runtime::run_state::RunStates>,
    );

    #[async_trait]
    impl ChildAgentRuntime for Answers {
        fn depth(&self) -> usize {
            0
        }
        fn max_depth(&self) -> usize {
            3
        }
        fn role(&self) -> AgentRole {
            AgentRole::Orchestrator
        }
        fn specialist_descriptions(&self) -> Vec<(String, String)> {
            Vec::new()
        }
        async fn run_child(&self, _request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun> {
            anyhow::bail!("not used")
        }
        async fn validate_executor_task_for_spawn(
            &self,
            _task_id: &str,
            _expected_goal_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn salvage_executor_task_outcome(
            &self,
            _task_id: &str,
            _timeout_secs: u64,
        ) -> Option<aidaemon_runtime::spawn::SalvagedTaskOutcome> {
            None
        }
        async fn deliver_background_child_result(
            &self,
            _router: Option<&Weak<dyn OutboundRouter>>,
            _parent_session_id: &str,
            _text: &str,
            _trust: aidaemon_core::SourceTrust,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
        fn helper_runs(&self) -> Arc<aidaemon_runtime::run_state::RunStates> {
            self.1.clone()
        }
        async fn mark_executor_task_timeout(&self, _task_id: &str, _timeout_secs: u64) {}
        async fn answer_after_runs(&self, answer: RunsAnswer) -> anyhow::Result<bool> {
            self.0.lock().unwrap().push(answer);
            Ok(true)
        }
    }

    #[derive(Default)]
    struct Sent(StdMutex<Vec<(String, String)>>);

    #[async_trait]
    impl OutboundRouter for Sent {
        async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .push((session_id.to_string(), text.to_string()));
            Ok(())
        }
        async fn send_text_tracked(&self, _: &str, _: &str) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn edit_text(&self, _: &str, _: &str, _: &str) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn send_media(&self, _: &str, _: &crate::types::MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media_strict(
            &self,
            _: &str,
            _: &crate::types::MediaMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn request_inline_approval(
            &self,
            _: &str,
            _: &str,
            _: crate::tools::command_risk::RiskLevel,
            _: &[String],
            _: crate::tools::command_risk::PermissionMode,
        ) -> anyhow::Result<crate::types::ApprovalResponse> {
            anyhow::bail!("no approvals here")
        }
        async fn register_session_route(&self, _: &str, _: &str) -> bool {
            false
        }
        async fn unregister_session_route(&self, _: &str, _: &str) {}
    }

    async fn settings() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let path = db_file.path().to_str().unwrap().to_string();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&path, 100, None, embedding)
                .await
                .unwrap(),
        );
        aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
        state
    }

    async fn boot(
        store: &SqliteStore,
        state: &Arc<SqliteStateStore>,
        sessions: &SessionMap,
        agent: &Arc<Answers>,
        hub: &Arc<Sent>,
    ) {
        boot_with_jobs(store, state, sessions, agent, hub, Vec::new()).await;
    }

    async fn boot_with_jobs(
        store: &SqliteStore,
        state: &Arc<SqliteStateStore>,
        sessions: &SessionMap,
        agent: &Arc<Answers>,
        hub: &Arc<Sent>,
        interrupted_jobs: Vec<crate::traits::BackgroundSpawn>,
    ) {
        let dyn_state = state.clone() as Arc<dyn crate::traits::StateStore>;
        let previous = mark_boot(store, &*dyn_state).await;
        let router: Arc<dyn OutboundRouter> = hub.clone();
        resume_cut_off_turns(
            ResumeDeps {
                store: SqliteStore::from_pool(store_pool(store)).await.unwrap(),
                state: dyn_state,
                event_store: Arc::new(EventStore::new(state.pool()).await.unwrap()),
                sessions: sessions.clone(),
                hub: Arc::downgrade(&router),
                agent: agent.clone(),
                interrupted_jobs,
                tools: Vec::new(),
            },
            previous,
        )
        .await;
        // The continuations run on their own tasks.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    fn store_pool(store: &SqliteStore) -> sqlx::SqlitePool {
        store.pool().clone()
    }

    #[tokio::test]
    async fn a_cut_off_chat_turn_is_continued_once_and_then_the_chat_is_told() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let state = settings().await;
        let chat = "app:synthetic-chat-1";
        let sessions: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::from([(
            chat.to_string(),
            "app".to_string(),
        )])));
        let agent = Arc::new(Answers::default());
        let hub = Arc::new(Sent::default());

        // First boot: nothing to compare against, nothing continued.
        boot(&store, &state, &sessions, &agent, &hub).await;
        assert!(agent.0.lock().unwrap().is_empty());

        // A turn in a chat and one in a session no chat routes to, both cut off.
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                chat,
                "Draw a logo.",
            ))
            .await
            .unwrap();
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                "background:synthetic",
                "Helper work.",
            ))
            .await
            .unwrap();
        boot(&store, &state, &sessions, &agent, &hub).await;
        {
            let answers = agent.0.lock().unwrap();
            assert_eq!(answers.len(), 1, "only the chat is continued");
            assert_eq!(answers[0].chat, chat);
            assert!(answers[0].followup.contains("reply it is still owed"));
        }

        // The continuation was cut off before it ended: the next boot does
        // not run it again, and the chat hears why.
        let mut record: BootRecord =
            serde_json::from_str(&state.get_setting(BOOT_KEY).await.unwrap().unwrap()).unwrap();
        record.continued = vec![chat.to_string()];
        write_record(&*state, &record).await;
        store
            .upsert_state(&aidaemon_core::RunState::new(
                "aidaemon",
                chat,
                "continuation",
            ))
            .await
            .unwrap();
        boot(&store, &state, &sessions, &agent, &hub).await;
        assert_eq!(agent.0.lock().unwrap().len(), 1, "not continued twice");
        let sent = hub.0.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].1.contains("stopped retrying"), "{}", sent[0].1);
    }

    fn job(chat: &str, mission: &str, minute: u32) -> crate::traits::BackgroundSpawn {
        crate::traits::BackgroundSpawn {
            id: format!("background_spawn:{mission}"),
            session_id: chat.to_string(),
            mission: mission.to_string(),
            started_at: format!("2026-11-21T05:{minute:02}:00+00:00"),
        }
    }

    /// The cut-off sub-agent run of a job started from `chat` by a turn
    /// that has ended, mid `call` when `call` is given.
    async fn cut_off_child(store: &SqliteStore, chat: &str, call: Option<&str>) {
        let parent = aidaemon_core::RunState::new("aidaemon", chat, "Build the site.");
        store.upsert_state(&parent).await.unwrap();
        store
            .append(
                parent.run_id,
                &aidaemon_core::Event::Finished {
                    status: aidaemon_core::RunStatus::Completed,
                    usage: Default::default(),
                    turns: 1,
                    elapsed_ms: 0,
                },
            )
            .await
            .unwrap();
        let mut child =
            aidaemon_core::RunState::new("aidaemon", "specialist:executor:synthetic", "Build it.")
                .with_parent(parent.run_id);
        if let Some(name) = call {
            child.in_flight = Some(aidaemon_core::InFlight {
                calls: vec![aidaemon_core::ToolCall::new(
                    "call-1",
                    name,
                    serde_json::json!({}),
                )],
                started: vec!["call-1".to_string()],
                done: Vec::new(),
            });
        }
        store.upsert_state(&child).await.unwrap();
    }

    /// Live 2026-11-20: a deploy restart stopped the owner's background
    /// site build, and nothing started it again. A job is started again
    /// once through its chat; one mid a call whose effects are not known
    /// to be safe to repeat, or in a chat no channel reaches, is only told.
    #[tokio::test]
    async fn a_cut_off_background_job_is_started_again_once_unless_risky_to_repeat() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let state = settings().await;
        let safe = "app:synthetic-chat-1";
        let risky = "app:synthetic-chat-2";
        let sessions: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::from([
            (safe.to_string(), "app".to_string()),
            (risky.to_string(), "app".to_string()),
        ])));
        let agent = Arc::new(Answers::default());
        let hub = Arc::new(Sent::default());
        boot(&store, &state, &sessions, &agent, &hub).await;

        cut_off_child(&store, safe, None).await;
        cut_off_child(&store, risky, Some("synthetic_publish")).await;
        boot_with_jobs(
            &store,
            &state,
            &sessions,
            &agent,
            &hub,
            vec![
                job(safe, "Build the synthetic site", 1),
                job(risky, "Publish the synthetic post", 2),
                job("goal:synthetic", "Tidy the synthetic notes", 3),
            ],
        )
        .await;
        {
            let answers = agent.0.lock().unwrap();
            assert_eq!(answers.len(), 1, "only the safe job is started again");
            assert_eq!(answers[0].chat, safe);
            assert!(
                answers[0].followup.contains("Build the synthetic site"),
                "{}",
                answers[0].followup
            );
        }
        let notices = state.get_pending_notifications(10).await.unwrap();
        let notice = |chat: &str| {
            notices
                .iter()
                .find(|entry| entry.session_id == chat)
                .map(|entry| entry.message.clone())
                .unwrap_or_default()
        };
        assert!(
            notice(safe).contains("Starting it again"),
            "{}",
            notice(safe)
        );
        assert!(
            notice(risky).contains("did not start the job again"),
            "{}",
            notice(risky)
        );
        assert!(notice(risky).contains("`synthetic_publish`"));
        assert!(
            notice("goal:synthetic").contains("Check before running it again"),
            "{}",
            notice("goal:synthetic")
        );

        // The job started again is cut off again: told, not started a
        // third time.
        cut_off_child(&store, safe, None).await;
        boot_with_jobs(
            &store,
            &state,
            &sessions,
            &agent,
            &hub,
            vec![job(safe, "Build the synthetic site", 4)],
        )
        .await;
        assert_eq!(agent.0.lock().unwrap().len(), 1, "not started a third time");
        let notices = state.get_pending_notifications(10).await.unwrap();
        assert!(
            notices
                .iter()
                .any(|entry| entry.message.contains("stopped restarting it")),
            "{notices:#?}"
        );
    }
}
