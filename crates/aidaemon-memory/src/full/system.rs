//! [`FullMemory`]: the full memory system as an
//! `aidaemon_policies::MemorySystem`.

use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Hook, Provider, Service, Tool};
use aidaemon_policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn};
use aidaemon_tools::services::SettingsStore;
use async_trait::async_trait;

use super::jobs::{JobCall, JobsHost, MemoryJobs};
use super::recall::{
    FullRecall, RecallHost, RecallSettings, RecallSource, RecallTurn, SpeculativeRecall,
};
use super::tools;
use super::transcript::{TranscriptLog, TranscriptRecorder};
use crate::render::RecallGoal;
use crate::stores::MemoryStores;

/// The host's facts about one turn, given to [`FullMemory`] as the turn's
/// [`HostTurn`](aidaemon_policies::HostTurn). Without them the system works
/// from the [`MemoryTurn`] alone: the `"owner"` scope is the owner's private
/// chat, any other a private group, with no goals and people off.
#[derive(Clone)]
pub struct FullTurn {
    pub recall: RecallTurn,
    pub host: Arc<dyn RecallHost>,
    /// Further run hooks the host adds with this memory (its run-end
    /// writer).
    pub hooks: Vec<Arc<dyn Hook>>,
}

/// The full memory system over the memory stores: per turn, the memory
/// block injected into the request (see [`FullRecall`]) and, on a turn that
/// may read or write memory, its tools.
pub struct FullMemory {
    recall: Arc<RecallSource>,
    tools: Vec<Arc<dyn Tool>>,
    /// Its background jobs, when it runs them.
    jobs: Option<Arc<MemoryJobs>>,
    /// Records owner turns for the jobs, when the host keeps no log.
    recorder: Option<Arc<TranscriptRecorder>>,
    /// What [`FullMemory::open`] opened, for [`FullMemory::with_background`].
    opened: Option<Opened>,
    /// Recall started when a session's latest message arrived
    /// ([`FullMemory::speculate`]), waiting for that turn.
    speculative: std::sync::Mutex<std::collections::HashMap<String, SpeculativeRecall>>,
    /// The store its tools write, for the per-turn `remember_fact` of a turn
    /// that works in a project ([`FullMemory::with_project_notes`]).
    store: Option<Arc<dyn MemoryStores>>,
    /// Project memory, when on.
    project_notes: Option<Arc<crate::project::ProjectNotes>>,
}

struct Opened {
    pool: sqlx::SqlitePool,
    store: Arc<dyn MemoryStores>,
    embedder: Arc<dyn crate::Embedder>,
}

/// One provider and model for every background call.
struct OneModel {
    provider: Arc<dyn Provider>,
    model: String,
}

#[async_trait]
impl JobsHost for OneModel {
    fn model(&self, _call: JobCall) -> (Arc<dyn Provider>, String) {
        (self.provider.clone(), self.model.clone())
    }
}

impl FullMemory {
    /// Recall only. A host that offers the memory tools through its own
    /// dispatcher (see [`tools::outcome_tools`]) starts here.
    pub fn new(recall: Arc<RecallSource>) -> Self {
        Self {
            recall,
            tools: Vec::new(),
            jobs: None,
            recorder: None,
            opened: None,
            speculative: std::sync::Mutex::default(),
            store: None,
            project_notes: None,
        }
    }

    /// Keep what a turn that works in a project learns about it in `notes`
    /// (see [`crate::project`]): recall shows the project's notes, and the
    /// turn's `remember_fact` writes project facts there instead of in the
    /// fact store. A turn without a project is unaffected. For a system
    /// from [`FullMemory::open`] or [`FullMemory::over`]; one built from a
    /// shared [`RecallSource`] gives the notes to that source with
    /// [`RecallSource::with_project_notes`] instead.
    pub fn with_project_notes(mut self, notes: Arc<crate::project::ProjectNotes>) -> Self {
        match Arc::get_mut(&mut self.recall) {
            Some(source) => source.set_project_notes(notes.clone()),
            None => tracing::warn!(
                "FullMemory::with_project_notes: the recall source is shared; \
                 give it the notes with RecallSource::with_project_notes"
            ),
        }
        self.project_notes = Some(notes);
        self
    }

    /// Starts recall for a message as it arrives, before its turn's memory
    /// policy is settled, so the turn's first model call finds the block
    /// ready. Nothing is written and nothing reaches a turn unless that
    /// turn may read memory and recalls for exactly these inputs; a newer
    /// message in the session replaces (and stops) an unclaimed one.
    pub fn speculate(&self, turn: RecallTurn, host: Arc<dyn RecallHost>) {
        let Some(speculative) = SpeculativeRecall::start(self.recall.clone(), host, turn) else {
            return;
        };
        if let Ok(mut pending) = self.speculative.lock() {
            pending.insert(speculative.session_id().to_string(), speculative);
        }
    }

    /// The session's speculative recall, when it is `turn`'s block. One
    /// that is not is dropped, which stops it.
    fn claim_speculative(&self, turn: &RecallTurn) -> Option<SpeculativeRecall> {
        let speculative = self.speculative.lock().ok()?.remove(&turn.session_id)?;
        let serves = speculative.serves(turn);
        tracing::info!(
            session_id = %turn.session_id,
            adopted = serves,
            "Speculative recall claimed"
        );
        serves.then_some(speculative)
    }

    /// Run `jobs` as this system's [`services`](MemorySystem::services).
    pub fn with_jobs(mut self, jobs: Arc<MemoryJobs>) -> Self {
        self.jobs = Some(jobs);
        self
    }

    /// For a system from [`FullMemory::open`]: record the owner's turns in
    /// the memory database ([`TranscriptLog`]) and run the background jobs
    /// over them, their model calls on `provider` with `model`. A system
    /// built another way is returned as it is.
    pub async fn with_background(
        mut self,
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let Some(opened) = &self.opened else {
            return Ok(self);
        };
        let log = Arc::new(TranscriptLog::open(opened.pool.clone()).await?);
        self.recorder = Some(Arc::new(log.recorder()));
        let host = Arc::new(OneModel {
            provider,
            model: model.into(),
        });
        let jobs = MemoryJobs::new(opened.store.clone(), log, host, opened.embedder.clone());
        self.jobs = Some(Arc::new(jobs));
        Ok(self)
    }

    /// Its background jobs, when it runs them.
    pub fn jobs(&self) -> Option<&Arc<MemoryJobs>> {
        self.jobs.as_ref()
    }

    /// The whole system over `store`: recall with `settings`, and the four
    /// memory tools ([`tools::kernel_tools`]). `host_settings` holds the
    /// `people_enabled` switch.
    pub fn over(
        store: Arc<dyn MemoryStores>,
        settings: RecallSettings,
        host_settings: Arc<dyn SettingsStore>,
    ) -> Self {
        let mut system = Self::new(Arc::new(RecallSource::new(store.clone(), settings)))
            .with_tools(tools::kernel_tools(store.clone(), host_settings));
        system.store = Some(store);
        system
    }

    /// The whole system in a SQLite file (created and migrated if needed),
    /// or in a private in-memory database when `path` is `None`. Embeddings
    /// come from `FastEmbedder` with the `fastembed` feature, else from
    /// [`HashEmbedder`](crate::HashEmbedder); people intelligence's switch is
    /// kept in memory ([`tools::LocalSettings`]).
    pub async fn open(path: Option<&std::path::Path>) -> anyhow::Result<Self> {
        use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
        let pool = match path {
            Some(path) => {
                let options = SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true)
                    .journal_mode(SqliteJournalMode::Wal)
                    .busy_timeout(std::time::Duration::from_secs(10));
                SqlitePoolOptions::new()
                    .max_connections(4)
                    .connect_with(options)
                    .await?
            }
            None => {
                // One connection that never closes: the database lives in it.
                SqlitePoolOptions::new()
                    .max_connections(1)
                    .min_connections(1)
                    .idle_timeout(None)
                    .max_lifetime(None)
                    .connect("sqlite::memory:")
                    .await?
            }
        };
        crate::sqlite::migrate(&pool).await?;
        #[cfg(feature = "fastembed")]
        let embedder: Arc<dyn crate::Embedder> = Arc::new(crate::FastEmbedder::new());
        #[cfg(not(feature = "fastembed"))]
        let embedder: Arc<dyn crate::Embedder> = Arc::new(crate::HashEmbedder::new(384));
        let store: Arc<dyn MemoryStores> = Arc::new(crate::sqlite::SqliteMemory::new(
            pool.clone(),
            embedder.clone(),
        ));
        let mut system = Self::over(
            store.clone(),
            RecallSettings {
                max_facts: DEFAULT_MAX_FACTS,
                assistant_name: None,
            },
            Arc::new(tools::LocalSettings::default()),
        );
        system.opened = Some(Opened {
            pool,
            store,
            embedder,
        });
        Ok(system)
    }

    /// Offer `tools` on every turn that may read or write memory.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
}

/// How many facts recall renders at most, by default (the daemon's
/// default too).
pub const DEFAULT_MAX_FACTS: usize = 20;

/// A host with nothing of its own: no goals, people intelligence off.
struct NoHost;

#[async_trait]
impl RecallHost for NoHost {
    async fn active_goals(&self) -> anyhow::Result<Vec<RecallGoal>> {
        Ok(Vec::new())
    }

    async fn people_enabled(&self) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// What recall knows about a turn given only the [`MemoryTurn`].
fn plain_turn(turn: &MemoryTurn) -> RecallTurn {
    let owner = turn.scope == "owner";
    RecallTurn {
        session_id: turn.session_id.clone(),
        task_id: String::new(),
        user_text: None,
        owner,
        personal: owner,
        visibility: if owner {
            ChannelVisibility::Private
        } else {
            ChannelVisibility::PrivateGroup
        },
        channel_id: None,
        sender_id: None,
        allowed: turn.read,
        depth: usize::from(turn.explicit_writes_only),
        project: turn.project.clone(),
    }
}

#[async_trait]
impl MemorySystem for FullMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let (recall_turn, host, extra): (RecallTurn, Arc<dyn RecallHost>, Vec<Arc<dyn Hook>>) =
            match turn.host.get::<FullTurn>() {
                Some(full) => (full.recall.clone(), full.host.clone(), full.hooks.clone()),
                None => (plain_turn(turn), Arc::new(NoHost), Vec::new()),
            };
        // Recall starts now, beside the rest of the turn's setup; the
        // injection hook awaits it at the first model call.
        let recall = match self.claim_speculative(&recall_turn) {
            Some(speculative) => {
                FullRecall::new(self.recall.clone(), host, recall_turn).adopt(speculative)
            }
            None => FullRecall::new(self.recall.clone(), host, recall_turn).prefetch(),
        };
        let mut hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(MemoryInjection::new(recall))];
        hooks.extend(extra);
        // Only the owner's own turns teach the jobs.
        if let Some(recorder) = &self.recorder
            && turn.write
            && turn.scope == "owner"
        {
            hooks.push(recorder.clone());
        }
        let mut tools = if turn.read || turn.write {
            self.tools.clone()
        } else {
            Vec::new()
        };
        // A turn in a project gets a `remember_fact` that knows it.
        if let (Some(project), Some(notes), Some(store)) =
            (&turn.project, &self.project_notes, &self.store)
            && let Some(slot) = tools
                .iter_mut()
                .find(|tool| tool.spec().name == "remember_fact")
        {
            let tool = tools::RememberFactTool::new(store.clone())
                .with_project_notes(notes.clone())
                .for_project(Some(project.clone()));
            *slot = Arc::new(aidaemon_tools::KernelAdapter::new(Arc::new(tool)));
        }
        MemoryParts { hooks, tools }
    }

    fn services(&self) -> Vec<Arc<dyn Service>> {
        self.jobs
            .as_ref()
            .map(MemoryJobs::services)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::SqliteMemory;

    #[tokio::test]
    async fn an_owner_turn_recalls_its_facts_without_host_facts() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let store = SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)));
        crate::stores::FactStore::upsert_fact(
            &store,
            "preference",
            "favorite_color",
            "teal",
            "user",
            None,
            crate::records::FactPrivacy::Global,
        )
        .await
        .unwrap();
        let source = Arc::new(RecallSource::new(
            Arc::new(store),
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
        ));
        let turn = MemoryTurn {
            session_id: "test:synthetic-user".into(),
            project: None,
            scope: "owner".into(),
            read: true,
            write: true,
            explicit_writes_only: false,
            learning: Default::default(),
            host: Default::default(),
        };
        let parts = FullMemory::new(source.clone()).turn(&turn).await;
        assert_eq!(parts.hooks.len(), 1);
        assert!(parts.tools.is_empty());
        let recall = FullRecall::new(source, Arc::new(NoHost), plain_turn(&turn));
        let note = aidaemon_policies::Recall::render(
            &recall,
            "test:synthetic-user",
            "what is my favorite color?",
            5,
        )
        .await
        .unwrap();
        assert!(note.contains("teal"), "{note}");
    }

    /// A turn in a project sees its notes after the owner's own sections;
    /// a turn without one, and a turn in another project, renders exactly
    /// what it rendered before; and the turn's `remember_fact` writes notes.
    #[tokio::test]
    async fn a_project_turn_sees_its_notes_and_keeps_the_rest() {
        use crate::project::{NoteKind, NoteSource, NoteWrite, ProjectNotes};
        let dir = tempfile::tempdir().unwrap();
        let notes = Arc::new(ProjectNotes::new(dir.path()));
        let system = FullMemory::open(None)
            .await
            .unwrap()
            .with_project_notes(notes.clone());
        let store = system.store.clone().unwrap();
        crate::stores::FactStore::upsert_fact(
            store.as_ref(),
            "preference",
            "favorite_color",
            "teal",
            "user",
            None,
            crate::records::FactPrivacy::Global,
        )
        .await
        .unwrap();
        let project = aidaemon_core::ProjectRef::new("project-a", "Acme Site", None).unwrap();
        let other = aidaemon_core::ProjectRef::new("project-b", "Other", None).unwrap();
        notes
            .write(
                &project,
                &NoteWrite {
                    kind: NoteKind::Fact,
                    key: "test_command".into(),
                    category: Some("technical".into()),
                    description: None,
                    body: "Tests run with make test.".into(),
                    source: NoteSource::Agent,
                },
            )
            .unwrap();
        let render = |project: Option<aidaemon_core::ProjectRef>| {
            let source = system.recall.clone();
            async move {
                let turn = MemoryTurn {
                    project,
                    ..MemoryTurn::owner("test:synthetic-user")
                };
                let recall = FullRecall::new(source, Arc::new(NoHost), plain_turn(&turn));
                recall.render_block("what is my favorite color?").await
            }
        };
        let plain = render(None).await;
        let in_other = render(Some(other)).await;
        let in_project = render(Some(project.clone())).await;
        assert!(plain.contains("teal"), "{plain}");
        assert!(!plain.contains("Project Memory"), "{plain}");
        assert_eq!(plain, in_other);
        assert!(in_project.starts_with(&plain), "{in_project}");
        assert!(
            in_project.contains("## Project Memory (Acme Site)"),
            "{in_project}"
        );
        assert!(in_project.contains("fact-test_command.md"), "{in_project}");

        // The project turn's remember_fact keeps the project's facts apart.
        let turn = MemoryTurn {
            project: Some(project.clone()),
            ..MemoryTurn::owner("test:synthetic-user")
        };
        let parts = system.turn(&turn).await;
        let tool = parts
            .tools
            .iter()
            .find(|t| t.spec().name == "remember_fact")
            .unwrap()
            .clone();
        let ctx = aidaemon_core::testing::tool_context();
        tool.call(
            serde_json::json!({"category": "technical", "key": "db", "value": "Postgres 16"}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(notes.read(&project, NoteKind::Fact, "db").is_some());
        let facts = crate::stores::FactStore::get_facts(store.as_ref(), None)
            .await
            .unwrap();
        assert!(facts.iter().all(|f| f.key != "db"), "{facts:?}");
    }

    /// With conversations placed in projects, an episode is recalled only on
    /// turns in its conversation's project: one project's history stays out
    /// of other projects and of plain chats, and plain chats keep theirs.
    /// Without placement every episode is recalled, as before.
    #[tokio::test]
    async fn episodes_are_recalled_only_in_their_conversations_project() {
        struct Placed;
        #[async_trait]
        impl crate::project::SessionProjects for Placed {
            async fn project_of(&self, session_id: &str) -> Option<String> {
                match session_id {
                    "app:in-a" => Some("project-a".into()),
                    "app:in-b" => Some("project-b".into()),
                    _ => None,
                }
            }
        }
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let store = SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)));
        for (session, summary) in [
            ("app:in-a", "Alpha history: deployed the Acme site."),
            ("app:in-b", "Beta history: tuned the ledger tests."),
            (
                "app:plain",
                "Gamma history: planned a birthday dinner for Mia.",
            ),
        ] {
            let now = chrono::Utc::now();
            crate::stores::EpisodeStore::record_session_episode(
                &store,
                &crate::episodes::SessionEpisode {
                    session_id: session.into(),
                    summary: summary.into(),
                    topics: vec![],
                    emotional_tone: "neutral".into(),
                    outcome: "done".into(),
                    importance: 0.5,
                    message_count: 4,
                    start_time: now,
                    end_time: now,
                    channel_id: None,
                },
            )
            .await
            .unwrap();
        }
        let store: Arc<dyn MemoryStores> = Arc::new(store);
        let settings = RecallSettings {
            max_facts: 10,
            assistant_name: None,
        };
        let render = |source: Arc<RecallSource>, project: Option<&'static str>| async move {
            let turn = MemoryTurn {
                project: project.map(|id| aidaemon_core::ProjectRef::new(id, id, None).unwrap()),
                ..MemoryTurn::owner("test:synthetic-user")
            };
            FullRecall::new(source, Arc::new(NoHost), plain_turn(&turn))
                .render_block("history")
                .await
        };

        let found = store.get_relevant_episodes("history", 12).await.unwrap();
        assert_eq!(found.len(), 3, "{found:?}");
        let unplaced = Arc::new(RecallSource::new(store.clone(), settings.clone()));
        let all = render(unplaced, None).await;
        assert!(
            ["Alpha", "Beta", "Gamma"]
                .iter()
                .any(|history| all.contains(history)),
            "{all}"
        );

        let placed =
            Arc::new(RecallSource::new(store, settings).with_session_projects(Arc::new(Placed)));
        let in_a = render(placed.clone(), Some("project-a")).await;
        assert!(in_a.contains("Alpha"), "{in_a}");
        assert!(!in_a.contains("Beta") && !in_a.contains("Gamma"), "{in_a}");
        let in_b = render(placed.clone(), Some("project-b")).await;
        assert!(in_b.contains("Beta"), "{in_b}");
        assert!(!in_b.contains("Alpha") && !in_b.contains("Gamma"), "{in_b}");
        let plain = render(placed, None).await;
        assert!(plain.contains("Gamma"), "{plain}");
        assert!(
            !plain.contains("Alpha") && !plain.contains("Beta"),
            "{plain}"
        );
    }

    /// A recall started early renders the same block the model call would
    /// have rendered itself, and the first render takes it.
    /// Counts the decision points recorded through it.
    #[derive(Default)]
    struct CountingHost {
        decisions: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl RecallHost for CountingHost {
        async fn active_goals(&self) -> anyhow::Result<Vec<RecallGoal>> {
            Ok(Vec::new())
        }

        async fn people_enabled(&self) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn decision(
            &self,
            _turn: &RecallTurn,
            _kind: super::super::recall::RecallDecision,
            _summary: String,
            _metadata: serde_json::Value,
        ) {
            self.decisions
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    async fn teal_source() -> Arc<RecallSource> {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let store = SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)));
        crate::stores::FactStore::upsert_fact(
            &store,
            "preference",
            "favorite_color",
            "teal",
            "user",
            None,
            crate::records::FactPrivacy::Global,
        )
        .await
        .unwrap();
        Arc::new(RecallSource::new(
            Arc::new(store),
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
        ))
    }

    fn arriving_turn() -> RecallTurn {
        RecallTurn {
            user_text: Some("what is my favorite color?".into()),
            ..plain_turn(&MemoryTurn {
                session_id: "test:synthetic-user".into(),
                project: None,
                scope: "owner".into(),
                read: true,
                write: false,
                explicit_writes_only: false,
                learning: Default::default(),
                host: Default::default(),
            })
        }
    }

    /// Recall started on arrival renders the turn's block, and its writes
    /// (here the decision points) are made by the turn that adopts it, on
    /// that turn's host, only when the block is used.
    #[tokio::test]
    async fn an_adopted_speculative_recall_renders_the_block_and_writes_as_its_turn() {
        let source = teal_source().await;
        let mut turn = arriving_turn();
        turn.task_id = "task-synthetic-1".into();
        let plain = FullRecall::new(source.clone(), Arc::new(NoHost), turn.clone())
            .render_block("")
            .await;

        let early_host = Arc::new(CountingHost::default());
        let speculative =
            SpeculativeRecall::start(source.clone(), early_host.clone(), arriving_turn()).unwrap();
        assert!(speculative.serves(&turn));
        let turn_host = Arc::new(CountingHost::default());
        let adopted = FullRecall::new(source, turn_host.clone(), turn).adopt(speculative);
        let note = aidaemon_policies::Recall::render(&adopted, "test:synthetic-user", "", 5)
            .await
            .unwrap();
        assert_eq!(note, plain);
        assert!(note.contains("teal"), "{note}");
        let decisions =
            |host: &CountingHost| host.decisions.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            decisions(&early_host),
            0,
            "a speculative render writes nothing"
        );
        assert!(
            decisions(&turn_host) >= 1,
            "the adopting turn records its recall"
        );
    }

    /// Only the turn it was started for may take it: a turn that may not
    /// read memory, or recalls for other inputs, never sees what it read.
    #[tokio::test]
    async fn a_speculative_recall_serves_only_its_own_allowed_turn() {
        let source = teal_source().await;
        let speculative =
            SpeculativeRecall::start(source, Arc::new(NoHost), arriving_turn()).unwrap();
        let mut forbidden = arriving_turn();
        forbidden.allowed = false;
        assert!(!speculative.serves(&forbidden));
        let mut other_text = arriving_turn();
        other_text.user_text = Some("and my favorite food?".into());
        assert!(!speculative.serves(&other_text));
        let mut public = arriving_turn();
        public.visibility = ChannelVisibility::PublicExternal;
        public.personal = false;
        assert!(!speculative.serves(&public));
        let mut child = arriving_turn();
        child.depth = 1;
        assert!(!speculative.serves(&child));
        let mut later_task = arriving_turn();
        later_task.task_id = "task-synthetic-2".into();
        assert!(speculative.serves(&later_task));
    }

    /// A newer message replaces an unclaimed recall, and a turn that is not
    /// the speculative one's drops it and recalls on its own.
    #[tokio::test]
    async fn the_memory_system_hands_a_speculative_recall_only_to_a_matching_turn() {
        let memory = FullMemory::new(teal_source().await);
        memory.speculate(arriving_turn(), Arc::new(NoHost));
        let mut forbidden = arriving_turn();
        forbidden.allowed = false;
        assert!(memory.claim_speculative(&forbidden).is_none());
        assert!(
            memory.claim_speculative(&arriving_turn()).is_none(),
            "dropped on mismatch"
        );
        memory.speculate(arriving_turn(), Arc::new(NoHost));
        assert!(memory.claim_speculative(&arriving_turn()).is_some());
    }

    #[tokio::test]
    async fn a_prefetched_recall_renders_the_same_block() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let store = SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)));
        crate::stores::FactStore::upsert_fact(
            &store,
            "preference",
            "favorite_color",
            "teal",
            "user",
            None,
            crate::records::FactPrivacy::Global,
        )
        .await
        .unwrap();
        let source = Arc::new(RecallSource::new(
            Arc::new(store),
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
        ));
        let mut turn = RecallTurn {
            user_text: Some("what is my favorite color?".into()),
            ..plain_turn(&MemoryTurn {
                session_id: "test:synthetic-user".into(),
                project: None,
                scope: "owner".into(),
                read: true,
                write: false,
                explicit_writes_only: false,
                learning: Default::default(),
                host: Default::default(),
            })
        };
        turn.task_id = "task-synthetic-1".into();
        let plain = FullRecall::new(source.clone(), Arc::new(NoHost), turn.clone())
            .render_block("")
            .await;
        let prefetched = FullRecall::new(source, Arc::new(NoHost), turn).prefetch();
        assert!(prefetched.prefetched.lock().unwrap().is_some());
        let note = aidaemon_policies::Recall::render(&prefetched, "test:synthetic-user", "", 5)
            .await
            .unwrap();
        assert!(note.contains("teal"), "{note}");
        assert_eq!(note, plain);
        assert!(prefetched.prefetched.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn the_whole_system_offers_its_tools_on_a_memory_turn() {
        let store = crate::full::tools::test_support::memory().await;
        let system = FullMemory::over(
            store,
            RecallSettings {
                max_facts: 10,
                assistant_name: None,
            },
            Arc::new(crate::full::tools::test_support::Settings::default()),
        );
        let mut turn = MemoryTurn {
            session_id: "test:synthetic-user".into(),
            project: None,
            scope: "owner".into(),
            read: true,
            write: true,
            explicit_writes_only: false,
            learning: Default::default(),
            host: Default::default(),
        };
        let names = |parts: &MemoryParts| -> Vec<String> {
            parts.tools.iter().map(|tool| tool.spec().name).collect()
        };
        let parts = system.turn(&turn).await;
        let mut offered = names(&parts);
        offered.sort();
        let mut expected: Vec<String> = crate::full::tools::TOOL_NAMES
            .iter()
            .map(|name| name.to_string())
            .collect();
        expected.sort();
        assert_eq!(offered, expected);

        turn.read = false;
        turn.write = false;
        assert!(system.turn(&turn).await.tools.is_empty());
    }
}
