//! A conversation's participants: each coding agent session that worked in
//! it, as one row. A participant is a thread of runs (protocol 1.48,
//! `RunSummary::thread`): its first run and every follow-up that went on in
//! the same agent session. Two Claude Code runs started apart are two
//! participants; a follow-up joins the one it continues.
//!
//! Every field comes from the runs' typed records; nothing is read from
//! what an agent wrote.

use crate::{RunPhase, RunSummary};

/// One agent session of a conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Participant {
    /// Its thread: its first run's id, stable for as long as it lasts.
    pub thread: String,
    /// Its coding agent ("Claude Code").
    pub agent: String,
    /// Its name: the one the owner gave it ("Designer"), else its agent,
    /// numbered when the conversation has several of that agent ("Codex",
    /// "Codex 2"), in the order they started. A task's title would go
    /// stale on the agent's next task.
    pub name: String,
    /// The owner named it.
    pub named: bool,
    /// One of its runs waits on the owner's answer to an approval.
    pub waiting: bool,
    /// Its latest run: what its row opens, and what a follow-up continues.
    pub latest: RunSummary,
    /// How many runs it took: its first and each follow-up.
    pub runs: usize,
    /// When it last started or ended a run.
    pub last_activity_ms: u64,
}

impl Participant {
    /// How a client heads it: its name and agent ("Designer · Codex"), or
    /// its numbered agent alone ("Codex 2") when the owner didn't name it.
    pub fn title(&self) -> String {
        if self.named {
            format!("{} · {}", self.name, self.agent)
        } else {
            self.name.clone()
        }
    }

    /// It is working now.
    pub fn working(&self) -> bool {
        self.latest.phase == RunPhase::Running
    }

    /// A follow-up can go on from it: its latest run ended and its agent
    /// kept its session.
    pub fn can_continue(&self) -> bool {
        self.latest.can_continue
    }
}

/// The participants among `runs` (one conversation's): those waiting on the
/// owner first, then working ones, then the most recently active; named as
/// `names` (by thread) says, waiting when one of their runs is among
/// `asking` (the runs an open approval card names). Runs with no coding
/// agent are left out.
pub fn participants(
    runs: &[RunSummary],
    names: &std::collections::BTreeMap<String, String>,
    asking: &std::collections::BTreeSet<String>,
) -> Vec<Participant> {
    let mut threads: Vec<(String, Vec<&RunSummary>)> = Vec::new();
    for run in runs.iter().filter(|run| run.agent.is_some()) {
        let thread = run.thread.clone().unwrap_or_else(|| run.id.clone());
        match threads.iter_mut().find(|(id, _)| *id == thread) {
            Some((_, list)) => list.push(run),
            None => threads.push((thread, vec![run])),
        }
    }
    let mut out: Vec<(u64, Participant)> = threads
        .into_iter()
        .filter_map(|(thread, mut list)| {
            list.sort_by_key(|run| run.started_at_ms);
            // Its first run names it; a thread whose first run is no
            // longer on record is named by its earliest one kept.
            let first = list.iter().find(|run| run.id == thread).or(list.first())?;
            let latest = (*list.last()?).clone();
            let last_activity_ms = list
                .iter()
                .map(|run| run.ended_at_ms.unwrap_or(run.started_at_ms))
                .max()
                .unwrap_or(0);
            let given = names.get(&thread);
            let waiting = list.iter().any(|run| asking.contains(&run.id));
            let agent = first.agent.clone().unwrap_or_default();
            Some((
                first.started_at_ms,
                Participant {
                    name: given.cloned().unwrap_or_else(|| agent.clone()),
                    agent,
                    named: given.is_some(),
                    waiting,
                    runs: list.len(),
                    thread,
                    latest,
                    last_activity_ms,
                },
            ))
        })
        .collect();
    // Each agent's sessions numbered in the order they started, the first
    // plain: "Codex", "Codex 2".
    out.sort_by(|(a, pa), (b, pb)| a.cmp(b).then(pa.thread.cmp(&pb.thread)));
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (_, p) in &mut out {
        let n = seen.entry(p.agent.clone()).or_default();
        *n += 1;
        if !p.named && *n > 1 {
            p.name = format!("{} {n}", p.agent);
        }
    }
    let mut out: Vec<Participant> = out.into_iter().map(|(_, p)| p).collect();
    out.sort_by(|a, b| {
        b.waiting
            .cmp(&a.waiting)
            .then(b.working().cmp(&a.working()))
            .then(b.last_activity_ms.cmp(&a.last_activity_ms))
            .then(a.thread.cmp(&b.thread))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        id: &str,
        agent: Option<&str>,
        thread: Option<&str>,
        at: u64,
        phase: &str,
    ) -> RunSummary {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "session_id": "app:synthetic",
            "label": format!("label {id}"),
            "agent": agent,
            "phase": phase,
            "background": true,
            "started_at_ms": at,
            "ended_at_ms": (phase != "running").then_some(at + 5),
            "thread": thread,
            "can_continue": phase == "succeeded",
        }))
        .expect("a run")
    }

    /// Two runs of the same agent started apart are two participants; a
    /// follow-up joins the thread it continues, and the latest run speaks
    /// for it.
    #[test]
    fn each_agent_session_is_one_participant() {
        let runs = [
            run("a", Some("Claude Code"), None, 100, "succeeded"),
            run("b", Some("Claude Code"), None, 110, "succeeded"),
            run("a2", Some("Claude Code"), Some("a"), 200, "running"),
            run("h", None, None, 120, "succeeded"),
        ];
        let names = [("b".to_string(), "Designer".to_string())].into();
        let asking = ["b".to_string()].into();
        let list = participants(&runs, &names, &asking);
        assert_eq!(list.len(), 2, "a helper with no coding agent is not one");
        assert_eq!(
            list[0].thread, "b",
            "the one waiting on the owner comes first"
        );
        assert!(list[0].waiting && !list[1].waiting);
        let list = participants(&runs, &names, &Default::default());
        assert_eq!(list[0].thread, "a", "then the working one");
        assert_eq!(list[0].name, "Claude Code", "named by its agent");
        assert_eq!(list[0].title(), "Claude Code");
        assert_eq!(list[0].latest.id, "a2");
        assert_eq!(list[0].runs, 2);
        assert!(list[0].working() && !list[0].can_continue());
        assert_eq!(list[1].thread, "b");
        assert!(list[1].can_continue());
        assert_eq!(list[1].name, "Designer", "the owner's name wins");
        assert_eq!(list[1].title(), "Designer · Claude Code");
        assert!(list[1].named && !list[0].named);
        // Unnamed, the second session of the same agent is numbered.
        let list = participants(&runs, &Default::default(), &Default::default());
        let named = |thread: &str| {
            list.iter()
                .find(|p| p.thread == thread)
                .map(|p| p.name.clone())
        };
        assert_eq!(named("a").as_deref(), Some("Claude Code"));
        assert_eq!(named("b").as_deref(), Some("Claude Code 2"));
    }
}
