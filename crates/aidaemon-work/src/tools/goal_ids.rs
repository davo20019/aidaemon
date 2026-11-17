//! How a goal handle the model gives picks one goal, for every tool that
//! takes one: the same handle resolves the same way everywhere.
//!
//! A handle is an id: the full id, or a prefix at least as long as the short
//! id tools show. An objective is accepted only when it equals one goal's
//! objective exactly (ignoring case and spacing); words inside an objective
//! never pick a goal. Anything else is an error naming candidate ids, never
//! a guess.

use crate::traits::Goal;

/// How many leading characters of a goal id tools show.
pub const SHORT_GOAL_ID_CHARS: usize = 8;

/// The first characters of a goal id, as tools show it.
pub fn short_goal_id(id: &str) -> String {
    id.chars().take(SHORT_GOAL_ID_CHARS).collect()
}

fn normalized_objective(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn status_rank(goal: &Goal) -> usize {
    match goal.status.as_str() {
        "active" => 0,
        "paused" => 1,
        "pending_confirmation" => 2,
        "failed" => 3,
        "cancelled" => 4,
        "completed" => 5,
        _ => 6,
    }
}

fn preview(goals: &mut Vec<&Goal>) -> String {
    goals.sort_by_key(|goal| status_rank(goal));
    goals
        .iter()
        .take(5)
        .map(|goal| {
            let objective: String = goal.description.chars().take(40).collect();
            format!("{} ({}, {objective})", short_goal_id(&goal.id), goal.status)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Pick the goal `handle` names among `candidates`: its exact id, a unique
/// id prefix of at least [`SHORT_GOAL_ID_CHARS`], or an objective equal to
/// exactly one goal's. More than one match, or none, is an error that lists
/// candidate ids, live ones first. `kind` names the candidates in the
/// not-found error ("Goal", "Scheduled goal").
pub fn pick_goal(handle: &str, candidates: &[Goal], kind: &str) -> anyhow::Result<String> {
    let handle = handle.trim();
    anyhow::ensure!(!handle.is_empty(), "Empty goal ID");
    if let Some(goal) = candidates.iter().find(|goal| goal.id == handle) {
        return Ok(goal.id.clone());
    }
    let mut matches: Vec<&Goal> = if handle.chars().count() >= SHORT_GOAL_ID_CHARS {
        candidates
            .iter()
            .filter(|goal| goal.id.starts_with(handle))
            .collect()
    } else {
        Vec::new()
    };
    if matches.is_empty() {
        let objective = normalized_objective(handle);
        matches = candidates
            .iter()
            .filter(|goal| normalized_objective(&goal.description) == objective)
            .collect();
    }
    match matches.as_slice() {
        [goal] => Ok(goal.id.clone()),
        [] => {
            let mut all: Vec<&Goal> = candidates.iter().collect();
            if all.is_empty() {
                anyhow::bail!("{kind} not found: {handle}");
            }
            anyhow::bail!(
                "{kind} not found: {handle}. Give a goal id (full, or its first {SHORT_GOAL_ID_CHARS} characters); candidates: {}.",
                preview(&mut all)
            )
        }
        _ => anyhow::bail!(
            "Goal handle '{handle}' is ambiguous ({} matches): {}. Use the full goal_id.",
            matches.len(),
            preview(&mut matches)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Goal;

    fn goal(id: &str, status: &str, description: &str) -> Goal {
        let mut goal = Goal::new_finite(description, "telegram:synthetic-user-1");
        goal.id = id.to_string();
        goal.status = status.to_string();
        goal
    }

    #[test]
    fn a_handle_picks_by_id_or_exact_objective_and_lists_candidates() {
        let goals = [
            goal(
                "abc12345-1111",
                "active",
                "Post the weekly synthetic digest",
            ),
            goal("abc12345-2222", "failed", "Water the synthetic plants"),
            goal("def00000-3333", "paused", "Archive old synthetic notes"),
        ];
        assert_eq!(
            pick_goal("def00000-3333", &goals, "Goal").unwrap(),
            "def00000-3333"
        );
        assert_eq!(
            pick_goal("def00000", &goals, "Goal").unwrap(),
            "def00000-3333"
        );
        // An objective counts only when it is the whole objective.
        assert_eq!(
            pick_goal("  post the WEEKLY synthetic   digest ", &goals, "Goal").unwrap(),
            "abc12345-1111"
        );
        let partial = pick_goal("weekly synthetic digest", &goals, "Goal")
            .unwrap_err()
            .to_string();
        assert!(partial.contains("not found"), "{partial}");
        assert!(partial.contains("abc12345"), "{partial}");
        // A short prefix never picks one.
        assert!(pick_goal("def", &goals, "Goal").is_err());
        let ambiguous = pick_goal("abc12345", &goals, "Goal")
            .unwrap_err()
            .to_string();
        assert!(ambiguous.contains("ambiguous"), "{ambiguous}");
        assert!(
            ambiguous.find("active").unwrap() < ambiguous.find("failed").unwrap(),
            "{ambiguous}"
        );
        assert!(
            pick_goal("zzz", &goals, "Scheduled goal")
                .unwrap_err()
                .to_string()
                .starts_with("Scheduled goal not found")
        );
        assert!(pick_goal("  ", &goals, "Goal").is_err());
    }

    #[test]
    fn equal_objectives_are_ambiguous_not_picked() {
        let goals = [
            goal("aaaa0000-1", "active", "Check the synthetic feed"),
            goal("bbbb0000-2", "paused", "check the  synthetic feed"),
        ];
        let error = pick_goal("Check the synthetic feed", &goals, "Goal")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("aaaa0000") && error.contains("bbbb0000"),
            "{error}"
        );
    }
}
