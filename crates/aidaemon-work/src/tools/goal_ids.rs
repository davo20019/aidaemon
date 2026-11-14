//! How a goal handle the model gives (a full id, a unique id prefix, or a
//! distinctive phrase from the objective) picks one goal, for every tool
//! that takes one: the same handle resolves the same way everywhere.

use crate::traits::Goal;

/// The first characters of a goal id, as tools show it.
pub fn short_goal_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Pick the goal `handle` names among `candidates`: by id prefix, else by a
/// phrase (8+ characters) from the objective, which overviews show instead
/// of ids. More than one match is an error that lists them, live ones
/// first. `kind` names the candidates in the not-found error ("Goal",
/// "Scheduled goal").
pub fn pick_goal(handle: &str, candidates: &[Goal], kind: &str) -> anyhow::Result<String> {
    let handle = handle.trim();
    anyhow::ensure!(!handle.is_empty(), "Empty goal ID");
    let mut matches: Vec<&Goal> = candidates
        .iter()
        .filter(|goal| goal.id.starts_with(handle))
        .collect();
    if matches.is_empty() {
        let needle = handle.to_lowercase();
        if needle.chars().count() >= 8 {
            matches = candidates
                .iter()
                .filter(|goal| goal.description.to_lowercase().contains(&needle))
                .collect();
        }
    }
    match matches.as_slice() {
        [] => anyhow::bail!("{kind} not found: {handle}"),
        [goal] => return Ok(goal.id.clone()),
        _ => {}
    }
    matches.sort_by_key(|goal| match goal.status.as_str() {
        "active" => 0usize,
        "paused" => 1,
        "pending_confirmation" => 2,
        "failed" => 3,
        "cancelled" => 4,
        "completed" => 5,
        _ => 6,
    });
    let preview = matches
        .iter()
        .take(5)
        .map(|goal| {
            let objective: String = goal.description.chars().take(40).collect();
            format!("{} ({}, {objective})", short_goal_id(&goal.id), goal.status)
        })
        .collect::<Vec<_>>()
        .join("; ");
    anyhow::bail!(
        "Goal ID prefix '{handle}' is ambiguous ({} matches): {preview}. Use full goal_id.",
        matches.len()
    )
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
    fn a_handle_picks_by_prefix_then_objective_and_lists_ambiguity() {
        let goals = [
            goal("abc12345-1", "active", "Post the weekly synthetic digest"),
            goal("abc99999-2", "failed", "Water the synthetic plants"),
            goal("def00000-3", "paused", "Archive old synthetic notes"),
        ];
        assert_eq!(pick_goal("def", &goals, "Goal").unwrap(), "def00000-3");
        assert_eq!(
            pick_goal("weekly synthetic digest", &goals, "Goal").unwrap(),
            "abc12345-1"
        );
        let ambiguous = pick_goal("abc", &goals, "Goal").unwrap_err().to_string();
        assert!(
            ambiguous.find("abc12345").unwrap() < ambiguous.find("abc99999").unwrap(),
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
}
