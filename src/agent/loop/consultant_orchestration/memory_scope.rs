use crate::traits::{Fact, Procedure};

fn normalize_for_project_match(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn text_matches_project_hint(text: &str, hint: &str) -> bool {
    let normalized_hint = normalize_for_project_match(hint);
    if normalized_hint.is_empty() {
        return false;
    }

    let lower_text = text.to_ascii_lowercase();
    if lower_text.contains(&hint.to_ascii_lowercase()) {
        return true;
    }
    let normalized_text = normalize_for_project_match(text);
    normalized_text.contains(&normalized_hint)
}

fn fact_matches_project_hints(fact: &Fact, project_hints: &[String]) -> bool {
    if project_hints.is_empty() {
        return true;
    }
    let haystack = format!(
        "{} {} {} {}",
        fact.category, fact.key, fact.value, fact.source
    );
    project_hints
        .iter()
        .any(|hint| text_matches_project_hint(&haystack, hint))
}

fn procedure_matches_project_hints(proc: &Procedure, project_hints: &[String]) -> bool {
    if project_hints.is_empty() {
        return true;
    }
    let haystack = format!(
        "{} {} {}",
        proc.name,
        proc.trigger_pattern,
        proc.steps.join(" ")
    );
    project_hints
        .iter()
        .any(|hint| text_matches_project_hint(&haystack, hint))
}

pub(super) fn scope_goal_memory_to_project_hints(
    relevant_facts: Vec<Fact>,
    relevant_procedures: Vec<Procedure>,
    project_hints: &[String],
) -> (Vec<Fact>, Vec<Procedure>) {
    if project_hints.is_empty() {
        return (relevant_facts, relevant_procedures);
    }

    let facts = relevant_facts
        .into_iter()
        .filter(|f| fact_matches_project_hints(f, project_hints))
        .collect();
    let procedures = relevant_procedures
        .into_iter()
        .filter(|p| procedure_matches_project_hints(p, project_hints))
        .collect();
    (facts, procedures)
}

pub(super) fn is_low_signal_goal_text(goal_user_text: &str) -> bool {
    let lower = goal_user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return true;
    }

    let generic_phrases = [
        "do what you consider the best",
        "do what you think is best",
        "do whatever you think is best",
        "whatever you think is best",
        "as you see fit",
        "use your best judgment",
    ];
    if generic_phrases.iter().any(|phrase| lower.contains(phrase))
        || (lower.starts_with("you are ")
            && lower.contains("do what")
            && (lower.contains("best") || lower.contains("judgment")))
    {
        return true;
    }

    if lower.contains('/')
        || lower.contains('\\')
        || lower.contains("file://")
        || lower.contains("http://")
        || lower.contains("https://")
        || lower.contains("project")
        || lower.contains("repo")
    {
        return false;
    }

    let stopwords = [
        "a", "an", "and", "are", "as", "at", "be", "best", "by", "do", "for", "from", "how", "i",
        "it", "its", "make", "modern", "my", "of", "on", "or", "please", "the", "this", "that",
        "to", "use", "with", "you", "your", "what", "which", "who", "why", "when",
    ];

    let informative_count = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .filter(|t| !stopwords.contains(t))
        .count();
    informative_count <= 1
}
