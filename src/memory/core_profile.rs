use std::collections::HashMap;
use std::sync::Arc;

use crate::traits::{Fact, StateStore};
use chrono::Utc;

#[derive(Debug, Clone)]
struct RankedEntity {
    id: String,
    lines: Vec<String>,
    salience: f32,
}

/// Build the core profile string containing relationships and other core identity facts.
/// Computes salience per entity and selects the top N.
pub async fn build_core_profile(
    state: &Arc<dyn StateStore>,
    cached_ids: Option<Vec<String>>,
    people_enabled: bool,
) -> anyhow::Result<(String, Option<Vec<String>>)> {
    let now = Utc::now();
    let mut entities = Vec::new();

    let mut owner_id = None;
    if people_enabled {
        let people = state.get_all_people().await.unwrap_or_default();
        for person in people {
            if person
                .relationship
                .as_deref()
                .map(|s| s.to_ascii_lowercase())
                == Some("owner".to_string())
            {
                owner_id = Some(person.id);
                continue; // Don't add owner as a social circle entity
            }

            let days_old = (now - person.updated_at).num_days() as f32;
            let recency_boost = (30.0 - days_old).max(0.0) * 0.1;
            let salience = person.interaction_count as f32 + recency_boost;

            let rel = person.relationship.as_deref().unwrap_or("contact");
            let mut parts = vec![person.name.clone()];
            if let Some(notes) = &person.notes {
                parts.push(format!("(b. {})", notes));
            }
            let line = format!("Your {}: {}", rel, parts.join(" "));

            entities.push(RankedEntity {
                id: format!("person_{}", person.id),
                lines: vec![line],
                salience,
            });
        }
    }

    if let Some(oid) = owner_id {
        let person_facts = state.get_person_facts(oid, None).await.unwrap_or_default();
        for fact in person_facts {
            let days_old = (now - fact.updated_at).num_days() as f32;
            let recency_boost = (30.0 - days_old).max(0.0) * 0.1;
            let salience = fact.confidence + recency_boost;

            entities.push(RankedEntity {
                id: format!("person_fact_{}", fact.id),
                lines: vec![format!("• {}: {}", fact.key, fact.value)],
                salience,
            });
        }
    }

    let flat_facts = state.get_facts(None).await.unwrap_or_default();
    let mut flat_groups: HashMap<String, Vec<Fact>> = HashMap::new();

    for fact in flat_facts {
        let key = fact.key.trim().to_ascii_lowercase();
        let base_id = if key == "wife"
            || key == "husband"
            || key == "spouse"
            || key == "partner"
            || key == "daughter"
            || key == "son"
            || key == "child"
            || key == "kid"
        {
            key.clone()
        } else if let Some(idx) = key.find('_') {
            let prefix = &key[..idx];
            if matches!(
                prefix,
                "wife"
                    | "husband"
                    | "spouse"
                    | "partner"
                    | "daughter"
                    | "son"
                    | "child"
                    | "kid"
                    | "mother"
                    | "father"
                    | "mom"
                    | "dad"
            ) {
                if let Some(idx2) = key[idx + 1..].find('_') {
                    if key[idx + 1..idx + 1 + idx2]
                        .chars()
                        .all(|c| c.is_ascii_digit())
                    {
                        key[..idx + 1 + idx2].to_string()
                    } else {
                        prefix.to_string()
                    }
                } else {
                    prefix.to_string()
                }
            } else {
                key.clone()
            }
        } else {
            key.clone()
        };
        flat_groups.entry(base_id).or_default().push(fact);
    }

    for (group_id, facts) in flat_groups {
        let mut max_recall = 0;
        let mut newest_update = chrono::DateTime::<Utc>::MIN_UTC;

        let mut name = None;
        let mut details = Vec::new();
        let mut generic_lines = Vec::new();

        let is_rel = group_id == "wife"
            || group_id == "husband"
            || group_id == "spouse"
            || group_id == "partner"
            || group_id.starts_with("daughter")
            || group_id.starts_with("son")
            || group_id.starts_with("child")
            || group_id.starts_with("kid");

        for fact in facts {
            max_recall = max_recall.max(fact.recall_count);
            if fact.updated_at > newest_update {
                newest_update = fact.updated_at;
            }

            if is_rel {
                let key = fact.key.to_ascii_lowercase();
                if key == group_id || key == format!("{}_name", group_id) {
                    name = Some(fact.value.clone());
                } else {
                    details.push(fact.value.clone());
                }
            } else {
                generic_lines.push(format!("• {}: {}", fact.key, fact.value));
            }
        }

        let days_old = (now - newest_update).num_days() as f32;
        let recency_boost = (30.0 - days_old).max(0.0) * 0.1;
        let salience = max_recall as f32 + recency_boost;

        let mut lines = Vec::new();
        if is_rel {
            let display_name = name.unwrap_or_else(|| "Unknown".to_string());
            let mut parts = vec![display_name];
            if !details.is_empty() {
                parts.push(format!("({})", details.join(", ")));
            }
            let label = if group_id == "wife"
                || group_id == "husband"
                || group_id == "spouse"
                || group_id == "partner"
            {
                "partner"
            } else if group_id.starts_with("daughter")
                || group_id.starts_with("son")
                || group_id.starts_with("child")
                || group_id.starts_with("kid")
            {
                "child"
            } else {
                group_id.as_str()
            };
            lines.push(format!("Your {}: {}", label, parts.join(" ")));
        } else {
            lines.extend(generic_lines);
        }

        entities.push(RankedEntity {
            id: format!("flat_group_{}", group_id),
            lines,
            salience,
        });
    }

    // Apply cache or rank
    let mut selected_entities = Vec::new();
    let mut new_cache = None;

    if let Some(cached) = cached_ids {
        for entity in entities {
            if cached.contains(&entity.id) {
                selected_entities.push(entity);
            }
        }
    } else {
        entities.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_n = entities.into_iter().take(20).collect::<Vec<_>>();

        let ids: Vec<String> = top_n.iter().map(|e| e.id.clone()).collect();
        new_cache = Some(ids);
        selected_entities = top_n;
    }

    if selected_entities.is_empty() {
        return Ok((String::new(), new_cache));
    }

    let mut out = String::from(
        "## Core Profile\n\
         Stored background memory about your operator (\"the owner\"). IMPORTANT: when \
         the user's message contains a definite reference (\"the owner\", \"the CEO\", \
         \"it\", \"they\") with an antecedent in the recent conversation, resolve it \
         against the conversation FIRST — e.g. after discussing a company, \"Who's the \
         owner?\" means that company's owner, NOT your operator. Only fall back to this \
         profile when no conversational antecedent exists.\n\n",
    );
    for entity in selected_entities {
        for line in entity.lines {
            out.push_str(&format!("{}\n", line));
        }
    }

    Ok((out.trim_end().to_string(), new_cache))
}
