use std::collections::HashMap;

use chrono::{DateTime, NaiveDate, Utc};
use sqlx::{Row, Sqlite, Transaction};

use super::*;
use crate::traits::{
    Fact, PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate,
    PersonalMemoryWrite, PersonalMemoryWriteResult, PersonalRelationshipCandidate,
};

fn words(value: &str) -> String {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn token(value: &str) -> String {
    words(value).replace(' ', "_")
}

fn alias_value(value: &str, alias_type: &str) -> String {
    if matches!(
        token(alias_type).as_str(),
        "username" | "handle" | "online_handle" | "account_name"
    ) {
        value.trim().trim_start_matches('@').to_lowercase()
    } else {
        words(value)
    }
}

fn date_value(value: &str) -> Option<String> {
    if let Ok(date) = NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d") {
        return Some(date.format("%Y-%m-%d").to_string());
    }
    let clean = value.trim().replace(',', "");
    ["%b %e %Y", "%B %e %Y", "%b %d %Y", "%B %d %Y"]
        .iter()
        .find_map(|format| NaiveDate::parse_from_str(&clean, format).ok())
        .map(|date| date.format("%Y-%m-%d").to_string())
}

fn predicate(value: &str) -> String {
    match token(value).as_str() {
        "birthday" | "date_of_birth" | "dob" => "birth_date".to_string(),
        "current_residence" | "home" | "location" => "residence".to_string(),
        "place_of_birth" => "birthplace".to_string(),
        "favorite_name" | "preferred_first_name" => "preferred_name".to_string(),
        other => other.to_string(),
    }
}

fn alias_type(value: &str) -> String {
    match token(value).as_str() {
        "favorite_name" | "preferred" | "preferred_first_name" => "preferred_name".to_string(),
        "online_nickname" | "online_username" => "online_handle".to_string(),
        "legal_name" | "full_name" => "legal_name_variant".to_string(),
        other => other.to_string(),
    }
}

fn entity_type(value: &str) -> String {
    match token(value).as_str() {
        kind @ ("person" | "organization" | "project" | "place" | "account") => kind.to_string(),
        _ => "person".to_string(),
    }
}

fn relation(value: &str) -> String {
    match token(value).as_str() {
        "parent" | "parent_of" | "father_of" | "mother_of" => "PARENT_OF".to_string(),
        "child" | "child_of" | "daughter_of" | "son_of" => "CHILD_OF".to_string(),
        "lives_with" | "cohabits_with" => "LIVES_WITH".to_string(),
        "lives_in" | "resident_of" => "LIVES_IN".to_string(),
        "uses_alias" => "USES_ALIAS".to_string(),
        "uses_handle" => "USES_HANDLE".to_string(),
        "has_account" => "HAS_ACCOUNT".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

fn inverse(value: &str) -> Option<&'static str> {
    match value {
        "PARENT_OF" => Some("CHILD_OF"),
        "CHILD_OF" => Some("PARENT_OF"),
        "LIVES_WITH" => Some("LIVES_WITH"),
        "PARTNER_OF" => Some("PARTNER_OF"),
        "SPOUSE_OF" => Some("SPOUSE_OF"),
        _ => None,
    }
}

fn legacy_person_name(value: &str) -> Option<(String, Vec<String>)> {
    let mut display = value.trim().to_string();
    if display.is_empty() || display.to_ascii_lowercase().contains(" and ") || display.contains('&')
    {
        return None;
    }

    let mut aliases = Vec::new();
    if let Some(start) = display.find('"') {
        if let Some(relative_end) = display[start + 1..].find('"') {
            let end = start + 1 + relative_end;
            let alias = display[start + 1..end].trim();
            if !alias.is_empty() {
                aliases.push(alias.to_string());
            }
            display.replace_range(start..=end, "");
        }
    }
    if let (Some(start), Some(end)) = (display.find('('), display.rfind(')')) {
        if end > start {
            let note = display[start + 1..end].trim();
            let lower = note.to_ascii_lowercase();
            let alias = lower
                .strip_prefix("nickname ")
                .or_else(|| lower.strip_prefix("aka "))
                .or_else(|| lower.strip_prefix("also known as "))
                .map(|prefix_removed| {
                    let offset = note.len().saturating_sub(prefix_removed.len());
                    note[offset..].trim()
                });
            if let Some(alias) = alias.filter(|alias| !alias.is_empty()) {
                aliases.push(alias.to_string());
                display.replace_range(start..=end, "");
            }
        }
    }
    let canonical = display.split_whitespace().collect::<Vec<_>>().join(" ");
    (!canonical.is_empty()).then_some((canonical, aliases))
}

fn split_legacy_people(value: &str) -> Vec<String> {
    value
        .replace(" & ", ",")
        .split([',', ';'])
        .flat_map(|part| part.split(" and "))
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

fn confidence(source: &str, direct: bool) -> f32 {
    if direct {
        1.0
    } else if matches!(
        source,
        "legacy" | "consolidation" | "inferred" | "task_learning"
    ) {
        0.6
    } else {
        0.8
    }
}

fn low_authority(source: &str) -> bool {
    matches!(
        source,
        "legacy" | "consolidation" | "inferred" | "progressive" | "task_learning"
    )
}

fn normalized_fact(predicate: &str, value: &str) -> anyhow::Result<(String, &'static str)> {
    if predicate == "birth_date" {
        Ok((
            date_value(value).ok_or_else(|| anyhow::anyhow!("unsupported date: {value}"))?,
            "date",
        ))
    } else {
        Ok((words(value), "text"))
    }
}

#[allow(clippy::too_many_arguments)]
async fn audit(
    tx: &mut Transaction<'_, Sqlite>,
    operation: &str,
    entity_id: Option<i64>,
    record_type: &str,
    record_id: Option<i64>,
    state: serde_json::Value,
    source: &str,
    provenance: Option<&str>,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO memory_write_audit
         (operation, entity_id, record_type, record_id, new_state_json, source, provenance, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(operation)
    .bind(entity_id)
    .bind(record_type)
    .bind(record_id)
    .bind(state.to_string())
    .bind(source)
    .bind(provenance)
    .bind(Utc::now().to_rfc3339())
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn review(
    tx: &mut Transaction<'_, Sqlite>,
    kind: &str,
    reference: &str,
    candidates: &[i64],
    payload: serde_json::Value,
    source: &str,
) -> anyhow::Result<()> {
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT OR IGNORE INTO memory_resolution_reviews
         (review_kind, normalized_reference, candidate_entity_ids_json, payload_json,
          source, status, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
    )
    .bind(kind)
    .bind(reference)
    .bind(serde_json::to_string(candidates)?)
    .bind(payload.to_string())
    .bind(source)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn owner_entity(
    tx: &mut Transaction<'_, Sqlite>,
    candidate: &PersonalEntityCandidate,
    source: &str,
    privacy: &str,
    channel: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<i64> {
    let existing = sqlx::query(
        "SELECT id, canonical_name FROM memory_entities
         WHERE is_owner = 1 AND status = 'active' LIMIT 1",
    )
    .fetch_optional(&mut **tx)
    .await?;
    let id = if let Some(row) = existing {
        row.get("id")
    } else if let Some(id) = sqlx::query_scalar::<_, i64>(
        "SELECT id FROM memory_entities
         WHERE entity_type = 'person' AND canonical_name = 'owner'
           AND status = 'active' LIMIT 1",
    )
    .fetch_optional(&mut **tx)
    .await?
    {
        sqlx::query(
            "UPDATE memory_entities SET is_owner = 1, privacy = ?,
             channel_id = COALESCE(channel_id, ?), updated_at = ? WHERE id = ?",
        )
        .bind(privacy)
        .bind(channel)
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&mut **tx)
        .await?;
        id
    } else {
        let now = Utc::now().to_rfc3339();
        let id = sqlx::query(
            "INSERT INTO memory_entities
             (entity_type, canonical_name, display_name, aliases_json, channel_id, privacy,
              status, is_owner, created_at, updated_at)
             VALUES ('person', 'owner', 'Owner', '[]', ?, ?, 'active', 1, ?, ?)",
        )
        .bind(channel)
        .bind(privacy)
        .bind(&now)
        .bind(&now)
        .execute(&mut **tx)
        .await?
        .last_insert_rowid();
        result.created_entities += 1;
        audit(
            tx,
            "create",
            Some(id),
            "entity",
            Some(id),
            serde_json::json!({"entity_type": "person", "canonical_name": "owner"}),
            source,
            None,
        )
        .await?;
        id
    };

    let display = candidate.canonical_name.trim();
    let canonical = words(display);
    if candidate.canonical_name_confirmed && !canonical.is_empty() && canonical != "owner" {
        let current: String =
            sqlx::query_scalar("SELECT canonical_name FROM memory_entities WHERE id = ?")
                .bind(id)
                .fetch_one(&mut **tx)
                .await?;
        if current != canonical {
            let collision: Option<i64> = sqlx::query_scalar(
                "SELECT id FROM memory_entities
                 WHERE entity_type = 'person' AND canonical_name = ? AND id != ?
                   AND status = 'active' LIMIT 1",
            )
            .bind(&canonical)
            .bind(id)
            .fetch_optional(&mut **tx)
            .await?;
            if let Some(collision_id) = collision {
                review(
                    tx,
                    "owner_name_collision",
                    &canonical,
                    &[id, collision_id],
                    serde_json::json!({}),
                    source,
                )
                .await?;
                result
                    .unresolved
                    .push("confirmed owner name matches another entity".to_string());
            } else {
                sqlx::query(
                    "UPDATE memory_entities SET canonical_name = ?, display_name = ?,
                     updated_at = ? WHERE id = ?",
                )
                .bind(&canonical)
                .bind(display)
                .bind(Utc::now().to_rfc3339())
                .bind(id)
                .execute(&mut **tx)
                .await?;
                result.updated_entities += 1;
                audit(
                    tx,
                    "update",
                    Some(id),
                    "entity",
                    Some(id),
                    serde_json::json!({"canonical_name": canonical}),
                    source,
                    None,
                )
                .await?;
            }
        }
    }
    Ok(id)
}

async fn resolve_entity(
    tx: &mut Transaction<'_, Sqlite>,
    candidate: &PersonalEntityCandidate,
    source: &str,
    privacy: &str,
    channel: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<Option<i64>> {
    if candidate.local_id.eq_ignore_ascii_case("owner") {
        return owner_entity(tx, candidate, source, privacy, channel, result)
            .await
            .map(Some);
    }
    let kind = entity_type(&candidate.entity_type);
    let display = candidate.canonical_name.trim();
    let canonical = words(display);
    if canonical.is_empty() {
        result
            .unresolved
            .push(format!("entity {} has no usable name", candidate.local_id));
        return Ok(None);
    }
    let exact = sqlx::query(
        "SELECT id, status, merged_into_entity_id FROM memory_entities
         WHERE entity_type = ? AND canonical_name = ? LIMIT 1",
    )
    .bind(&kind)
    .bind(&canonical)
    .fetch_optional(&mut **tx)
    .await?;
    if let Some(row) = exact {
        let id: i64 = row.get("id");
        let status: String = row.get("status");
        if status == "active" {
            return Ok(Some(id));
        }
        if status == "merged" {
            if let Some(target_id) = row.get::<Option<i64>, _>("merged_into_entity_id") {
                let target_is_active: bool = sqlx::query_scalar(
                    "SELECT EXISTS(
                         SELECT 1 FROM memory_entities WHERE id = ? AND status = 'active'
                     )",
                )
                .bind(target_id)
                .fetch_one(&mut **tx)
                .await?;
                if target_is_active {
                    return Ok(Some(target_id));
                }
            }
        }
        review(
            tx,
            "inactive_entity_reference",
            &canonical,
            &[id],
            serde_json::json!({"local_id": candidate.local_id, "status": status}),
            source,
        )
        .await?;
        result
            .unresolved
            .push(format!("entity '{display}' exists but is not active"));
        return Ok(None);
    }
    if candidate.is_reference {
        let aliases: Vec<i64> = sqlx::query_scalar(
            "SELECT DISTINCT a.entity_id FROM memory_aliases a
             JOIN memory_entities e ON e.id = a.entity_id
             WHERE a.normalized_value = ? AND a.status = 'active'
               AND e.entity_type = ? AND e.status = 'active'",
        )
        .bind(&canonical)
        .bind(&kind)
        .fetch_all(&mut **tx)
        .await?;
        if aliases.len() == 1 {
            return Ok(aliases.first().copied());
        }
        let kind_review = if aliases.is_empty() {
            "unresolved_reference"
        } else {
            "ambiguous_alias"
        };
        review(
            tx,
            kind_review,
            &canonical,
            &aliases,
            serde_json::json!({"local_id": candidate.local_id}),
            source,
        )
        .await?;
        result
            .unresolved
            .push(format!("unresolved entity reference '{display}'"));
        return Ok(None);
    }
    let now = Utc::now().to_rfc3339();
    let id = sqlx::query(
        "INSERT INTO memory_entities
         (entity_type, canonical_name, display_name, aliases_json, channel_id, privacy,
          status, is_owner, created_at, updated_at)
         VALUES (?, ?, ?, '[]', ?, ?, 'active', 0, ?, ?)",
    )
    .bind(&kind)
    .bind(&canonical)
    .bind(display)
    .bind(channel)
    .bind(privacy)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?
    .last_insert_rowid();
    result.created_entities += 1;
    audit(
        tx,
        "create",
        Some(id),
        "entity",
        Some(id),
        serde_json::json!({"entity_type": kind, "canonical_name": canonical}),
        source,
        None,
    )
    .await?;
    Ok(Some(id))
}

#[allow(clippy::too_many_arguments)]
async fn write_alias(
    tx: &mut Transaction<'_, Sqlite>,
    entity_id: i64,
    candidate: &PersonalAliasCandidate,
    source: &str,
    provenance: Option<&str>,
    confidence: f32,
    direct: bool,
    privacy: &str,
    channel: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<()> {
    let kind = alias_type(&candidate.alias_type);
    let normalized = alias_value(&candidate.value, &kind);
    if normalized.is_empty() {
        return Ok(());
    }
    let now = Utc::now().to_rfc3339();
    if let Some(id) = sqlx::query_scalar::<_, i64>(
        "SELECT id FROM memory_aliases
         WHERE entity_id = ? AND alias_type = ? AND normalized_value = ?",
    )
    .bind(entity_id)
    .bind(&kind)
    .bind(&normalized)
    .fetch_optional(&mut **tx)
    .await?
    {
        sqlx::query(
            "UPDATE memory_aliases SET status = 'active', value = ?,
             confidence = MAX(confidence, ?), source = ?, provenance = COALESCE(?, provenance),
             last_confirmed_at = CASE WHEN ? THEN ? ELSE last_confirmed_at END,
             confirmed_at = CASE WHEN ? THEN COALESCE(confirmed_at, ?) ELSE confirmed_at END,
             valid_to = NULL, updated_at = ? WHERE id = ?",
        )
        .bind(candidate.value.trim())
        .bind(confidence)
        .bind(source)
        .bind(provenance)
        .bind(direct)
        .bind(&now)
        .bind(direct)
        .bind(&now)
        .bind(&now)
        .bind(id)
        .execute(&mut **tx)
        .await?;
        result.confirmed_aliases += 1;
        return Ok(());
    }
    let id = sqlx::query(
        "INSERT INTO memory_aliases
         (entity_id, alias_type, value, normalized_value, status, source, provenance,
          confidence, channel_id, privacy, asserted_at, confirmed_at, last_confirmed_at,
          created_at, updated_at)
         VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?,
                 CASE WHEN ? THEN ? END, CASE WHEN ? THEN ? END, ?, ?)",
    )
    .bind(entity_id)
    .bind(&kind)
    .bind(candidate.value.trim())
    .bind(&normalized)
    .bind(source)
    .bind(provenance)
    .bind(confidence)
    .bind(channel)
    .bind(privacy)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?
    .last_insert_rowid();
    result.created_aliases += 1;
    audit(
        tx,
        "create",
        Some(entity_id),
        "alias",
        Some(id),
        serde_json::json!({"alias_type": kind, "normalized_value": normalized}),
        source,
        provenance,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn write_fact(
    tx: &mut Transaction<'_, Sqlite>,
    entity_id: i64,
    candidate: &PersonalFactCandidate,
    source: &str,
    provenance: Option<&str>,
    confidence: f32,
    direct: bool,
    correction: bool,
    privacy: &str,
    channel: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<()> {
    let pred = predicate(&candidate.predicate);
    let (normalized, value_type) = normalized_fact(&pred, &candidate.value)?;
    let display = candidate
        .display_value
        .as_deref()
        .unwrap_or(candidate.value.trim());
    let now = Utc::now().to_rfc3339();
    if let Some(id) = sqlx::query_scalar::<_, i64>(
        "SELECT id FROM memory_entity_facts
         WHERE subject_entity_id = ? AND predicate = ? AND normalized_value = ?
           AND status = 'active' AND valid_to IS NULL LIMIT 1",
    )
    .bind(entity_id)
    .bind(&pred)
    .bind(&normalized)
    .fetch_optional(&mut **tx)
    .await?
    {
        sqlx::query(
            "UPDATE memory_entity_facts SET display_value = ?, source = ?,
             confidence = MAX(confidence, ?), provenance = COALESCE(?, provenance),
             last_confirmed_at = CASE WHEN ? THEN ? ELSE last_confirmed_at END,
             confirmed_at = CASE WHEN ? THEN COALESCE(confirmed_at, ?) ELSE confirmed_at END,
             updated_at = ? WHERE id = ?",
        )
        .bind(display)
        .bind(source)
        .bind(confidence)
        .bind(provenance)
        .bind(direct)
        .bind(&now)
        .bind(direct)
        .bind(&now)
        .bind(&now)
        .bind(id)
        .execute(&mut **tx)
        .await?;
        result.confirmed_facts += 1;
        return Ok(());
    }
    let active = sqlx::query(
        "SELECT id, normalized_value, source, confidence FROM memory_entity_facts
         WHERE subject_entity_id = ? AND predicate = ? AND status = 'active'
           AND valid_to IS NULL ORDER BY confidence DESC, updated_at DESC LIMIT 1",
    )
    .bind(entity_id)
    .bind(&pred)
    .fetch_optional(&mut **tx)
    .await?;
    let mut supersedes = None;
    if let Some(row) = active {
        let old_id: i64 = row.get("id");
        let old_source: String = row.get("source");
        let old_confidence = row.get::<f64, _>("confidence") as f32;
        if direct && (correction || old_confidence < 0.9 || low_authority(&old_source)) {
            sqlx::query(
                "UPDATE memory_entity_facts SET status = 'superseded', valid_to = ?,
                 updated_at = ? WHERE id = ?",
            )
            .bind(&now)
            .bind(&now)
            .bind(old_id)
            .execute(&mut **tx)
            .await?;
            result.superseded_facts += 1;
            supersedes = Some(old_id);
            audit(
                tx,
                "supersede",
                Some(entity_id),
                "fact",
                Some(old_id),
                serde_json::json!({"replacement_value": normalized}),
                source,
                provenance,
            )
            .await?;
        } else {
            let disputed_id = sqlx::query(
                "INSERT INTO memory_entity_facts
                 (subject_entity_id, predicate, value_type, value, normalized_value,
                  display_value, status, source, provenance, confidence, channel_id,
                  privacy, asserted_at, confirmed_at, last_confirmed_at, valid_from,
                  valid_to, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, 'disputed', ?, ?, ?, ?, ?, ?,
                         CASE WHEN ? THEN ? END, CASE WHEN ? THEN ? END, ?, ?, ?, ?)",
            )
            .bind(entity_id)
            .bind(&pred)
            .bind(value_type)
            .bind(&normalized)
            .bind(&normalized)
            .bind(display)
            .bind(source)
            .bind(provenance)
            .bind(confidence)
            .bind(channel)
            .bind(privacy)
            .bind(&now)
            .bind(direct)
            .bind(&now)
            .bind(direct)
            .bind(&now)
            .bind(candidate.valid_from.as_deref())
            .bind(candidate.valid_to.as_deref())
            .bind(&now)
            .bind(&now)
            .execute(&mut **tx)
            .await?
            .last_insert_rowid();
            review(
                tx,
                "conflicting_fact",
                &format!("{entity_id}:{pred}"),
                &[entity_id],
                serde_json::json!({"active_fact_id": old_id, "disputed_fact_id": disputed_id}),
                source,
            )
            .await?;
            result.disputed_facts += 1;
            result
                .unresolved
                .push(format!("conflicting confirmed value for {pred}"));
            return Ok(());
        }
    }
    let id = sqlx::query(
        "INSERT INTO memory_entity_facts
         (subject_entity_id, predicate, value_type, value, normalized_value,
          display_value, status, source, provenance, confidence, channel_id,
          privacy, asserted_at, confirmed_at, last_confirmed_at, valid_from,
          valid_to, supersedes_fact_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?,
                 CASE WHEN ? THEN ? END, CASE WHEN ? THEN ? END, ?, ?, ?, ?, ?)",
    )
    .bind(entity_id)
    .bind(&pred)
    .bind(value_type)
    .bind(&normalized)
    .bind(&normalized)
    .bind(display)
    .bind(source)
    .bind(provenance)
    .bind(confidence)
    .bind(channel)
    .bind(privacy)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(candidate.valid_from.as_deref())
    .bind(candidate.valid_to.as_deref())
    .bind(supersedes)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?
    .last_insert_rowid();
    result.created_facts += 1;
    audit(
        tx,
        "create",
        Some(entity_id),
        "fact",
        Some(id),
        serde_json::json!({"predicate": pred, "normalized_value": normalized}),
        source,
        provenance,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn one_edge(
    tx: &mut Transaction<'_, Sqlite>,
    from: i64,
    rel: &str,
    to: i64,
    source: &str,
    provenance: Option<&str>,
    confidence: f32,
    direct: bool,
    privacy: &str,
    channel: Option<&str>,
    valid_from: Option<&str>,
    valid_to: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<i64> {
    let now = Utc::now().to_rfc3339();
    if let Some(id) = sqlx::query_scalar::<_, i64>(
        "SELECT id FROM memory_relationships
         WHERE source_entity_id = ? AND relationship_type = ? AND target_entity_id = ?
           AND status = 'active' AND valid_to IS NULL LIMIT 1",
    )
    .bind(from)
    .bind(rel)
    .bind(to)
    .fetch_optional(&mut **tx)
    .await?
    {
        sqlx::query(
            "UPDATE memory_relationships SET source = ?, confidence = MAX(confidence, ?),
             provenance = COALESCE(?, provenance),
             last_confirmed_at = CASE WHEN ? THEN ? ELSE last_confirmed_at END,
             confirmed_at = CASE WHEN ? THEN COALESCE(confirmed_at, ?) ELSE confirmed_at END,
             updated_at = ? WHERE id = ?",
        )
        .bind(source)
        .bind(confidence)
        .bind(provenance)
        .bind(direct)
        .bind(&now)
        .bind(direct)
        .bind(&now)
        .bind(&now)
        .bind(id)
        .execute(&mut **tx)
        .await?;
        result.confirmed_relationships += 1;
        return Ok(id);
    }
    let id = sqlx::query(
        "INSERT INTO memory_relationships
         (source_entity_id, relationship_type, target_entity_id, status, source,
          provenance, confidence, channel_id, privacy, asserted_at, confirmed_at,
          last_confirmed_at, valid_from, valid_to, created_at, updated_at)
         VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?,
                 CASE WHEN ? THEN ? END, CASE WHEN ? THEN ? END, ?, ?, ?, ?)",
    )
    .bind(from)
    .bind(rel)
    .bind(to)
    .bind(source)
    .bind(provenance)
    .bind(confidence)
    .bind(channel)
    .bind(privacy)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(direct)
    .bind(&now)
    .bind(valid_from)
    .bind(valid_to)
    .bind(&now)
    .bind(&now)
    .execute(&mut **tx)
    .await?
    .last_insert_rowid();
    result.created_relationships += 1;
    audit(
        tx,
        "create",
        Some(from),
        "relationship",
        Some(id),
        serde_json::json!({"relationship_type": rel, "target_entity_id": to}),
        source,
        provenance,
    )
    .await?;
    Ok(id)
}

#[allow(clippy::too_many_arguments)]
async fn write_edge(
    tx: &mut Transaction<'_, Sqlite>,
    from: i64,
    to: i64,
    candidate: &PersonalRelationshipCandidate,
    source: &str,
    provenance: Option<&str>,
    confidence: f32,
    direct: bool,
    privacy: &str,
    channel: Option<&str>,
    result: &mut PersonalMemoryWriteResult,
) -> anyhow::Result<()> {
    if from == to {
        return Ok(());
    }
    let rel = relation(&candidate.relationship_type);
    let first = one_edge(
        tx,
        from,
        &rel,
        to,
        source,
        provenance,
        confidence,
        direct,
        privacy,
        channel,
        candidate.valid_from.as_deref(),
        candidate.valid_to.as_deref(),
        result,
    )
    .await?;
    if let Some(back_relation) = inverse(&rel) {
        let second = one_edge(
            tx,
            to,
            back_relation,
            from,
            source,
            provenance,
            confidence,
            direct,
            privacy,
            channel,
            candidate.valid_from.as_deref(),
            candidate.valid_to.as_deref(),
            result,
        )
        .await?;
        sqlx::query(
            "UPDATE memory_relationships SET inverse_relationship_id =
             CASE WHEN id = ? THEN ? ELSE ? END WHERE id IN (?, ?)",
        )
        .bind(first)
        .bind(second)
        .bind(first)
        .bind(first)
        .bind(second)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

impl SqliteStateStore {
    pub(crate) async fn reconcile_structured_personal_memory(
        &self,
        write: &PersonalMemoryWrite,
        source: &str,
        provenance: Option<&str>,
        channel: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<PersonalMemoryWriteResult> {
        let mut tx = self.pool.begin().await?;
        let mut result = PersonalMemoryWriteResult::default();
        let confidence = confidence(source, write.direct_user_statement);
        let privacy = privacy.to_string();
        let mut entities = HashMap::new();
        for candidate in &write.entities {
            if let Some(id) =
                resolve_entity(&mut tx, candidate, source, &privacy, channel, &mut result).await?
            {
                entities.insert(candidate.local_id.clone(), id);
            }
        }
        for candidate in &write.aliases {
            if let Some(&id) = entities.get(&candidate.entity_local_id) {
                write_alias(
                    &mut tx,
                    id,
                    candidate,
                    source,
                    provenance,
                    confidence,
                    write.direct_user_statement,
                    &privacy,
                    channel,
                    &mut result,
                )
                .await?;
            } else {
                result.unresolved.push(format!(
                    "alias '{}' has no resolved entity",
                    candidate.value
                ));
            }
        }
        for candidate in &write.facts {
            if let Some(&id) = entities.get(&candidate.subject_local_id) {
                write_fact(
                    &mut tx,
                    id,
                    candidate,
                    source,
                    provenance,
                    confidence,
                    write.direct_user_statement,
                    write.correction,
                    &privacy,
                    channel,
                    &mut result,
                )
                .await?;
            } else {
                result.unresolved.push(format!(
                    "fact '{}' has no resolved entity",
                    candidate.predicate
                ));
            }
        }
        for candidate in &write.relationships {
            if let (Some(&from), Some(&to)) = (
                entities.get(&candidate.source_local_id),
                entities.get(&candidate.target_local_id),
            ) {
                write_edge(
                    &mut tx,
                    from,
                    to,
                    candidate,
                    source,
                    provenance,
                    confidence,
                    write.direct_user_statement,
                    &privacy,
                    channel,
                    &mut result,
                )
                .await?;
            } else {
                result.unresolved.push(format!(
                    "relationship '{}' has an unresolved endpoint",
                    candidate.relationship_type
                ));
            }
        }
        tx.commit().await?;
        Ok(result)
    }

    pub(crate) async fn canonical_personal_facts(&self) -> anyhow::Result<Vec<Fact>> {
        let rows = sqlx::query(
            "SELECT f.id, f.predicate, f.display_value, f.value, f.source, f.asserted_at,
                    f.updated_at, f.channel_id, f.privacy, e.display_name, e.is_owner
             FROM memory_entity_facts f
             JOIN memory_entities e ON e.id = f.subject_entity_id
             WHERE f.status = 'active' AND f.valid_to IS NULL AND e.status = 'active'
             ORDER BY f.updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut output = Vec::new();
        for row in rows {
            let asserted = DateTime::parse_from_rfc3339(&row.get::<String, _>("asserted_at"))?
                .with_timezone(&Utc);
            let updated = DateTime::parse_from_rfc3339(&row.get::<String, _>("updated_at"))?
                .with_timezone(&Utc);
            let owner = row.get::<i64, _>("is_owner") != 0;
            let name: String = row.get("display_name");
            let pred: String = row.get("predicate");
            output.push(Fact {
                id: -1_000_000_000 - row.get::<i64, _>("id"),
                category: if owner { "user" } else { "people" }.to_string(),
                key: if owner {
                    pred
                } else {
                    format!("{name} · {pred}")
                },
                value: row
                    .get::<Option<String>, _>("display_value")
                    .unwrap_or_else(|| row.get("value")),
                source: format!("canonical:{}", row.get::<String, _>("source")),
                created_at: asserted,
                updated_at: updated,
                superseded_at: None,
                recall_count: 0,
                last_recalled_at: None,
                channel_id: row.get("channel_id"),
                privacy: FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                first_seen_at: Some(asserted),
                source_excerpt: None,
            });
        }
        let aliases = sqlx::query(
            "SELECT a.id, a.alias_type, a.value, a.source, a.asserted_at, a.updated_at,
                    a.channel_id, a.privacy, e.display_name, e.is_owner
             FROM memory_aliases a JOIN memory_entities e ON e.id = a.entity_id
             WHERE a.status = 'active' AND a.valid_to IS NULL AND e.status = 'active'",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in aliases {
            let asserted = DateTime::parse_from_rfc3339(&row.get::<String, _>("asserted_at"))?
                .with_timezone(&Utc);
            let updated = DateTime::parse_from_rfc3339(&row.get::<String, _>("updated_at"))?
                .with_timezone(&Utc);
            let owner = row.get::<i64, _>("is_owner") != 0;
            let name: String = row.get("display_name");
            let kind: String = row.get("alias_type");
            output.push(Fact {
                id: -2_000_000_000 - row.get::<i64, _>("id"),
                category: if owner { "user" } else { "people" }.to_string(),
                key: if owner {
                    kind
                } else {
                    format!("{name} · {kind}")
                },
                value: row.get("value"),
                source: format!("canonical:{}", row.get::<String, _>("source")),
                created_at: asserted,
                updated_at: updated,
                superseded_at: None,
                recall_count: 0,
                last_recalled_at: None,
                channel_id: row.get("channel_id"),
                privacy: FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                first_seen_at: Some(asserted),
                source_excerpt: None,
            });
        }
        let edges = sqlx::query(
            "SELECT r.id, r.relationship_type, r.source, r.asserted_at, r.updated_at,
                    r.channel_id, r.privacy, a.display_name source_name, b.display_name target_name
             FROM memory_relationships r
             JOIN memory_entities a ON a.id = r.source_entity_id
             JOIN memory_entities b ON b.id = r.target_entity_id
             WHERE r.status = 'active' AND r.valid_to IS NULL
               AND a.status = 'active' AND b.status = 'active'",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in edges {
            let asserted = DateTime::parse_from_rfc3339(&row.get::<String, _>("asserted_at"))?
                .with_timezone(&Utc);
            let updated = DateTime::parse_from_rfc3339(&row.get::<String, _>("updated_at"))?
                .with_timezone(&Utc);
            let rel: String = row.get("relationship_type");
            output.push(Fact {
                id: -3_000_000_000 - row.get::<i64, _>("id"),
                category: "relationships".to_string(),
                key: rel.clone(),
                value: format!(
                    "{} {} {}",
                    row.get::<String, _>("source_name"),
                    rel,
                    row.get::<String, _>("target_name")
                ),
                source: format!("canonical:{}", row.get::<String, _>("source")),
                created_at: asserted,
                updated_at: updated,
                superseded_at: None,
                recall_count: 0,
                last_recalled_at: None,
                channel_id: row.get("channel_id"),
                privacy: FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                first_seen_at: Some(asserted),
                source_excerpt: None,
            });
        }
        Ok(output)
    }

    /// Conservatively dual-writes only unambiguous owner-profile legacy keys.
    /// Relationship-shaped and compound-name facts stay untouched and pending.
    pub(crate) async fn backfill_structured_personal_memory(&self) -> anyhow::Result<usize> {
        // The legacy people table carries an explicit relationship field, so it
        // is safe to create separate entities and edges from those rows. We do
        // not infer relationships from names, co-residence, or notes.
        let people =
            sqlx::query("SELECT id, name, aliases_json, relationship FROM people ORDER BY id")
                .fetch_all(&self.pool)
                .await?;
        for person in people {
            let legacy_id: i64 = person.get("id");
            let name: String = person.get("name");
            let relationship: Option<String> = person.get("relationship");
            let relationship_key = relationship.as_deref().map(token);
            let local_id = format!("legacy_person_{legacy_id}");
            let is_owner = relationship_key.as_deref() == Some("owner");
            let mut write = PersonalMemoryWrite {
                entities: vec![
                    PersonalEntityCandidate {
                        local_id: "owner".to_string(),
                        entity_type: "person".to_string(),
                        canonical_name: if is_owner {
                            name.clone()
                        } else {
                            "Owner".to_string()
                        },
                        is_reference: false,
                        // A legacy people-row display name is often a preferred
                        // short name. Let an explicit profile `full_name` fact
                        // promote the canonical identity below.
                        canonical_name_confirmed: false,
                    },
                    PersonalEntityCandidate {
                        local_id: local_id.clone(),
                        entity_type: "person".to_string(),
                        canonical_name: name.clone(),
                        is_reference: false,
                        canonical_name_confirmed: true,
                    },
                ],
                ..Default::default()
            };
            if is_owner {
                write.entities.pop();
            } else {
                let aliases: Vec<String> =
                    serde_json::from_str(&person.get::<String, _>("aliases_json"))
                        .unwrap_or_default();
                write
                    .aliases
                    .extend(aliases.into_iter().map(|value| PersonalAliasCandidate {
                        entity_local_id: local_id.clone(),
                        value,
                        alias_type: "nickname".to_string(),
                    }));
                match relationship_key.as_deref() {
                    Some("father" | "mother" | "parent") => {
                        write.relationships.push(PersonalRelationshipCandidate {
                            source_local_id: local_id.clone(),
                            relationship_type: "PARENT_OF".to_string(),
                            target_local_id: "owner".to_string(),
                            valid_from: None,
                            valid_to: None,
                        });
                    }
                    Some("daughter" | "son" | "child") => {
                        write.relationships.push(PersonalRelationshipCandidate {
                            source_local_id: "owner".to_string(),
                            relationship_type: "PARENT_OF".to_string(),
                            target_local_id: local_id,
                            valid_from: None,
                            valid_to: None,
                        });
                    }
                    _ => {}
                }
            }
            self.reconcile_structured_personal_memory(
                &write,
                "legacy",
                Some("legacy people table"),
                None,
                FactPrivacy::Private,
            )
            .await?;
        }

        // Explicit relationship-shaped legacy facts are safe to project when
        // the predicate itself names the relationship. Parent/partner facts
        // may create a person from a single unambiguous name; child references
        // only attach to an already-known canonical name or alias so nicknames
        // are never guessed into merges.
        let relationship_facts = sqlx::query(
            "SELECT f.id, f.key, f.value, f.source, f.channel_id, f.privacy, f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile', 'family')
               AND lower(f.key) IN (
                 'dad_name', 'father_name', 'mom_name', 'mother_name',
                 'partner_name', 'spouse_name', 'daughter_name', 'daughter_name_2',
                 'son_name', 'son_name_2', 'child_name', 'oldest_daughter_name',
                 'youngest_daughter_name', 'oldest_son_name', 'youngest_son_name',
                 'daughter_names', 'son_names', 'children_names',
                 'has_daughter', 'has_son', 'has_children'
               )
               AND NOT EXISTS (
                 SELECT 1 FROM memory_write_audit a
                 WHERE a.operation = 'legacy_relationship_backfill'
                   AND a.source_fact_id = f.id
               )
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in relationship_facts {
            let fact_id: i64 = row.get("id");
            let key = token(&row.get::<String, _>("key"));
            let value: String = row.get("value");
            let is_parent = matches!(
                key.as_str(),
                "dad_name" | "father_name" | "mom_name" | "mother_name"
            );
            let is_partner = matches!(key.as_str(), "partner_name" | "spouse_name");
            let is_plural_child = matches!(
                key.as_str(),
                "daughter_names"
                    | "son_names"
                    | "children_names"
                    | "has_daughter"
                    | "has_son"
                    | "has_children"
            );
            let names = if is_plural_child {
                split_legacy_people(&value)
            } else {
                vec![value.clone()]
            };
            let direct = matches!(
                row.get::<String, _>("source").as_str(),
                "agent" | "user_stated" | "owner" | "owner-stated"
            );
            let mut wrote_any = false;

            for (index, raw_name) in names.into_iter().enumerate() {
                let Some((canonical_name, aliases)) = legacy_person_name(&raw_name) else {
                    let now = Utc::now().to_rfc3339();
                    sqlx::query(
                        "INSERT OR IGNORE INTO memory_resolution_reviews
                         (review_kind, normalized_reference, candidate_entity_ids_json,
                          payload_json, source, source_fact_id, status, created_at, updated_at)
                         VALUES ('ambiguous_legacy_relationship', ?, '[]', ?, 'legacy', ?,
                                 'pending', ?, ?)",
                    )
                    .bind(words(&raw_name))
                    .bind(serde_json::json!({"key": &key, "value": &value}).to_string())
                    .bind(fact_id)
                    .bind(&now)
                    .bind(&now)
                    .execute(&self.pool)
                    .await?;
                    continue;
                };

                let relative_id = format!("legacy_relative_{fact_id}_{index}");
                let relative_is_reference = !is_parent && !is_partner;
                let relationship_type = if is_parent {
                    "PARENT_OF"
                } else if key == "spouse_name" {
                    "SPOUSE_OF"
                } else if is_partner {
                    "PARTNER_OF"
                } else {
                    "PARENT_OF"
                };
                let mut write = PersonalMemoryWrite {
                    entities: vec![
                        PersonalEntityCandidate {
                            local_id: "owner".to_string(),
                            entity_type: "person".to_string(),
                            canonical_name: "Owner".to_string(),
                            is_reference: true,
                            canonical_name_confirmed: false,
                        },
                        PersonalEntityCandidate {
                            local_id: relative_id.clone(),
                            entity_type: "person".to_string(),
                            canonical_name,
                            is_reference: relative_is_reference,
                            canonical_name_confirmed: is_parent || is_partner,
                        },
                    ],
                    direct_user_statement: direct,
                    ..Default::default()
                };
                write
                    .aliases
                    .extend(aliases.into_iter().map(|value| PersonalAliasCandidate {
                        entity_local_id: relative_id.clone(),
                        value,
                        alias_type: "nickname".to_string(),
                    }));
                write.relationships.push(PersonalRelationshipCandidate {
                    source_local_id: if is_parent {
                        relative_id.clone()
                    } else {
                        "owner".to_string()
                    },
                    relationship_type: relationship_type.to_string(),
                    target_local_id: if is_parent {
                        "owner".to_string()
                    } else {
                        relative_id
                    },
                    valid_from: None,
                    valid_to: None,
                });
                let result = self
                    .reconcile_structured_personal_memory(
                        &write,
                        "legacy",
                        row.get::<Option<String>, _>("source_excerpt").as_deref(),
                        row.get::<Option<String>, _>("channel_id").as_deref(),
                        FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                    )
                    .await?;
                wrote_any |= result.created_relationships > 0 || result.confirmed_relationships > 0;
            }

            if wrote_any {
                sqlx::query(
                    "INSERT INTO memory_write_audit
                     (operation, record_type, record_id, source, source_fact_id, created_at)
                     VALUES ('legacy_relationship_backfill', 'legacy_fact', ?, 'legacy', ?, ?)",
                )
                .bind(fact_id)
                .bind(fact_id)
                .bind(Utc::now().to_rfc3339())
                .execute(&self.pool)
                .await?;
            }
        }

        // Promote an explicit child full-name record onto the already-known
        // child entity when the record itself names exactly one nickname.
        // Example: `daughter_isabella_full_name =
        // "Isabella (nickname Bella)"` can safely rename the existing Bella
        // child and retain Bella as a nickname. Alias-only similarities are
        // intentionally insufficient.
        let child_identity_facts = sqlx::query(
            "SELECT f.id, f.key, f.value, f.source, f.channel_id, f.privacy,
                    f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile', 'family')
               AND (
                    lower(f.key) LIKE 'daughter_%_full_name'
                    OR lower(f.key) LIKE 'son_%_full_name'
               )
               AND NOT EXISTS (
                    SELECT 1 FROM memory_write_audit a
                    WHERE a.operation = 'legacy_child_identity_backfill'
                      AND a.source_fact_id = f.id
               )
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in child_identity_facts {
            let fact_id: i64 = row.get("id");
            let value: String = row.get("value");
            let Some((display_name, aliases)) = legacy_person_name(&value) else {
                continue;
            };
            if aliases.len() != 1 {
                continue;
            }
            let alias = aliases[0].clone();
            let alias_normalized = words(&alias);
            let candidates = sqlx::query_scalar::<_, i64>(
                "SELECT DISTINCT child.id
                 FROM memory_entities owner
                 JOIN memory_relationships rel
                   ON rel.source_entity_id = owner.id
                  AND rel.relationship_type = 'PARENT_OF'
                  AND rel.status = 'active' AND rel.valid_to IS NULL
                 JOIN memory_entities child
                   ON child.id = rel.target_entity_id
                  AND child.entity_type = 'person' AND child.status = 'active'
                 WHERE owner.is_owner = 1 AND owner.status = 'active'
                   AND (
                       child.canonical_name = ?
                       OR EXISTS (
                           SELECT 1 FROM memory_aliases a
                           WHERE a.entity_id = child.id AND a.status = 'active'
                             AND a.valid_to IS NULL AND a.normalized_value = ?
                       )
                   )",
            )
            .bind(&alias_normalized)
            .bind(&alias_normalized)
            .fetch_all(&self.pool)
            .await?;
            if candidates.len() != 1 {
                let mut tx = self.pool.begin().await?;
                review(
                    &mut tx,
                    "ambiguous_legacy_child_identity",
                    &alias_normalized,
                    &candidates,
                    serde_json::json!({"legacy_fact_id": fact_id, "value": value}),
                    "legacy",
                )
                .await?;
                tx.commit().await?;
                continue;
            }

            let child_id = candidates[0];
            let canonical_name = words(&display_name);
            let collision = sqlx::query_scalar::<_, i64>(
                "SELECT id FROM memory_entities
                 WHERE entity_type = 'person' AND canonical_name = ? AND status = 'active'
                   AND id != ? LIMIT 1",
            )
            .bind(&canonical_name)
            .bind(child_id)
            .fetch_optional(&self.pool)
            .await?;
            if let Some(collision_id) = collision {
                let mut tx = self.pool.begin().await?;
                review(
                    &mut tx,
                    "legacy_child_name_collision",
                    &canonical_name,
                    &[child_id, collision_id],
                    serde_json::json!({"legacy_fact_id": fact_id, "value": value}),
                    "legacy",
                )
                .await?;
                tx.commit().await?;
                continue;
            }

            let direct = matches!(
                row.get::<String, _>("source").as_str(),
                "agent" | "user_stated" | "owner" | "owner-stated"
            );
            let privacy = FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")).to_string();
            let channel = row.get::<Option<String>, _>("channel_id");
            let provenance = row.get::<Option<String>, _>("source_excerpt");
            let mut result = PersonalMemoryWriteResult::default();
            let mut tx = self.pool.begin().await?;
            let current: String =
                sqlx::query_scalar("SELECT canonical_name FROM memory_entities WHERE id = ?")
                    .bind(child_id)
                    .fetch_one(&mut *tx)
                    .await?;
            if current != canonical_name {
                sqlx::query(
                    "UPDATE memory_entities SET canonical_name = ?, display_name = ?,
                     updated_at = ? WHERE id = ?",
                )
                .bind(&canonical_name)
                .bind(display_name.trim())
                .bind(Utc::now().to_rfc3339())
                .bind(child_id)
                .execute(&mut *tx)
                .await?;
                audit(
                    &mut tx,
                    "update",
                    Some(child_id),
                    "entity",
                    Some(child_id),
                    serde_json::json!({
                        "canonical_name": canonical_name,
                        "evidence": "explicit child full name with nickname"
                    }),
                    "legacy",
                    provenance.as_deref(),
                )
                .await?;
            }
            write_alias(
                &mut tx,
                child_id,
                &PersonalAliasCandidate {
                    entity_local_id: "child".to_string(),
                    value: alias,
                    alias_type: "nickname".to_string(),
                },
                "legacy",
                provenance.as_deref(),
                confidence(&row.get::<String, _>("source"), direct),
                direct,
                &privacy,
                channel.as_deref(),
                &mut result,
            )
            .await?;
            sqlx::query(
                "INSERT INTO memory_write_audit
                 (operation, entity_id, record_type, record_id, source, source_fact_id,
                  provenance, created_at)
                 VALUES ('legacy_child_identity_backfill', ?, 'legacy_fact', ?,
                         'legacy', ?, ?, ?)",
            )
            .bind(child_id)
            .bind(fact_id)
            .bind(fact_id)
            .bind(provenance.as_deref())
            .bind(Utc::now().to_rfc3339())
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
        }

        // A person-qualified legacy birthday key is safe to attach when its
        // name resolves to exactly one existing child canonical name or alias.
        // Generic keys such as `daughter_birthday` and ordinal keys remain
        // untouched because they do not identify a stable subject.
        let child_birth_facts = sqlx::query(
            "SELECT f.id, f.key, f.value, f.source, f.channel_id, f.privacy,
                    f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile', 'family')
               AND (
                    lower(f.key) LIKE 'daughter_%_birthday'
                    OR lower(f.key) LIKE 'daughter_%_birth_date'
                    OR lower(f.key) LIKE 'son_%_birthday'
                    OR lower(f.key) LIKE 'son_%_birth_date'
               )
               AND NOT EXISTS (
                    SELECT 1 FROM memory_write_audit a
                    WHERE a.operation = 'legacy_child_birth_backfill'
                      AND a.source_fact_id = f.id
               )
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in child_birth_facts {
            let fact_id: i64 = row.get("id");
            let key = token(&row.get::<String, _>("key"));
            let reference = ["daughter_", "son_"]
                .iter()
                .find_map(|prefix| key.strip_prefix(prefix))
                .and_then(|rest| {
                    rest.strip_suffix("_birthday")
                        .or_else(|| rest.strip_suffix("_birth_date"))
                })
                .map(str::to_string);
            let Some(reference) = reference else {
                continue;
            };
            if reference.is_empty()
                || matches!(
                    reference.as_str(),
                    "oldest" | "youngest" | "first" | "second"
                )
            {
                continue;
            }
            let value: String = row.get("value");
            if date_value(&value).is_none() {
                continue;
            }
            let candidates = sqlx::query_scalar::<_, i64>(
                "SELECT DISTINCT child.id
                 FROM memory_entities owner
                 JOIN memory_relationships rel
                   ON rel.source_entity_id = owner.id
                  AND rel.relationship_type = 'PARENT_OF'
                  AND rel.status = 'active' AND rel.valid_to IS NULL
                 JOIN memory_entities child
                   ON child.id = rel.target_entity_id
                  AND child.entity_type = 'person' AND child.status = 'active'
                 WHERE owner.is_owner = 1 AND owner.status = 'active'
                   AND (
                       child.canonical_name = ?
                       OR EXISTS (
                           SELECT 1 FROM memory_aliases a
                           WHERE a.entity_id = child.id AND a.status = 'active'
                             AND a.valid_to IS NULL AND a.normalized_value = ?
                       )
                   )",
            )
            .bind(&reference)
            .bind(&reference)
            .fetch_all(&self.pool)
            .await?;
            if candidates.len() != 1 {
                continue;
            }
            let child_id = candidates[0];
            let source: String = row.get("source");
            let direct = matches!(
                source.as_str(),
                "agent" | "user_stated" | "owner" | "owner-stated"
            );
            let privacy = FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")).to_string();
            let channel = row.get::<Option<String>, _>("channel_id");
            let provenance = row.get::<Option<String>, _>("source_excerpt");
            let mut result = PersonalMemoryWriteResult::default();
            let mut tx = self.pool.begin().await?;
            write_fact(
                &mut tx,
                child_id,
                &PersonalFactCandidate {
                    subject_local_id: "child".to_string(),
                    predicate: "birth_date".to_string(),
                    value,
                    display_value: None,
                    valid_from: None,
                    valid_to: None,
                },
                "legacy",
                provenance.as_deref(),
                confidence(&source, direct),
                direct,
                false,
                &privacy,
                channel.as_deref(),
                &mut result,
            )
            .await?;
            sqlx::query(
                "INSERT INTO memory_write_audit
                 (operation, entity_id, record_type, record_id, source, source_fact_id,
                  provenance, created_at)
                 VALUES ('legacy_child_birth_backfill', ?, 'legacy_fact', ?,
                         'legacy', ?, ?, ?)",
            )
            .bind(child_id)
            .bind(fact_id)
            .bind(fact_id)
            .bind(provenance.as_deref())
            .bind(Utc::now().to_rfc3339())
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
        }

        // Preserve explicitly stated plural co-residence without inventing a
        // plural canonical fact. The already-established children each receive
        // a typed LIVES_WITH edge; the symmetric inverse is materialized by
        // write_edge. Residence is not inferred from this pass.
        let co_residence_facts = sqlx::query(
            "SELECT f.id, f.source, f.channel_id, f.privacy, f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile', 'family')
               AND lower(f.key) IN (
                   'daughters_live_with_user', 'sons_live_with_user',
                   'children_live_with_user'
               )
               AND lower(f.value) LIKE '%live%with%'
               AND NOT EXISTS (
                    SELECT 1 FROM memory_write_audit a
                    WHERE a.operation = 'legacy_children_lives_with_backfill'
                      AND a.source_fact_id = f.id
               )
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in co_residence_facts {
            let fact_id: i64 = row.get("id");
            let owner_id: i64 = sqlx::query_scalar(
                "SELECT id FROM memory_entities
                 WHERE is_owner = 1 AND status = 'active' LIMIT 1",
            )
            .fetch_one(&self.pool)
            .await?;
            let child_ids = sqlx::query_scalar::<_, i64>(
                "SELECT target_entity_id FROM memory_relationships
                 WHERE source_entity_id = ? AND relationship_type = 'PARENT_OF'
                   AND status = 'active' AND valid_to IS NULL
                 ORDER BY target_entity_id",
            )
            .bind(owner_id)
            .fetch_all(&self.pool)
            .await?;
            let source: String = row.get("source");
            let direct = matches!(
                source.as_str(),
                "agent" | "user_stated" | "owner" | "owner-stated"
            );
            let privacy = FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")).to_string();
            let channel = row.get::<Option<String>, _>("channel_id");
            let provenance = row.get::<Option<String>, _>("source_excerpt");
            let mut result = PersonalMemoryWriteResult::default();
            let mut tx = self.pool.begin().await?;
            for child_id in child_ids {
                write_edge(
                    &mut tx,
                    child_id,
                    owner_id,
                    &PersonalRelationshipCandidate {
                        source_local_id: "child".to_string(),
                        relationship_type: "LIVES_WITH".to_string(),
                        target_local_id: "owner".to_string(),
                        valid_from: None,
                        valid_to: None,
                    },
                    "legacy",
                    provenance.as_deref(),
                    confidence(&source, direct),
                    direct,
                    &privacy,
                    channel.as_deref(),
                    &mut result,
                )
                .await?;
            }
            sqlx::query(
                "INSERT INTO memory_write_audit
                 (operation, entity_id, record_type, record_id, source, source_fact_id,
                  provenance, created_at)
                 VALUES ('legacy_children_lives_with_backfill', ?, 'legacy_fact', ?,
                         'legacy', ?, ?, ?)",
            )
            .bind(owner_id)
            .bind(fact_id)
            .bind(fact_id)
            .bind(provenance.as_deref())
            .bind(Utc::now().to_rfc3339())
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
        }

        // A legacy short-name row can predate a later, explicit full-name fact
        // that identifies the short name as a nickname (for example, a Person
        // named "Sol" followed by `Marisol "Sol" Vega`). Merge
        // only the narrow, auditable case where both entities have the same
        // complete active relationship set and the short-name entity carries
        // no independent aliases or facts. This is stronger than an alias-only
        // match and does not guess across ambiguous people.
        let merge_candidates = sqlx::query(
            "SELECT DISTINCT canonical.id AS canonical_id, duplicate.id AS duplicate_id,
                    alias.normalized_value
             FROM memory_aliases alias
             JOIN memory_entities canonical ON canonical.id = alias.entity_id
             JOIN memory_entities duplicate
               ON duplicate.entity_type = canonical.entity_type
              AND duplicate.canonical_name = alias.normalized_value
              AND duplicate.id != canonical.id
             WHERE alias.alias_type = 'nickname' AND alias.status = 'active'
               AND canonical.status = 'active' AND duplicate.status = 'active'
               AND NOT EXISTS (
                   SELECT 1 FROM memory_aliases duplicate_alias
                   WHERE duplicate_alias.entity_id = duplicate.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_entity_facts duplicate_fact
                   WHERE duplicate_fact.subject_entity_id = duplicate.id
               )
               AND EXISTS (
                   SELECT 1 FROM memory_relationships duplicate_rel
                   WHERE duplicate_rel.status = 'active'
                     AND (duplicate_rel.source_entity_id = duplicate.id
                          OR duplicate_rel.target_entity_id = duplicate.id)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_relationships duplicate_rel
                   WHERE duplicate_rel.status = 'active'
                     AND (duplicate_rel.source_entity_id = duplicate.id
                          OR duplicate_rel.target_entity_id = duplicate.id)
                     AND NOT EXISTS (
                         SELECT 1 FROM memory_relationships canonical_rel
                         WHERE canonical_rel.status = 'active'
                           AND canonical_rel.relationship_type =
                               duplicate_rel.relationship_type
                           AND (
                               (duplicate_rel.source_entity_id = duplicate.id
                                AND canonical_rel.source_entity_id = canonical.id
                                AND canonical_rel.target_entity_id =
                                    duplicate_rel.target_entity_id)
                               OR
                               (duplicate_rel.target_entity_id = duplicate.id
                                AND canonical_rel.target_entity_id = canonical.id
                                AND canonical_rel.source_entity_id =
                                    duplicate_rel.source_entity_id)
                           )
                     )
               )",
        )
        .fetch_all(&self.pool)
        .await?;
        for candidate in merge_candidates {
            let canonical_id: i64 = candidate.get("canonical_id");
            let duplicate_id: i64 = candidate.get("duplicate_id");
            let normalized_value: String = candidate.get("normalized_value");
            let now = Utc::now().to_rfc3339();
            let mut tx = self.pool.begin().await?;
            sqlx::query(
                "UPDATE memory_relationships
                 SET status = 'superseded', valid_to = ?, updated_at = ?
                 WHERE status = 'active'
                   AND (source_entity_id = ? OR target_entity_id = ?)",
            )
            .bind(&now)
            .bind(&now)
            .bind(duplicate_id)
            .bind(duplicate_id)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "UPDATE memory_entities
                 SET status = 'merged', merged_into_entity_id = ?, updated_at = ?
                 WHERE id = ? AND status = 'active'",
            )
            .bind(canonical_id)
            .bind(&now)
            .bind(duplicate_id)
            .execute(&mut *tx)
            .await?;
            audit(
                &mut tx,
                "merge",
                Some(duplicate_id),
                "entity",
                Some(duplicate_id),
                serde_json::json!({
                    "merged_into_entity_id": canonical_id,
                    "evidence": "explicit nickname plus identical relationships",
                    "nickname": normalized_value,
                }),
                "legacy",
                Some("legacy relationship backfill reconciliation"),
            )
            .await?;
            tx.commit().await?;
        }

        // Malformed combined parent identities are preserved but explicitly
        // quarantined for review; they are never parsed into one Person.
        let compound_parents = sqlx::query(
            "SELECT id, key, value FROM facts
             WHERE superseded_at IS NULL
               AND lower(key) IN ('father_name', 'mother_name', 'parent_name', 'parents')
               AND (lower(value) LIKE '% and %' OR value LIKE '%&%')",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in compound_parents {
            let id: i64 = row.get("id");
            let now = Utc::now().to_rfc3339();
            sqlx::query(
                "INSERT OR IGNORE INTO memory_resolution_reviews
                 (review_kind, normalized_reference, candidate_entity_ids_json, payload_json,
                  source, source_fact_id, status, created_at, updated_at)
                 VALUES ('compound_parent_identity', ?, '[]', ?, 'legacy', ?, 'pending', ?, ?)",
            )
            .bind(words(&row.get::<String, _>("value")))
            .bind(
                serde_json::json!({"legacy_fact_id": id, "key": row.get::<String, _>("key")})
                    .to_string(),
            )
            .bind(id)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }

        let rows = sqlx::query(
            "SELECT f.id, f.key, f.value, f.source, f.channel_id, f.privacy, f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile')
               AND lower(f.key) IN (
                 'name', 'full_name', 'preferred_name', 'nickname', 'username',
                 'online_nickname', 'online_handle', 'handle', 'birthday',
                 'birth_date', 'date_of_birth', 'residence', 'current_residence',
                 'birthplace', 'place_of_birth'
               )
               AND NOT EXISTS (
                 SELECT 1 FROM memory_write_audit a
                 WHERE a.operation = 'legacy_backfill' AND a.source_fact_id = f.id
               )
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut migrated = 0;
        for row in rows {
            let id: i64 = row.get("id");
            let key: String = row.get("key");
            let value: String = row.get("value");
            let normalized_key = predicate(&key);
            if matches!(normalized_key.as_str(), "name" | "full_name")
                && (value.to_lowercase().contains(" and ") || value.contains('&'))
            {
                let now = Utc::now().to_rfc3339();
                sqlx::query(
                    "INSERT OR IGNORE INTO memory_resolution_reviews
                     (review_kind, normalized_reference, candidate_entity_ids_json, payload_json,
                      source, source_fact_id, status, created_at, updated_at)
                     VALUES ('compound_legacy_name', ?, '[]', ?, 'legacy', ?, 'pending', ?, ?)",
                )
                .bind(words(&value))
                .bind(serde_json::json!({"legacy_fact_id": id}).to_string())
                .bind(id)
                .bind(&now)
                .bind(&now)
                .execute(&self.pool)
                .await?;
                continue;
            }
            let direct = matches!(
                row.get::<String, _>("source").as_str(),
                "agent" | "user_stated" | "owner" | "owner-stated"
            );
            let mut write = PersonalMemoryWrite {
                entities: vec![PersonalEntityCandidate {
                    local_id: "owner".to_string(),
                    entity_type: "person".to_string(),
                    canonical_name: if matches!(normalized_key.as_str(), "name" | "full_name") {
                        value.clone()
                    } else {
                        "Owner".to_string()
                    },
                    is_reference: false,
                    canonical_name_confirmed: matches!(
                        normalized_key.as_str(),
                        "name" | "full_name"
                    ),
                }],
                direct_user_statement: direct,
                ..Default::default()
            };
            match normalized_key.as_str() {
                "name" | "full_name" => write.aliases.push(PersonalAliasCandidate {
                    entity_local_id: "owner".to_string(),
                    value: value.clone(),
                    alias_type: "legal_name_variant".to_string(),
                }),
                "preferred_name" | "nickname" => write.aliases.push(PersonalAliasCandidate {
                    entity_local_id: "owner".to_string(),
                    value: value.clone(),
                    alias_type: normalized_key,
                }),
                "username" | "online_nickname" | "online_handle" | "handle" => {
                    write.aliases.push(PersonalAliasCandidate {
                        entity_local_id: "owner".to_string(),
                        value: value.clone(),
                        alias_type: "online_handle".to_string(),
                    })
                }
                _ => write.facts.push(PersonalFactCandidate {
                    subject_local_id: "owner".to_string(),
                    predicate: normalized_key,
                    value: value.clone(),
                    display_value: None,
                    valid_from: None,
                    valid_to: None,
                }),
            }
            let result = self
                .reconcile_structured_personal_memory(
                    &write,
                    "legacy",
                    row.get::<Option<String>, _>("source_excerpt").as_deref(),
                    row.get::<Option<String>, _>("channel_id").as_deref(),
                    FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                )
                .await;
            if result.is_ok() {
                sqlx::query(
                    "INSERT INTO memory_write_audit
                     (operation, record_type, record_id, source, source_fact_id, created_at)
                     VALUES ('legacy_backfill', 'legacy_fact', ?, 'legacy', ?, ?)",
                )
                .bind(id)
                .bind(id)
                .bind(Utc::now().to_rfc3339())
                .execute(&self.pool)
                .await?;
                migrated += 1;
            }
        }

        // Earlier versions may already have marked every profile fact as
        // migrated while leaving a preferred short name as the owner
        // canonical name. Re-evaluate only explicit `full_name` facts and
        // promote the strongest direct statement once. A startup does not
        // refresh confirmation timestamps when the canonical name is already
        // correct.
        if let Some(row) = sqlx::query(
            "SELECT f.value, f.source, f.channel_id, f.privacy, f.source_excerpt
             FROM facts f
             WHERE f.superseded_at IS NULL
               AND lower(f.category) IN ('user', 'personal', 'profile')
               AND lower(f.key) = 'full_name'
               AND lower(f.value) NOT LIKE '% and %' AND f.value NOT LIKE '%&%'
             ORDER BY
               CASE WHEN f.source IN ('user_stated', 'owner', 'owner-stated', 'agent')
                    THEN 0 ELSE 1 END,
               length(f.value) DESC, f.id DESC
             LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?
        {
            let value: String = row.get("value");
            let expected = words(&value);
            let current: Option<String> = sqlx::query_scalar(
                "SELECT canonical_name FROM memory_entities
                 WHERE is_owner = 1 AND status = 'active' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await?;
            if current.as_deref() != Some(expected.as_str()) {
                let source: String = row.get("source");
                self.reconcile_structured_personal_memory(
                    &PersonalMemoryWrite {
                        entities: vec![PersonalEntityCandidate {
                            local_id: "owner".to_string(),
                            entity_type: "person".to_string(),
                            canonical_name: value.clone(),
                            is_reference: false,
                            canonical_name_confirmed: true,
                        }],
                        aliases: vec![PersonalAliasCandidate {
                            entity_local_id: "owner".to_string(),
                            value,
                            alias_type: "legal_name_variant".to_string(),
                        }],
                        direct_user_statement: matches!(
                            source.as_str(),
                            "user_stated" | "owner" | "owner-stated" | "agent"
                        ),
                        ..Default::default()
                    },
                    "legacy",
                    row.get::<Option<String>, _>("source_excerpt").as_deref(),
                    row.get::<Option<String>, _>("channel_id").as_deref(),
                    FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy")),
                )
                .await?;
            }
        }
        Ok(migrated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::FactStore;
    use std::sync::Arc;

    async fn store() -> (tempfile::TempDir, SqliteStateStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("structured-memory.db");
        let embeddings = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(path.to_str().unwrap(), 20, None, embeddings)
            .await
            .unwrap();
        (dir, store)
    }

    fn entity(local_id: &str, canonical_name: &str, is_reference: bool) -> PersonalEntityCandidate {
        PersonalEntityCandidate {
            local_id: local_id.to_string(),
            entity_type: "person".to_string(),
            canonical_name: canonical_name.to_string(),
            is_reference,
            canonical_name_confirmed: !is_reference,
        }
    }

    fn alias(local_id: &str, value: &str, kind: &str) -> PersonalAliasCandidate {
        PersonalAliasCandidate {
            entity_local_id: local_id.to_string(),
            value: value.to_string(),
            alias_type: kind.to_string(),
        }
    }

    fn fact(local_id: &str, predicate: &str, value: &str) -> PersonalFactCandidate {
        PersonalFactCandidate {
            subject_local_id: local_id.to_string(),
            predicate: predicate.to_string(),
            value: value.to_string(),
            display_value: None,
            valid_from: None,
            valid_to: None,
        }
    }

    fn edge(from: &str, relation: &str, to: &str) -> PersonalRelationshipCandidate {
        PersonalRelationshipCandidate {
            source_local_id: from.to_string(),
            relationship_type: relation.to_string(),
            target_local_id: to.to_string(),
            valid_from: None,
            valid_to: None,
        }
    }

    #[test]
    fn dates_share_one_canonical_form() {
        assert_eq!(date_value("Feb 4 1988").as_deref(), Some("1988-02-04"));
        assert_eq!(
            date_value("February 4, 1988").as_deref(),
            Some("1988-02-04")
        );
        assert_eq!(date_value("1988-02-04").as_deref(), Some("1988-02-04"));
    }

    #[test]
    fn handles_are_case_insensitive() {
        assert_eq!(alias_value("@RiveraDev", "online_handle"), "riveradev");
    }

    #[test]
    fn inverse_family_edges_are_explicit() {
        assert_eq!(inverse("PARENT_OF"), Some("CHILD_OF"));
        assert_eq!(inverse("LIVES_WITH"), Some("LIVES_WITH"));
    }

    #[tokio::test]
    async fn owner_identity_aliases_and_handle_are_idempotent() {
        let (_dir, store) = store().await;
        let write = PersonalMemoryWrite {
            entities: vec![entity("owner", "Alice Morgan Rivera", false)],
            aliases: vec![
                alias("owner", "Morgan", "preferred_name"),
                alias("owner", "rivera_dev_42", "online_handle"),
            ],
            direct_user_statement: true,
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &write,
                "agent",
                Some("synthetic owner statement"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        store
            .reconcile_structured_personal_memory(
                &write,
                "agent",
                Some("synthetic owner statement"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();

        let owners: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities WHERE is_owner = 1 AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let canonical: String =
            sqlx::query_scalar("SELECT canonical_name FROM memory_entities WHERE is_owner = 1")
                .fetch_one(&store.pool)
                .await
                .unwrap();
        let aliases: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_aliases
             WHERE entity_id = (SELECT id FROM memory_entities WHERE is_owner = 1)",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let separate_preferred_person: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE entity_type = 'person' AND canonical_name = 'morgan' AND is_owner = 0",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(owners, 1);
        assert_eq!(canonical, "alice morgan rivera");
        assert_eq!(aliases, 2);
        assert_eq!(separate_preferred_person, 0);
    }

    #[tokio::test]
    async fn children_aliases_dates_and_inverse_relationships_are_canonical() {
        let (_dir, store) = store().await;
        let first = PersonalMemoryWrite {
            entities: vec![
                entity("owner", "Owner", false),
                entity("child_one", "Alexandra River", false),
                entity("child_two", "Jordan River", false),
            ],
            aliases: vec![
                alias("child_one", "Lexi", "nickname"),
                alias("child_two", "Jori", "nickname"),
            ],
            facts: vec![
                fact("child_one", "birth_date", "April 3, 2019"),
                fact("child_two", "birth_date", "November 12 2011"),
            ],
            relationships: vec![
                edge("owner", "PARENT_OF", "child_one"),
                edge("owner", "PARENT_OF", "child_two"),
            ],
            direct_user_statement: true,
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &first,
                "agent",
                Some("synthetic family statement"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();

        let later = PersonalMemoryWrite {
            entities: vec![entity("child", "Lexi", true)],
            facts: vec![fact("child", "favorite_color", "violet")],
            direct_user_statement: true,
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &later,
                "agent",
                Some("synthetic alias reuse"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        let residence = PersonalMemoryWrite {
            entities: vec![
                entity("owner", "Owner", false),
                entity("first", "Lexi", true),
                entity("second", "Jori", true),
                PersonalEntityCandidate {
                    local_id: "city".to_string(),
                    entity_type: "place".to_string(),
                    canonical_name: "Riverton, VA".to_string(),
                    is_reference: false,
                    canonical_name_confirmed: true,
                },
            ],
            relationships: vec![
                edge("first", "LIVES_WITH", "owner"),
                edge("second", "LIVES_WITH", "owner"),
                edge("owner", "LIVES_IN", "city"),
            ],
            direct_user_statement: true,
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &residence,
                "agent",
                Some("synthetic shared residence"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        // Repeat the relationship-bearing statement to verify full idempotency.
        store
            .reconcile_structured_personal_memory(
                &first,
                "agent",
                Some("synthetic family statement"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();

        let people: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities WHERE entity_type = 'person' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let birth_dates: Vec<String> = sqlx::query_scalar(
            "SELECT normalized_value FROM memory_entity_facts
             WHERE predicate = 'birth_date' AND status = 'active' ORDER BY normalized_value",
        )
        .fetch_all(&store.pool)
        .await
        .unwrap();
        let parent_edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let child_edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE relationship_type = 'CHILD_OF' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let nickname_entities: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities WHERE canonical_name = 'lexi'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let child_lives_with_owner: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships r
             JOIN memory_entities source ON source.id = r.source_entity_id
             JOIN memory_entities target ON target.id = r.target_entity_id
             WHERE r.relationship_type = 'LIVES_WITH' AND source.is_owner = 0
               AND target.is_owner = 1 AND r.status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let owner_lives_in: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships r
             JOIN memory_entities source ON source.id = r.source_entity_id
             WHERE r.relationship_type = 'LIVES_IN' AND source.is_owner = 1
               AND r.status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(people, 3);
        assert_eq!(birth_dates, vec!["2011-11-12", "2019-04-03"]);
        assert_eq!((parent_edges, child_edges), (2, 2));
        assert_eq!(nickname_entities, 0);
        assert_eq!(child_lives_with_owner, 2);
        assert_eq!(owner_lives_in, 1);
    }

    #[tokio::test]
    async fn direct_correction_supersedes_legacy_and_retrieval_prefers_active() {
        let (_dir, store) = store().await;
        let legacy = PersonalMemoryWrite {
            entities: vec![entity("owner", "Owner", false)],
            facts: vec![fact("owner", "birth_date", "March 6 1991")],
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &legacy,
                "legacy",
                Some("synthetic import"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        store
            .upsert_fact(
                "user",
                "birthday",
                "1991-03-06",
                "legacy",
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        let correction = PersonalMemoryWrite {
            entities: vec![entity("owner", "Owner", false)],
            facts: vec![fact("owner", "birth_date", "February 4 1988")],
            direct_user_statement: true,
            ..Default::default()
        };
        let result = store
            .reconcile_structured_personal_memory(
                &correction,
                "agent",
                Some("synthetic correction"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        assert_eq!(result.superseded_facts, 1);

        let active: String = sqlx::query_scalar(
            "SELECT normalized_value FROM memory_entity_facts
             WHERE predicate = 'birth_date' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let superseded: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entity_facts
             WHERE predicate = 'birth_date' AND status = 'superseded'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let recalled = store
            .get_relevant_facts("what is my birth date", 10)
            .await
            .unwrap();
        assert_eq!(active, "1988-02-04");
        assert_eq!(superseded, 1);
        assert!(recalled.iter().any(|fact| {
            fact.source.starts_with("canonical:") && fact.value == "February 4 1988"
        }));
        assert!(!recalled.iter().any(|fact| fact.value == "1991-03-06"));
    }

    #[tokio::test]
    async fn confirmed_conflicts_and_ambiguous_aliases_require_review() {
        let (_dir, store) = store().await;
        let initial = PersonalMemoryWrite {
            entities: vec![
                entity("owner", "Owner", false),
                entity("one", "Jordan Lee", false),
                entity("two", "Jordan Patel", false),
            ],
            aliases: vec![
                alias("one", "Jo", "nickname"),
                alias("two", "Jo", "nickname"),
            ],
            facts: vec![fact("owner", "birth_date", "2000-01-01")],
            direct_user_statement: true,
            ..Default::default()
        };
        store
            .reconcile_structured_personal_memory(
                &initial,
                "agent",
                Some("synthetic confirmed facts"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        let conflict = PersonalMemoryWrite {
            entities: vec![entity("owner", "Owner", false)],
            facts: vec![fact("owner", "birth_date", "2001-01-01")],
            direct_user_statement: true,
            correction: false,
            ..Default::default()
        };
        let conflict_result = store
            .reconcile_structured_personal_memory(
                &conflict,
                "agent",
                Some("synthetic conflicting statement"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        let ambiguous = PersonalMemoryWrite {
            entities: vec![entity("reference", "Jo", true)],
            facts: vec![fact("reference", "city", "Raleigh")],
            direct_user_statement: true,
            ..Default::default()
        };
        let ambiguous_result = store
            .reconcile_structured_personal_memory(
                &ambiguous,
                "agent",
                Some("synthetic ambiguous reference"),
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        let reviews: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_resolution_reviews WHERE status = 'pending'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(conflict_result.disputed_facts, 1);
        assert!(!ambiguous_result.unresolved.is_empty());
        assert_eq!(reviews, 2);
    }

    #[tokio::test]
    async fn legacy_parent_records_backfill_separately_and_quarantine_compound_fact() {
        let (_dir, store) = store().await;
        let now = Utc::now().to_rfc3339();
        for (name, relationship, aliases) in [
            ("Alice Rivera", "owner", "[]"),
            ("Marco Rivera", "father", "[]"),
            ("Sofia Mendez", "mother", "[\"Sofi\"]"),
        ] {
            sqlx::query(
                "INSERT INTO people
                 (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
                 VALUES (?, ?, ?, '{}', ?, ?)",
            )
            .bind(name)
            .bind(aliases)
            .bind(relationship)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }
        sqlx::query(
            "INSERT INTO facts
             (category, key, value, source, created_at, updated_at, privacy)
             VALUES ('user', 'father_name', 'Marco Rivera and Sofia Mendez (Sofi)',
                     'legacy', ?, ?, 'private')",
        )
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();

        store.backfill_structured_personal_memory().await.unwrap();
        let parents: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let combined_entity: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE canonical_name = 'marco rivera and sofia mendez sofi'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let reviews: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_resolution_reviews
             WHERE review_kind = 'compound_parent_identity' AND status = 'pending'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(parents, 2);
        assert_eq!(combined_entity, 0);
        assert_eq!(reviews, 1);
    }

    #[tokio::test]
    async fn explicit_flat_relationship_facts_backfill_idempotently() {
        let (_dir, store) = store().await;
        let now = Utc::now().to_rfc3339();
        for (name, relationship) in [
            ("Alice Rivera", "owner"),
            ("Mia Rivera", ""),
            ("Zoe Rivera", ""),
        ] {
            sqlx::query(
                "INSERT INTO people
                 (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
                 VALUES (?, '[]', ?, '{}', ?, ?)",
            )
            .bind(name)
            .bind(relationship)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }
        for (key, value) in [
            ("dad_name", "Jordan Lee"),
            ("mother_name", "Alexandra \"Lexi\" Rivera"),
            ("daughter_names", "Mia Rivera and Zoe Rivera"),
        ] {
            sqlx::query(
                "INSERT INTO facts
                 (category, key, value, source, created_at, updated_at, privacy)
                 VALUES ('user', ?, ?, 'user_stated', ?, ?, 'private')",
            )
            .bind(key)
            .bind(value)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }

        store.backfill_structured_personal_memory().await.unwrap();
        store.backfill_structured_personal_memory().await.unwrap();

        let parent_edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let child_edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE relationship_type = 'CHILD_OF' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let nickname_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_aliases
             WHERE alias_type = 'nickname' AND normalized_value = 'lexi'
               AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!((parent_edges, child_edges), (4, 4));
        assert_eq!(nickname_count, 1);
    }

    #[tokio::test]
    async fn explicit_nickname_and_matching_relationship_merge_legacy_short_name_entity() {
        let (_dir, store) = store().await;
        let now = Utc::now().to_rfc3339();
        for (name, relationship) in [("Alice Rivera", "owner"), ("Sol", "mother")] {
            sqlx::query(
                "INSERT INTO people
                 (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
                 VALUES (?, '[]', ?, '{}', ?, ?)",
            )
            .bind(name)
            .bind(relationship)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }
        sqlx::query(
            "INSERT INTO facts
             (category, key, value, source, created_at, updated_at, privacy)
             VALUES ('user', 'mother_name', 'Marisol Vega \"Sol\"',
                     'user_stated', ?, ?, 'private')",
        )
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();

        store.backfill_structured_personal_memory().await.unwrap();
        store.backfill_structured_personal_memory().await.unwrap();

        let active_people: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE status = 'active'
               AND canonical_name IN ('sol', 'marisol vega')",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let merged_people: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE status = 'merged' AND canonical_name = 'sol'
               AND merged_into_entity_id IS NOT NULL",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let active_parent_edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships
             WHERE status = 'active' AND relationship_type = 'PARENT_OF'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(active_people, 1);
        assert_eq!(merged_people, 1);
        assert_eq!(active_parent_edges, 1);
    }

    #[tokio::test]
    async fn legacy_full_names_promote_short_owner_and_child_names_idempotently() {
        let (_dir, store) = store().await;
        let now = Utc::now().to_rfc3339();
        for (name, relationship) in [("Morgan", "owner"), ("Zo", "daughter")] {
            sqlx::query(
                "INSERT INTO people
                 (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
                 VALUES (?, '[]', ?, '{}', ?, ?)",
            )
            .bind(name)
            .bind(relationship)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }
        for (key, value) in [
            ("full_name", "Alice Morgan Rivera"),
            ("daughter_zoe_full_name", "Zoe (nickname Zo)"),
            ("daughter_zo_birthday", "April 12, 2015"),
            (
                "daughters_live_with_user",
                "Both daughters live with me in Exampletown, VA",
            ),
        ] {
            sqlx::query(
                "INSERT INTO facts
                 (category, key, value, source, created_at, updated_at, privacy)
                 VALUES ('user', ?, ?, 'user_stated', ?, ?, 'private')",
            )
            .bind(key)
            .bind(value)
            .bind(&now)
            .bind(&now)
            .execute(&store.pool)
            .await
            .unwrap();
        }

        store.backfill_structured_personal_memory().await.unwrap();
        store.backfill_structured_personal_memory().await.unwrap();

        let owner_name: String = sqlx::query_scalar(
            "SELECT canonical_name FROM memory_entities
             WHERE is_owner = 1 AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let zoe_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE canonical_name = 'zoe' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let zo_entity_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_entities
             WHERE canonical_name = 'zo' AND status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let zo_alias_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_aliases a
             JOIN memory_entities e ON e.id = a.entity_id
             WHERE e.canonical_name = 'zoe' AND e.status = 'active'
               AND a.alias_type = 'nickname' AND a.normalized_value = 'zo'
               AND a.status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let zoe_birth_date: String = sqlx::query_scalar(
            "SELECT f.normalized_value FROM memory_entity_facts f
             JOIN memory_entities e ON e.id = f.subject_entity_id
             WHERE e.canonical_name = 'zoe' AND e.status = 'active'
               AND f.predicate = 'birth_date' AND f.status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        let child_lives_with_owner: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_relationships r
             JOIN memory_entities child ON child.id = r.source_entity_id
             JOIN memory_entities owner ON owner.id = r.target_entity_id
             WHERE child.canonical_name = 'zoe' AND child.status = 'active'
               AND owner.is_owner = 1 AND owner.status = 'active'
               AND r.relationship_type = 'LIVES_WITH' AND r.status = 'active'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(owner_name, "alice morgan rivera");
        assert_eq!(zoe_count, 1);
        assert_eq!(zo_entity_count, 0);
        assert_eq!(zo_alias_count, 1);
        assert_eq!(zoe_birth_date, "2015-04-12");
        assert_eq!(child_lives_with_owner, 1);
    }
}
