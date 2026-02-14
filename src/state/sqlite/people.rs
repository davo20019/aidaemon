use super::*;

#[async_trait]
impl crate::traits::PeopleStore for SqliteStateStore {
    async fn upsert_person(&self, person: &crate::traits::Person) -> anyhow::Result<i64> {
        let aliases_json = serde_json::to_string(&person.aliases)?;
        let platform_ids_json = serde_json::to_string(&person.platform_ids)?;
        let now = chrono::Utc::now().to_rfc3339();

        if person.id > 0 {
            sqlx::query(
                "UPDATE people SET name = ?, aliases_json = ?, relationship = ?, platform_ids_json = ?, \
                 notes = ?, communication_style = ?, language_preference = ?, updated_at = ? WHERE id = ?"
            )
            .bind(&person.name)
            .bind(&aliases_json)
            .bind(&person.relationship)
            .bind(&platform_ids_json)
            .bind(&person.notes)
            .bind(&person.communication_style)
            .bind(&person.language_preference)
            .bind(&now)
            .bind(person.id)
            .execute(&self.pool)
            .await?;
            Ok(person.id)
        } else {
            // Use INSERT OR IGNORE to handle the unique index on LOWER(name).
            // If a duplicate exists, the INSERT silently does nothing and we
            // return the existing row's id.
            sqlx::query(
                "INSERT OR IGNORE INTO people (name, aliases_json, relationship, platform_ids_json, notes, \
                 communication_style, language_preference, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&person.name)
            .bind(&aliases_json)
            .bind(&person.relationship)
            .bind(&platform_ids_json)
            .bind(&person.notes)
            .bind(&person.communication_style)
            .bind(&person.language_preference)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;

            // Fetch the id — works whether we just inserted or the row already existed
            let row = sqlx::query("SELECT id FROM people WHERE LOWER(name) = ?")
                .bind(person.name.to_lowercase())
                .fetch_one(&self.pool)
                .await?;
            Ok(row.get::<i64, _>("id"))
        }
    }

    async fn get_person(&self, id: i64) -> anyhow::Result<Option<crate::traits::Person>> {
        let row = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Self::row_to_person(&r)))
    }

    async fn get_person_by_platform_id(
        &self,
        platform_id: &str,
    ) -> anyhow::Result<Option<crate::traits::Person>> {
        // Search platform_ids_json for a key matching the platform_id
        // SQLite json_each lets us iterate JSON object keys
        let row = sqlx::query(
            "SELECT p.id, p.name, p.aliases_json, p.relationship, p.platform_ids_json, p.notes, \
             p.communication_style, p.language_preference, p.last_interaction_at, p.interaction_count, \
             p.created_at, p.updated_at \
             FROM people p, json_each(p.platform_ids_json) j \
             WHERE j.key = ?"
        )
        .bind(platform_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Self::row_to_person(&r)))
    }

    async fn find_person_by_name(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<crate::traits::Person>> {
        let name_lower = name.to_lowercase();
        // Check name first (case-insensitive)
        let row = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people WHERE LOWER(name) = ?",
        )
        .bind(&name_lower)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            return Ok(Some(Self::row_to_person(&r)));
        }

        // Check aliases (JSON array search)
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people",
        )
        .fetch_all(&self.pool)
        .await?;

        for r in &rows {
            let aliases_str: String = r.get("aliases_json");
            if let Ok(aliases) = serde_json::from_str::<Vec<String>>(&aliases_str) {
                if aliases.iter().any(|a| a.to_lowercase() == name_lower) {
                    return Ok(Some(Self::row_to_person(r)));
                }
            }
        }

        Ok(None)
    }

    async fn get_all_people(&self) -> anyhow::Result<Vec<crate::traits::Person>> {
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people ORDER BY name ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_person).collect())
    }

    async fn delete_person(&self, id: i64) -> anyhow::Result<()> {
        // person_facts has ON DELETE CASCADE, but be explicit
        sqlx::query("DELETE FROM person_facts WHERE person_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM people WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn link_platform_id(
        &self,
        person_id: i64,
        platform_id: &str,
        display_name: &str,
    ) -> anyhow::Result<()> {
        // Read current platform_ids, add new one, write back
        let row = sqlx::query("SELECT platform_ids_json FROM people WHERE id = ?")
            .bind(person_id)
            .fetch_optional(&self.pool)
            .await?;

        let mut ids: std::collections::HashMap<String, String> = match row {
            Some(r) => {
                let json_str: String = r.get("platform_ids_json");
                serde_json::from_str(&json_str).unwrap_or_default()
            }
            None => return Err(anyhow::anyhow!("Person {} not found", person_id)),
        };

        ids.insert(platform_id.to_string(), display_name.to_string());
        let updated_json = serde_json::to_string(&ids)?;
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query("UPDATE people SET platform_ids_json = ?, updated_at = ? WHERE id = ?")
            .bind(&updated_json)
            .bind(&now)
            .bind(person_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn touch_person_interaction(&self, person_id: i64) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE people SET last_interaction_at = ?, interaction_count = interaction_count + 1, updated_at = ? WHERE id = ?"
        )
        .bind(&now)
        .bind(&now)
        .bind(person_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn upsert_person_fact(
        &self,
        person_id: i64,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        confidence: f32,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO person_facts (person_id, category, key, value, source, confidence, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(person_id, category, key) DO UPDATE SET \
             value = excluded.value, source = excluded.source, confidence = excluded.confidence, updated_at = excluded.updated_at"
        )
        .bind(person_id)
        .bind(category)
        .bind(key)
        .bind(value)
        .bind(source)
        .bind(confidence)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_person_facts(
        &self,
        person_id: i64,
        category: Option<&str>,
    ) -> anyhow::Result<Vec<crate::traits::PersonFact>> {
        let rows = if let Some(cat) = category {
            sqlx::query(
                "SELECT id, person_id, category, key, value, source, confidence, created_at, updated_at \
                 FROM person_facts WHERE person_id = ? AND category = ? ORDER BY category, key"
            )
            .bind(person_id)
            .bind(cat)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, person_id, category, key, value, source, confidence, created_at, updated_at \
                 FROM person_facts WHERE person_id = ? ORDER BY category, key"
            )
            .bind(person_id)
            .fetch_all(&self.pool)
            .await?
        };

        Ok(rows.iter().map(Self::row_to_person_fact).collect())
    }

    async fn delete_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM person_facts WHERE id = ?")
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn confirm_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("UPDATE person_facts SET confidence = 1.0, source = 'owner', updated_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_people_with_upcoming_dates(
        &self,
        within_days: i32,
    ) -> anyhow::Result<Vec<(crate::traits::Person, crate::traits::PersonFact)>> {
        // Get all birthday/important_date facts, then filter by upcoming date in Rust
        let rows = sqlx::query(
            "SELECT pf.id as fact_id, pf.person_id, pf.category, pf.key, pf.value, pf.source, pf.confidence, \
             pf.created_at as fact_created, pf.updated_at as fact_updated, \
             p.id, p.name, p.aliases_json, p.relationship, p.platform_ids_json, p.notes, \
             p.communication_style, p.language_preference, p.last_interaction_at, p.interaction_count, \
             p.created_at, p.updated_at \
             FROM person_facts pf JOIN people p ON pf.person_id = p.id \
             WHERE pf.category IN ('birthday', 'important_date')"
        )
        .fetch_all(&self.pool)
        .await?;

        let today = chrono::Utc::now().date_naive();
        let mut results = Vec::new();

        for r in &rows {
            let value: String = r.get("value");
            // Try to parse month-day from various formats (e.g., "March 15", "03-15", "2000-03-15")
            if let Some(upcoming_in) = days_until_date(&value, today) {
                if upcoming_in <= within_days as i64 && upcoming_in >= 0 {
                    let person = Self::row_to_person(r);
                    let fact = crate::traits::PersonFact {
                        id: r.get("fact_id"),
                        person_id: r.get("person_id"),
                        category: r.get("category"),
                        key: r.get("key"),
                        value: r.get("value"),
                        source: r.get("source"),
                        confidence: r.get("confidence"),
                        created_at: parse_dt(r.get::<String, _>("fact_created")),
                        updated_at: parse_dt(r.get::<String, _>("fact_updated")),
                    };
                    results.push((person, fact));
                }
            }
        }

        Ok(results)
    }

    async fn prune_stale_person_facts(&self, retention_days: u32) -> anyhow::Result<u64> {
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::days(retention_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM person_facts WHERE source = 'consolidation' AND confidence < 1.0 AND updated_at < ?"
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn get_people_needing_reconnect(
        &self,
        inactive_days: u32,
    ) -> anyhow::Result<Vec<crate::traits::Person>> {
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::days(inactive_days as i64)).to_rfc3339();
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people \
             WHERE last_interaction_at IS NOT NULL AND last_interaction_at < ? \
             AND relationship IN ('friend', 'family', 'coworker') \
             ORDER BY last_interaction_at ASC",
        )
        .bind(&cutoff)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_person).collect())
    }
}
