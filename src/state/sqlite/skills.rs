use super::*;

#[async_trait]
impl crate::traits::SkillStore for SqliteStateStore {
    async fn add_dynamic_skill(&self, skill: &crate::traits::DynamicSkill) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_skills (name, description, triggers_json, body, source, source_url, enabled, version, resources_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&skill.name)
        .bind(&skill.description)
        .bind(&skill.triggers_json)
        .bind(&skill.body)
        .bind(&skill.source)
        .bind(&skill.source_url)
        .bind(skill.enabled)
        .bind(&skill.version)
        .bind(&skill.resources_json)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<crate::traits::DynamicSkill>> {
        let rows = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source, source_url, enabled, version, resources_json, created_at
             FROM dynamic_skills ORDER BY created_at ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut skills = Vec::new();
        for row in rows {
            skills.push(crate::traits::DynamicSkill {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                description: row.get::<String, _>("description"),
                triggers_json: row.get::<String, _>("triggers_json"),
                body: row.get::<String, _>("body"),
                source: row.get::<String, _>("source"),
                source_url: row.get::<Option<String>, _>("source_url"),
                enabled: row.get::<bool, _>("enabled"),
                version: row.get::<Option<String>, _>("version"),
                created_at: row.get::<String, _>("created_at"),
                resources_json: row
                    .try_get::<String, _>("resources_json")
                    .unwrap_or_else(|_| "[]".to_string()),
            });
        }
        Ok(skills)
    }

    async fn delete_dynamic_skill(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_skills WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_skill_enabled(&self, id: i64, enabled: bool) -> anyhow::Result<()> {
        sqlx::query("UPDATE dynamic_skills SET enabled = ? WHERE id = ?")
            .bind(enabled)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_promotable_procedures(
        &self,
        min_success: i32,
        min_rate: f32,
    ) -> anyhow::Result<Vec<crate::traits::Procedure>> {
        let rows = sqlx::query(
            "SELECT id, name, trigger_pattern, steps, success_count, failure_count,
                    avg_duration_secs, last_used_at, created_at, updated_at
             FROM procedures
             WHERE success_count >= ?
               AND CAST(success_count AS REAL) / CAST(success_count + failure_count AS REAL) >= ?
             ORDER BY success_count DESC",
        )
        .bind(min_success)
        .bind(min_rate)
        .fetch_all(&self.pool)
        .await?;

        let mut procedures = Vec::new();
        for row in rows {
            let steps_json: String = row.get("steps");
            let steps: Vec<String> = serde_json::from_str(&steps_json).unwrap_or_default();
            procedures.push(crate::traits::Procedure {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                trigger_pattern: row.get::<String, _>("trigger_pattern"),
                steps,
                success_count: row.get::<i32, _>("success_count"),
                failure_count: row.get::<i32, _>("failure_count"),
                avg_duration_secs: row.get::<Option<f32>, _>("avg_duration_secs"),
                last_used_at: row
                    .get::<Option<String>, _>("last_used_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc)),
                created_at: row
                    .get::<Option<String>, _>("created_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: row
                    .get::<Option<String>, _>("updated_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
            });
        }
        Ok(procedures)
    }

    async fn add_skill_draft(&self, draft: &crate::traits::SkillDraft) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO skill_drafts (name, description, triggers_json, body, source_procedure, status, created_at)
             VALUES (?, ?, ?, ?, ?, 'pending', datetime('now'))",
        )
        .bind(&draft.name)
        .bind(&draft.description)
        .bind(&draft.triggers_json)
        .bind(&draft.body)
        .bind(&draft.source_procedure)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<crate::traits::SkillDraft>> {
        let rows = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source_procedure, status, created_at
             FROM skill_drafts WHERE status = 'pending' ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut drafts = Vec::new();
        for row in rows {
            drafts.push(crate::traits::SkillDraft {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                description: row.get::<String, _>("description"),
                triggers_json: row.get::<String, _>("triggers_json"),
                body: row.get::<String, _>("body"),
                source_procedure: row.get::<String, _>("source_procedure"),
                status: row.get::<String, _>("status"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(drafts)
    }

    async fn get_skill_draft(&self, id: i64) -> anyhow::Result<Option<crate::traits::SkillDraft>> {
        let row = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source_procedure, status, created_at
             FROM skill_drafts WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|row| crate::traits::SkillDraft {
            id: row.get::<i64, _>("id"),
            name: row.get::<String, _>("name"),
            description: row.get::<String, _>("description"),
            triggers_json: row.get::<String, _>("triggers_json"),
            body: row.get::<String, _>("body"),
            source_procedure: row.get::<String, _>("source_procedure"),
            status: row.get::<String, _>("status"),
            created_at: row.get::<String, _>("created_at"),
        }))
    }

    async fn update_skill_draft_status(&self, id: i64, status: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE skill_drafts SET status = ? WHERE id = ?")
            .bind(status)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn skill_draft_exists_for_procedure(&self, procedure_name: &str) -> anyhow::Result<bool> {
        let row =
            sqlx::query("SELECT COUNT(*) as cnt FROM skill_drafts WHERE source_procedure = ?")
                .bind(procedure_name)
                .fetch_one(&self.pool)
                .await?;
        Ok(row.get::<i64, _>("cnt") > 0)
    }
}
