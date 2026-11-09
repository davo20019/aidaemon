//! A plain-text copy of everything [`SqliteMemory`] holds, for people to read
//! or to carry to another memory system by hand.

use std::fmt::Write as _;

use sqlx::Row;

use super::SqliteMemory;

/// One file of a readable export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportFile {
    pub name: &'static str,
    pub content: String,
    /// How many entries it holds.
    pub entries: usize,
}

fn one_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl SqliteMemory {
    /// The memory as Markdown files. Reads only.
    pub async fn export_readable(&self) -> anyhow::Result<Vec<ExportFile>> {
        Ok(vec![
            self.export_facts(false).await?,
            self.export_facts(true).await?,
            self.export_people().await?,
            self.export_episodes().await?,
            self.export_learned().await?,
        ])
    }

    async fn export_facts(&self, superseded: bool) -> anyhow::Result<ExportFile> {
        let filter = if superseded { "IS NOT NULL" } else { "IS NULL" };
        let rows = sqlx::query(&format!(
            "SELECT category, key, value, source, privacy, updated_at, superseded_at
             FROM facts WHERE superseded_at {filter} ORDER BY category, key, id"
        ))
        .fetch_all(&self.pool)
        .await?;
        let (name, title) = if superseded {
            ("facts-history.md", "Earlier versions of facts")
        } else {
            ("facts.md", "Facts")
        };
        let mut content = format!("# {title}\n");
        let mut category = None;
        for row in &rows {
            let current: String = row.get("category");
            if category.as_ref() != Some(&current) {
                let _ = write!(content, "\n## {current}\n\n");
                category = Some(current);
            }
            let _ = writeln!(
                content,
                "- **{}**: {} _(source: {}, privacy: {}, updated: {})_",
                row.get::<String, _>("key"),
                one_line(&row.get::<String, _>("value")),
                row.get::<String, _>("source"),
                row.try_get::<Option<String>, _>("privacy")?
                    .unwrap_or_else(|| "global".to_string()),
                row.get::<String, _>("updated_at"),
            );
        }
        Ok(ExportFile {
            name,
            content,
            entries: rows.len(),
        })
    }

    async fn export_people(&self) -> anyhow::Result<ExportFile> {
        let people = sqlx::query(
            "SELECT id, name, relationship, notes FROM people ORDER BY LOWER(name), id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut content = String::from("# People\n");
        for person in &people {
            let _ = write!(content, "\n## {}\n\n", person.get::<String, _>("name"));
            if let Some(relationship) = person.get::<Option<String>, _>("relationship") {
                let _ = writeln!(content, "- Relationship: {}", one_line(&relationship));
            }
            if let Some(notes) = person.get::<Option<String>, _>("notes") {
                let _ = writeln!(content, "- Notes: {}", one_line(&notes));
            }
            let facts = sqlx::query(
                "SELECT category, key, value FROM person_facts
                 WHERE person_id = ? ORDER BY category, key, id",
            )
            .bind(person.get::<i64, _>("id"))
            .fetch_all(&self.pool)
            .await?;
            for fact in &facts {
                let _ = writeln!(
                    content,
                    "- {} / {}: {}",
                    fact.get::<String, _>("category"),
                    fact.get::<String, _>("key"),
                    one_line(&fact.get::<String, _>("value")),
                );
            }
        }
        Ok(ExportFile {
            name: "people.md",
            content,
            entries: people.len(),
        })
    }

    async fn export_episodes(&self) -> anyhow::Result<ExportFile> {
        let rows = sqlx::query(
            "SELECT session_id, summary, start_time, end_time FROM episodes
             ORDER BY start_time, id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut content = String::from("# Conversation summaries\n");
        for row in &rows {
            let _ = write!(
                content,
                "\n## {} to {}\n\n{}\n",
                row.get::<String, _>("start_time"),
                row.get::<String, _>("end_time"),
                row.get::<String, _>("summary").trim(),
            );
        }
        Ok(ExportFile {
            name: "episodes.md",
            content,
            entries: rows.len(),
        })
    }

    async fn export_learned(&self) -> anyhow::Result<ExportFile> {
        let mut content = String::from("# Learned habits\n\n## Procedures\n\n");
        let mut entries = 0;
        for row in sqlx::query(
            "SELECT name, trigger_pattern, steps, success_count, failure_count
             FROM procedures ORDER BY name, id",
        )
        .fetch_all(&self.pool)
        .await?
        {
            entries += 1;
            let _ = writeln!(
                content,
                "- **{}** when {}: {} _(worked {}, failed {})_",
                row.get::<String, _>("name"),
                one_line(&row.get::<String, _>("trigger_pattern")),
                one_line(&row.get::<String, _>("steps")),
                row.get::<i64, _>("success_count"),
                row.get::<i64, _>("failure_count"),
            );
        }
        content.push_str("\n## Error solutions\n\n");
        for row in sqlx::query(
            "SELECT error_pattern, solution_summary FROM error_solutions ORDER BY error_pattern, id",
        )
        .fetch_all(&self.pool)
        .await?
        {
            entries += 1;
            let _ = writeln!(
                content,
                "- {}: {}",
                one_line(&row.get::<String, _>("error_pattern")),
                one_line(&row.get::<String, _>("solution_summary")),
            );
        }
        content.push_str("\n## Behavior patterns\n\n");
        for row in sqlx::query(
            "SELECT pattern_type, description FROM behavior_patterns ORDER BY pattern_type, id",
        )
        .fetch_all(&self.pool)
        .await?
        {
            entries += 1;
            let _ = writeln!(
                content,
                "- {}: {}",
                row.get::<String, _>("pattern_type"),
                one_line(&row.get::<String, _>("description")),
            );
        }
        content.push_str("\n## Expertise\n\n");
        for row in sqlx::query(
            "SELECT domain, current_level, tasks_attempted FROM expertise ORDER BY domain, id",
        )
        .fetch_all(&self.pool)
        .await?
        {
            entries += 1;
            let _ = writeln!(
                content,
                "- {}: {} ({} tasks)",
                row.get::<String, _>("domain"),
                row.get::<String, _>("current_level"),
                row.get::<i64, _>("tasks_attempted"),
            );
        }
        Ok(ExportFile {
            name: "learned.md",
            content,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashEmbedder;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::sync::Arc;

    #[tokio::test]
    async fn every_kind_of_memory_is_readable() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(SqliteConnectOptions::new().filename(file.path()))
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        for statement in [
            "INSERT INTO facts (category, key, value, source, created_at, updated_at, superseded_at) VALUES
                ('pets', 'iguana', 'Zorblat', 'user', '2025-01-01', '2025-01-01', '2026-01-01'),
                ('pets', 'iguana', 'Quibble
                 the second', 'user', '2026-01-01', '2026-01-01', NULL)",
            "INSERT INTO people (id, name, relationship, created_at, updated_at) VALUES
                (1, 'Alice Rivera', 'coworker', '2025-01-01', '2025-01-01')",
            "INSERT INTO person_facts (person_id, category, key, value, source, created_at, updated_at) VALUES
                (1, 'preference', 'snack', 'pretzels', 'user', '2025-01-01', '2025-01-01')",
            "INSERT INTO episodes (session_id, summary, message_count, start_time, end_time, created_at) VALUES
                ('s', 'Planned the Acme launch.', 2, '2025-01-01', '2025-01-02', '2025-01-02')",
            "INSERT INTO procedures (name, trigger_pattern, steps, created_at, updated_at) VALUES
                ('release', 'shipping', '[\"test\"]', '2025-01-01', '2025-01-01')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        let memory = SqliteMemory::new(pool, Arc::new(HashEmbedder::default()));

        let files = memory.export_readable().await.unwrap();

        let file = |name: &str| files.iter().find(|f| f.name == name).unwrap();
        assert_eq!(file("facts.md").entries, 1);
        assert!(file("facts.md")
            .content
            .contains("**iguana**: Quibble the second"));
        assert!(!file("facts.md").content.contains("Zorblat"));
        assert!(file("facts-history.md").content.contains("Zorblat"));
        assert!(file("people.md").content.contains("## Alice Rivera"));
        assert!(file("people.md").content.contains("snack: pretzels"));
        assert!(file("episodes.md")
            .content
            .contains("Planned the Acme launch."));
        assert!(file("learned.md")
            .content
            .contains("**release** when shipping"));
    }
}
