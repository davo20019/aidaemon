use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{Person, PersonFact, StateStore, Tool};

pub struct ManagePeopleTool {
    state: Arc<dyn StateStore>,
}

impl ManagePeopleTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }
}

#[derive(Deserialize)]
struct ManagePeopleArgs {
    action: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    id: Option<i64>,
    #[serde(default)]
    relationship: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    communication_style: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    value: Option<String>,
    #[serde(default)]
    person_name: Option<String>,
    #[serde(default)]
    platform_id: Option<String>,
    #[serde(default)]
    fact_id: Option<i64>,
    #[serde(default)]
    display_name: Option<String>,
}

#[async_trait]
impl Tool for ManagePeopleTool {
    fn name(&self) -> &str {
        "manage_people"
    }

    fn description(&self) -> &str {
        "Manage the owner's contacts and social circle. Track people, their preferences, relationships, and important dates."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_people",
            "description": "Manage the owner's contacts and social circle. Use 'enable'/'disable'/'status' to toggle People Intelligence at runtime. Other actions: add, list, view, update, remove, add_fact, remove_fact, link, export, purge, audit, confirm",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["enable", "disable", "status", "add", "list", "view", "update", "remove", "add_fact", "remove_fact", "link", "export", "purge", "audit", "confirm"],
                        "description": "Action to perform. 'enable'/'disable' toggle people tracking; 'status' shows current state."
                    },
                    "name": {
                        "type": "string",
                        "description": "Person's name (for add, view)"
                    },
                    "id": {
                        "type": "integer",
                        "description": "Person's database ID (for update, remove)"
                    },
                    "relationship": {
                        "type": "string",
                        "description": "Relationship type: spouse, family, friend, coworker, acquaintance, etc."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Free-form notes about the person"
                    },
                    "communication_style": {
                        "type": "string",
                        "description": "How to communicate with this person: casual, formal, warm, etc."
                    },
                    "language": {
                        "type": "string",
                        "description": "Preferred language for interaction"
                    },
                    "person_name": {
                        "type": "string",
                        "description": "Person's name (for add_fact, link, export, purge, audit)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Fact category: birthday, preference, interest, work, family, important_date, personality, gift_idea"
                    },
                    "key": {
                        "type": "string",
                        "description": "Fact key (e.g., 'birthday', 'favorite_food')"
                    },
                    "value": {
                        "type": "string",
                        "description": "Fact value"
                    },
                    "platform_id": {
                        "type": "string",
                        "description": "Platform-qualified ID (e.g., 'slack:U123', 'telegram:456')"
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Display name for the platform identity"
                    },
                    "fact_id": {
                        "type": "integer",
                        "description": "Fact ID (for remove_fact, confirm)"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManagePeopleArgs = serde_json::from_str(arguments)?;

        // Toggle actions are always allowed regardless of enabled state
        match args.action.as_str() {
            "enable" => return self.handle_enable().await,
            "disable" => return self.handle_disable().await,
            "status" => return self.handle_status().await,
            _ => {}
        }

        // Gate all other actions behind the runtime setting
        if !self.is_people_enabled().await {
            return Ok(
                "People Intelligence is disabled. Use action 'enable' to turn it on.".to_string(),
            );
        }

        match args.action.as_str() {
            "add" => self.handle_add(&args).await,
            "list" => self.handle_list(&args).await,
            "view" => self.handle_view(&args).await,
            "update" => self.handle_update(&args).await,
            "remove" => self.handle_remove(&args).await,
            "add_fact" => self.handle_add_fact(&args).await,
            "remove_fact" => self.handle_remove_fact(&args).await,
            "link" => self.handle_link(&args).await,
            "export" => self.handle_export(&args).await,
            "purge" => self.handle_purge(&args).await,
            "audit" => self.handle_audit(&args).await,
            "confirm" => self.handle_confirm(&args).await,
            other => Ok(format!("Unknown action: {}. Use: enable, disable, status, add, list, view, update, remove, add_fact, remove_fact, link, export, purge, audit, confirm", other)),
        }
    }
}

impl ManagePeopleTool {
    async fn handle_add(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let name = match &args.name {
            Some(n) => n.clone(),
            None => return Ok("Missing required field: name".to_string()),
        };

        let person = Person {
            id: 0,
            name: name.clone(),
            aliases: vec![],
            relationship: args.relationship.clone(),
            platform_ids: HashMap::new(),
            notes: args.notes.clone(),
            communication_style: args.communication_style.clone(),
            language_preference: args.language.clone(),
            last_interaction_at: None,
            interaction_count: 0,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let id = self.state.upsert_person(&person).await?;
        Ok(format!("Added person '{}' with ID {}", name, id))
    }

    async fn handle_list(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let people = self.state.get_all_people().await?;
        if people.is_empty() {
            return Ok("No people tracked yet.".to_string());
        }

        let mut result = format!("**People** ({} total)\n", people.len());
        for p in &people {
            let rel = p.relationship.as_deref().unwrap_or("—");
            let style = p.communication_style.as_deref().unwrap_or("—");
            let interaction_info = if p.interaction_count > 0 {
                format!(", {} interactions", p.interaction_count)
            } else {
                String::new()
            };

            // Filter by relationship if specified
            if let Some(ref filter) = args.relationship {
                if p.relationship.as_deref() != Some(filter.as_str()) {
                    continue;
                }
            }

            result.push_str(&format!(
                "- **{}** (ID: {}) — {} | style: {}{}\n",
                p.name, p.id, rel, style, interaction_info
            ));
        }
        Ok(result)
    }

    async fn handle_view(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person = if let Some(id) = args.id {
            self.state.get_person(id).await?
        } else if let Some(ref name) = args.name {
            self.state.find_person_by_name(name).await?
        } else {
            return Ok("Provide 'name' or 'id' to view a person.".to_string());
        };

        let person = match person {
            Some(p) => p,
            None => return Ok("Person not found.".to_string()),
        };

        let facts = self.state.get_person_facts(person.id, None).await?;
        let mut result = format!("## {}", person.name);
        if let Some(ref rel) = person.relationship {
            result.push_str(&format!(" ({})", rel));
        }
        result.push('\n');

        if !person.aliases.is_empty() {
            result.push_str(&format!("**Aliases:** {}\n", person.aliases.join(", ")));
        }
        if let Some(ref style) = person.communication_style {
            result.push_str(&format!("**Communication style:** {}\n", style));
        }
        if let Some(ref lang) = person.language_preference {
            result.push_str(&format!("**Language:** {}\n", lang));
        }
        if !person.platform_ids.is_empty() {
            let ids: Vec<String> = person
                .platform_ids
                .iter()
                .map(|(k, v)| format!("{} ({})", k, v))
                .collect();
            result.push_str(&format!("**Platform IDs:** {}\n", ids.join(", ")));
        }
        if let Some(ref notes) = person.notes {
            result.push_str(&format!("**Notes:** {}\n", notes));
        }
        result.push_str(&format!("**Interactions:** {}\n", person.interaction_count));
        if let Some(ref last) = person.last_interaction_at {
            result.push_str(&format!(
                "**Last interaction:** {}\n",
                last.format("%Y-%m-%d %H:%M")
            ));
        }

        if !facts.is_empty() {
            result.push_str("\n**Facts:**\n");
            for f in &facts {
                let confidence_marker = if f.confidence < 1.0 {
                    format!(
                        " (confidence: {:.0}%, source: {})",
                        f.confidence * 100.0,
                        f.source
                    )
                } else {
                    String::new()
                };
                result.push_str(&format!(
                    "- [{}] {}: {}{}\n",
                    f.category, f.key, f.value, confidence_marker
                ));
            }
        }

        Ok(result)
    }

    async fn handle_update(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person = if let Some(id) = args.id {
            self.state.get_person(id).await?
        } else if let Some(ref name) = args.name {
            self.state.find_person_by_name(name).await?
        } else {
            return Ok("Provide 'name' or 'id' to update a person.".to_string());
        };

        let mut person = match person {
            Some(p) => p,
            None => return Ok("Person not found.".to_string()),
        };

        if let Some(ref name) = args.name {
            if args.id.is_some() {
                // Only update name if id was used to identify (name is being changed)
                person.name = name.clone();
            }
        }
        if let Some(ref rel) = args.relationship {
            person.relationship = Some(rel.clone());
        }
        if let Some(ref notes) = args.notes {
            person.notes = Some(notes.clone());
        }
        if let Some(ref style) = args.communication_style {
            person.communication_style = Some(style.clone());
        }
        if let Some(ref lang) = args.language {
            person.language_preference = Some(lang.clone());
        }

        self.state.upsert_person(&person).await?;
        Ok(format!("Updated person '{}'", person.name))
    }

    async fn handle_remove(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person = if let Some(id) = args.id {
            self.state.get_person(id).await?
        } else if let Some(ref name) = args.name {
            self.state.find_person_by_name(name).await?
        } else {
            return Ok("Provide 'name' or 'id' to remove a person.".to_string());
        };

        let person = match person {
            Some(p) => p,
            None => return Ok("Person not found.".to_string()),
        };

        self.state.delete_person(person.id).await?;
        Ok(format!(
            "Removed '{}' and all associated facts.",
            person.name
        ))
    }

    async fn handle_add_fact(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person_name = match &args.person_name {
            Some(n) => n,
            None => return Ok("Missing required field: person_name".to_string()),
        };
        let category = match &args.category {
            Some(c) => c,
            None => return Ok("Missing required field: category".to_string()),
        };
        let key = match &args.key {
            Some(k) => k,
            None => return Ok("Missing required field: key".to_string()),
        };
        let value = match &args.value {
            Some(v) => v,
            None => return Ok("Missing required field: value".to_string()),
        };

        let person = match self.state.find_person_by_name(person_name).await? {
            Some(p) => p,
            None => {
                return Ok(format!(
                    "Person '{}' not found. Add them first.",
                    person_name
                ))
            }
        };

        self.state
            .upsert_person_fact(person.id, category, key, value, "agent", 1.0)
            .await?;
        Ok(format!(
            "Added fact [{}/{}] = '{}' for {}",
            category, key, value, person.name
        ))
    }

    async fn handle_remove_fact(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let fact_id = match args.fact_id {
            Some(id) => id,
            None => return Ok("Missing required field: fact_id".to_string()),
        };

        self.state.delete_person_fact(fact_id).await?;
        Ok(format!("Removed fact {}", fact_id))
    }

    async fn handle_link(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person_name = match &args.person_name {
            Some(n) => n,
            None => return Ok("Missing required field: person_name".to_string()),
        };
        let platform_id = match &args.platform_id {
            Some(p) => p,
            None => {
                return Ok(
                    "Missing required field: platform_id (e.g., 'slack:U123', 'telegram:456')"
                        .to_string(),
                )
            }
        };
        let display_name = args.display_name.as_deref().unwrap_or("");

        let person = match self.state.find_person_by_name(person_name).await? {
            Some(p) => p,
            None => return Ok(format!("Person '{}' not found.", person_name)),
        };

        self.state
            .link_platform_id(person.id, platform_id, display_name)
            .await?;
        Ok(format!(
            "Linked platform ID '{}' to {}",
            platform_id, person.name
        ))
    }

    async fn handle_export(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person_name = match &args.person_name {
            Some(n) => n,
            None => match &args.name {
                Some(n) => n,
                None => return Ok("Missing required field: person_name or name".to_string()),
            },
        };

        let person = match self.state.find_person_by_name(person_name).await? {
            Some(p) => p,
            None => return Ok(format!("Person '{}' not found.", person_name)),
        };

        let facts = self.state.get_person_facts(person.id, None).await?;
        let export = json!({
            "person": person,
            "facts": facts,
        });

        Ok(serde_json::to_string_pretty(&export)?)
    }

    async fn handle_purge(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let person_name = match &args.person_name {
            Some(n) => n,
            None => match &args.name {
                Some(n) => n,
                None => return Ok("Missing required field: person_name or name".to_string()),
            },
        };

        let person = match self.state.find_person_by_name(person_name).await? {
            Some(p) => p,
            None => return Ok(format!("Person '{}' not found.", person_name)),
        };

        let facts = self.state.get_person_facts(person.id, None).await?;
        self.state.delete_person(person.id).await?;
        Ok(format!(
            "Purged '{}': deleted person record + {} facts + all platform links.",
            person.name,
            facts.len()
        ))
    }

    async fn handle_audit(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        if let Some(ref name) = args.person_name.as_ref().or(args.name.as_ref()) {
            let person = match self.state.find_person_by_name(name).await? {
                Some(p) => p,
                None => return Ok(format!("Person '{}' not found.", name)),
            };

            let facts = self.state.get_person_facts(person.id, None).await?;
            let auto_facts: Vec<&PersonFact> =
                facts.iter().filter(|f| f.confidence < 1.0).collect();

            if auto_facts.is_empty() {
                return Ok(format!(
                    "No auto-extracted facts for {}. All facts are owner-verified.",
                    person.name
                ));
            }

            let mut result = format!(
                "**Auto-extracted facts for {}** ({} unverified)\n",
                person.name,
                auto_facts.len()
            );
            for f in auto_facts {
                result.push_str(&format!(
                    "- [ID: {}] [{}/{}] = '{}' (confidence: {:.0}%, source: {})\n",
                    f.id,
                    f.category,
                    f.key,
                    f.value,
                    f.confidence * 100.0,
                    f.source
                ));
            }
            result.push_str("\nUse `confirm` with `fact_id` to verify a fact.");
            Ok(result)
        } else {
            // Audit all people
            let people = self.state.get_all_people().await?;
            let mut total_unverified = 0;
            let mut result = String::from("**Audit Summary**\n");

            for p in &people {
                let facts = self.state.get_person_facts(p.id, None).await?;
                let unverified = facts.iter().filter(|f| f.confidence < 1.0).count();
                if unverified > 0 {
                    result.push_str(&format!(
                        "- **{}**: {} unverified facts\n",
                        p.name, unverified
                    ));
                    total_unverified += unverified;
                }
            }

            if total_unverified == 0 {
                return Ok("All people facts are verified.".to_string());
            }
            result.push_str(&format!(
                "\nTotal: {} unverified facts across all people.",
                total_unverified
            ));
            Ok(result)
        }
    }

    async fn handle_confirm(&self, args: &ManagePeopleArgs) -> anyhow::Result<String> {
        let fact_id = match args.fact_id {
            Some(id) => id,
            None => return Ok("Missing required field: fact_id".to_string()),
        };

        self.state.confirm_person_fact(fact_id).await?;
        Ok(format!(
            "Confirmed fact {} (confidence set to 100%, source set to 'owner').",
            fact_id
        ))
    }

    async fn is_people_enabled(&self) -> bool {
        self.state
            .get_setting("people_enabled")
            .await
            .ok()
            .flatten()
            .as_deref()
            == Some("true")
    }

    async fn handle_enable(&self) -> anyhow::Result<String> {
        self.state.set_setting("people_enabled", "true").await?;
        Ok("People Intelligence enabled. I'll now track contacts, learn about people you mention, and provide proactive social reminders.".to_string())
    }

    async fn handle_disable(&self) -> anyhow::Result<String> {
        self.state.set_setting("people_enabled", "false").await?;
        Ok("People Intelligence disabled. All existing data is preserved — use 'enable' to turn it back on.".to_string())
    }

    async fn handle_status(&self) -> anyhow::Result<String> {
        let enabled = self.is_people_enabled().await;
        let people_count = self.state.get_all_people().await.unwrap_or_default().len();
        Ok(format!(
            "People Intelligence: **{}**\nPeople tracked: {}",
            if enabled { "enabled" } else { "disabled" },
            people_count
        ))
    }
}
