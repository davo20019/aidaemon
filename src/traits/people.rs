use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A person in the owner's social circle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i64,
    pub name: String,
    pub aliases: Vec<String>,
    pub relationship: Option<String>,
    pub platform_ids: HashMap<String, String>,
    pub notes: Option<String>,
    pub communication_style: Option<String>,
    pub language_preference: Option<String>,
    pub last_interaction_at: Option<DateTime<Utc>>,
    pub interaction_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A fact about a person (birthday, preference, interest, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonFact {
    pub id: i64,
    pub person_id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    pub source: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
