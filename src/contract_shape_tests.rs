//! Serde shape of the shared contract types, pinned byte for byte.
//!
//! These types are persisted (config, database rows) or travel in events, so
//! moving them between crates must not change how they encode. Each test
//! encodes a representative value, compares it with the exact text it had
//! when the test was written, and decodes that text back.

use chrono::{TimeZone, Utc};
use serde::{de::DeserializeOwned, Serialize};

use crate::traits::{
    CliAgentInvocation, DynamicBot, DynamicCliAgent, DynamicMcpServer, DynamicSkill, SkillDraft,
    SpecialistKind,
};
use crate::types::{TerminalConfinement, WorkspaceAccessLevel, WorkspaceGrant};

/// `value` encodes to exactly `json`, and `json` decodes to a value that
/// encodes the same way again.
fn assert_json_shape<T: Serialize + DeserializeOwned>(value: &T, json: &str) {
    let encoded = serde_json::to_string(value).expect("encode");
    assert_eq!(encoded, json);
    let decoded: T = serde_json::from_str(json).expect("decode");
    assert_eq!(serde_json::to_string(&decoded).expect("re-encode"), json);
}

fn grant() -> WorkspaceGrant {
    WorkspaceGrant {
        platform: "slack".to_string(),
        workspace_id: "T_TEST".to_string(),
        channel_id: "C_TEST".to_string(),
        user_id: "U_TEST".to_string(),
        project_root: "/tmp/example-project".to_string(),
        access: WorkspaceAccessLevel::Read,
        expires_at: Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap(),
    }
}

#[test]
fn terminal_confinement_shape() {
    assert_json_shape(&TerminalConfinement::Host, r#""host""#);
    assert_json_shape(&TerminalConfinement::Sandbox, r#""sandbox""#);
    let from_toml: toml::Value = toml::Value::try_from(TerminalConfinement::Sandbox).unwrap();
    assert_eq!(from_toml.as_str(), Some("sandbox"));
}

#[test]
fn workspace_grant_shape() {
    assert_json_shape(&WorkspaceAccessLevel::Edit, r#""edit""#);
    assert_json_shape(
        &grant(),
        r#"{"platform":"slack","workspace_id":"T_TEST","channel_id":"C_TEST","user_id":"U_TEST","project_root":"/tmp/example-project","access":"read","expires_at":"2026-01-02T03:04:05Z"}"#,
    );
    let toml_text = toml::to_string(&grant()).unwrap();
    assert_eq!(
        toml_text,
        "platform = \"slack\"\nworkspace_id = \"T_TEST\"\nchannel_id = \"C_TEST\"\nuser_id = \"U_TEST\"\nproject_root = \"/tmp/example-project\"\naccess = \"read\"\nexpires_at = \"2026-01-02T03:04:05Z\"\n"
    );
    let decoded: WorkspaceGrant = toml::from_str(&toml_text).unwrap();
    assert_eq!(decoded, grant());
}

#[test]
fn specialist_kind_shape() {
    assert_json_shape(&SpecialistKind::ArtifactWriter, r#""ArtifactWriter""#);
    assert_json_shape(&SpecialistKind::TaskLead, r#""TaskLead""#);
}

#[test]
fn dynamic_skill_and_draft_shape() {
    let skill = DynamicSkill {
        id: 7,
        name: "example-skill".to_string(),
        description: "An example".to_string(),
        triggers_json: r#"["example"]"#.to_string(),
        body: "Do the example.".to_string(),
        source: "inline".to_string(),
        source_url: None,
        enabled: true,
        version: Some("1.0".to_string()),
        created_at: "2026-01-01 00:00:00".to_string(),
        resources_json: "[]".to_string(),
    };
    assert_json_shape(
        &skill,
        r#"{"id":7,"name":"example-skill","description":"An example","triggers_json":"[\"example\"]","body":"Do the example.","source":"inline","source_url":null,"enabled":true,"version":"1.0","created_at":"2026-01-01 00:00:00","resources_json":"[]"}"#,
    );
    // Rows written before `resources_json` existed still decode.
    let legacy: DynamicSkill = serde_json::from_str(
        r#"{"id":7,"name":"n","description":"d","triggers_json":"[]","body":"b","source":"url","source_url":"https://example.test/s","enabled":false,"version":null,"created_at":"c"}"#,
    )
    .unwrap();
    assert_eq!(legacy.resources_json, "[]");

    let draft = SkillDraft {
        id: 3,
        name: "draft".to_string(),
        description: "A draft".to_string(),
        triggers_json: "[]".to_string(),
        body: "Steps.".to_string(),
        source_procedure: "example-procedure".to_string(),
        status: "pending".to_string(),
        created_at: "2026-01-01 00:00:00".to_string(),
    };
    assert_json_shape(
        &draft,
        r#"{"id":3,"name":"draft","description":"A draft","triggers_json":"[]","body":"Steps.","source_procedure":"example-procedure","status":"pending","created_at":"2026-01-01 00:00:00"}"#,
    );
}

#[test]
fn dynamic_bot_shape() {
    let bot = DynamicBot {
        id: 2,
        channel_type: "slack".to_string(),
        bot_token: "xoxb-test".to_string(),
        app_token: Some("xapp-test".to_string()),
        allowed_user_ids: vec!["U1".to_string(), "U2".to_string()],
        extra_config: "{}".to_string(),
        created_at: "2026-01-01".to_string(),
    };
    assert_json_shape(
        &bot,
        r#"{"id":2,"channel_type":"slack","bot_token":"xoxb-test","app_token":"xapp-test","allowed_user_ids":["U1","U2"],"extra_config":"{}","created_at":"2026-01-01"}"#,
    );
}

#[test]
fn dynamic_mcp_server_shape() {
    let server = DynamicMcpServer {
        id: 4,
        name: "example".to_string(),
        command: "npx".to_string(),
        args_json: r#"["-y","example-mcp"]"#.to_string(),
        env_keys_json: r#"["EXAMPLE_KEY"]"#.to_string(),
        triggers_json: "[]".to_string(),
        enabled: true,
        created_at: "2026-01-01".to_string(),
    };
    assert_json_shape(
        &server,
        r#"{"id":4,"name":"example","command":"npx","args_json":"[\"-y\",\"example-mcp\"]","env_keys_json":"[\"EXAMPLE_KEY\"]","triggers_json":"[]","enabled":true,"created_at":"2026-01-01"}"#,
    );
}

#[test]
fn dynamic_cli_agent_and_invocation_shape() {
    let agent = DynamicCliAgent {
        id: 5,
        name: "example-agent".to_string(),
        command: "example".to_string(),
        args_json: "[]".to_string(),
        description: "An example agent".to_string(),
        timeout_secs: Some(600),
        max_output_chars: None,
        enabled: true,
        created_at: "2026-01-01".to_string(),
    };
    assert_json_shape(
        &agent,
        r#"{"id":5,"name":"example-agent","command":"example","args_json":"[]","description":"An example agent","timeout_secs":600,"max_output_chars":null,"enabled":true,"created_at":"2026-01-01"}"#,
    );

    let invocation = CliAgentInvocation {
        id: 9,
        task_id: Some("task-1".to_string()),
        session_id: "session-1".to_string(),
        agent_name: "example-agent".to_string(),
        prompt_summary: "Summarize".to_string(),
        working_dir: None,
        started_at: "2026-01-01 00:00:00".to_string(),
        completed_at: Some("2026-01-01 00:01:00".to_string()),
        exit_code: Some(0),
        output_summary: Some("done".to_string()),
        success: Some(true),
        duration_secs: Some(60.5),
    };
    assert_json_shape(
        &invocation,
        r#"{"id":9,"task_id":"task-1","session_id":"session-1","agent_name":"example-agent","prompt_summary":"Summarize","working_dir":null,"started_at":"2026-01-01 00:00:00","completed_at":"2026-01-01 00:01:00","exit_code":0,"output_summary":"done","success":true,"duration_secs":60.5}"#,
    );
}
