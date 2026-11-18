use crate::traits::Message;
use aidaemon_memory::cosine as cosine_similarity;
use chrono::Utc;

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 1.0];
    let b = vec![1.0, 0.0, 1.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

    let c = vec![0.0, 1.0, 0.0];
    assert!((cosine_similarity(&a, &c)).abs() < 0.001);
}

#[test]
fn test_scoring_heuristic() {
    let mut msg = Message {
        id: "1".into(),
        session_id: "s1".into(),
        role: "user".into(),
        content: Some("hello".into()),
        tool_call_id: None,
        tool_name: None,
        tool_calls_json: None,
        created_at: Utc::now(),
        importance: 0.0,
        ..Message::runtime_defaults()
    };

    // Short message "hello" gets penalty (-0.2)
    // User base (0.6) - 0.2 = 0.4
    let score = aidaemon_memory::turn_importance(&msg.role, msg.content.as_deref());
    assert!(
        score < 0.5,
        "Short 'hello' should be low importance: {}",
        score
    );

    // Medium message
    msg.content = Some(
        "I am writing a longer message to test the length heuristic functionality properly.".into(),
    );
    let score_medium = aidaemon_memory::turn_importance(&msg.role, msg.content.as_deref());
    assert!(
        score_medium > 0.5,
        "Medium message should be normal importance: {}",
        score_medium
    );

    // Long messages carry more; wording adds nothing.
    msg.content = Some("a".repeat(250));
    let score_high = aidaemon_memory::turn_importance(&msg.role, msg.content.as_deref());
    assert!(
        score_high >= 0.8,
        "A long message should be high importance: {}",
        score_high
    );
    msg.content = Some("this is a very important secret password".into());
    assert_eq!(
        aidaemon_memory::turn_importance(&msg.role, msg.content.as_deref()),
        score_medium
    );

    // System role
    msg.role = "system".into();
    assert_eq!(
        aidaemon_memory::turn_importance(&msg.role, msg.content.as_deref()),
        0.0
    );
}

#[test]
fn fact_lifetime_is_durable_unless_explicitly_transient() {
    #[derive(serde::Deserialize)]
    struct Row {
        #[serde(default)]
        lifetime: aidaemon_memory::task_learning::FactLifetime,
    }
    let parse = |json: &str| serde_json::from_str::<Row>(json).unwrap().lifetime;
    assert!(parse(r#"{"lifetime":"transient"}"#).is_transient());
    assert!(parse(r#"{"lifetime":" Transient "}"#).is_transient());
    for durable in [
        r#"{}"#,
        r#"{"lifetime":null}"#,
        r#"{"lifetime":"durable"}"#,
        r#"{"lifetime":"weekly"}"#,
        r#"{"lifetime":3}"#,
    ] {
        assert!(!parse(durable).is_transient(), "{durable}");
    }
}
