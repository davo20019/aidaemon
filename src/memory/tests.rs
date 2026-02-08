#[cfg(test)]
mod tests {
    use crate::memory::math::cosine_similarity;
    use crate::memory::scoring::score_message;
    use crate::traits::Message;
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
            embedding: None,
        };

        // Short message "hello" gets penalty (-0.2)
        // User base (0.6) - 0.2 = 0.4
        let score = score_message(&msg);
        assert!(
            score < 0.5,
            "Short 'hello' should be low importance: {}",
            score
        );

        // Medium message
        msg.content = Some(
            "I am writing a longer message to test the length heuristic functionality properly."
                .into(),
        );
        let score_medium = score_message(&msg);
        assert!(
            score_medium > 0.5,
            "Medium message should be normal importance: {}",
            score_medium
        );

        // Keyword boost
        msg.content = Some("this is a very important secret password".into());
        let score_high = score_message(&msg);
        assert!(
            score_high >= 0.8,
            "Score should be high for keywords: {}",
            score_high
        );

        // System role
        msg.role = "system".into();
        assert_eq!(score_message(&msg), 0.0);
    }
}
