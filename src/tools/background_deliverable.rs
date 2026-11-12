//! Typed background artifact attribution (defined in `aidaemon-tools`).

pub use aidaemon_tools::background_deliverable::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::path::{Path, PathBuf};
    #[allow(unused_imports)]
    use std::time::{Duration, SystemTime};

    #[tokio::test]
    async fn only_exact_declared_fresh_files_are_candidates() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let declared = root.path().join("declared.txt");
        let undeclared = root.path().join("undeclared.txt");
        let start = SystemTime::now() - Duration::from_secs(2);
        std::fs::write(&declared, "declared").expect("declared fixture");
        std::fs::write(&undeclared, "undeclared").expect("undeclared fixture");
        let end = SystemTime::now();

        let context = attribute_declared_deliverables_backend(
            backend,
            &[declared.to_string_lossy().to_string()],
            start,
            end,
        )
        .await;

        assert_eq!(context.produced_candidates, [declared]);
        assert!(context.unconfirmed_candidates.is_empty());
        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::One(_)
        ));
    }

    #[tokio::test]
    async fn directories_relative_paths_and_sinks_are_not_artifacts() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let now = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[
                root.path().to_string_lossy().to_string(),
                "relative.txt".to_string(),
                "/dev/null".to_string(),
            ],
            now - Duration::from_secs(1),
            now,
        )
        .await;

        assert!(context.produced_candidates.is_empty());
        assert!(context.unconfirmed_candidates.is_empty());
        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::None
        ));
    }

    #[tokio::test]
    async fn exact_missing_target_is_reported_without_searching() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let missing = root.path().join("missing.txt");
        let now = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[missing.to_string_lossy().to_string()],
            now - Duration::from_secs(1),
            now,
        )
        .await;

        assert!(context.produced_candidates.is_empty());
        assert_eq!(context.unconfirmed_candidates, [missing]);
    }

    #[tokio::test]
    async fn two_declared_fresh_files_are_ambiguous() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let first = root.path().join("a.txt");
        let second = root.path().join("b.txt");
        let start = SystemTime::now() - Duration::from_secs(2);
        std::fs::write(&first, "a").expect("first fixture");
        std::fs::write(&second, "b").expect("second fixture");
        let end = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[
                second.to_string_lossy().to_string(),
                first.to_string_lossy().to_string(),
            ],
            start,
            end,
        )
        .await;

        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::Ambiguous(paths) if paths == vec![first, second]
        ));
    }
}
