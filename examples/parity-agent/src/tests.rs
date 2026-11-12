//! Every behavior case as a test, for each memory system.

use aidaemon_sdk::policies::ApprovalMode;
use aidaemon_sdk::testing::behavior::{cases, run, run_case, Requires, Subject, Verdict};

use super::{assemble, subjects, Kind};

fn named(name: &str) -> Subject {
    subjects()
        .into_iter()
        .find(|s| s.name().starts_with(name))
        .unwrap()
}

/// Passed, or skipped only because the case needs the full memory system
/// and the subject is not it.
async fn check(name: &str, case: &str) {
    let subject = named(name);
    match run_case(&subject, case).await {
        Verdict::Passed => {}
        Verdict::Skipped(Requires::FullMemory) if !subject.name().starts_with("FullMemory") => {}
        other => panic!("{}: {case}: {other:?}", subject.name()),
    }
}

macro_rules! cases_for {
    ($module:ident, $subject:literal) => {
        mod $module {
            use super::check;

            #[tokio::test]
            async fn memory_write() {
                check($subject, "memory_write").await;
            }
            #[tokio::test]
            async fn memory_recall() {
                check($subject, "memory_recall").await;
            }
            #[tokio::test]
            async fn tool_use() {
                check($subject, "tool_use").await;
            }
            #[tokio::test]
            async fn approvals() {
                check($subject, "approvals").await;
            }
            #[tokio::test]
            async fn sub_agents() {
                check($subject, "sub_agents").await;
            }
            #[tokio::test]
            async fn channel_round_trip() {
                check($subject, "channel_round_trip").await;
            }
            #[tokio::test]
            async fn full_memory_tools() {
                check($subject, "full_memory_tools").await;
            }
            #[tokio::test]
            async fn full_memory_manage() {
                check($subject, "full_memory_manage").await;
            }
        }
    };
}

cases_for!(full_memory, "FullMemory");
cases_for!(scoped_memory, "ScopedMemory");
cases_for!(notebook, "Notebook");

#[test]
fn every_case_is_listed_in_the_macro() {
    let here = include_str!("tests.rs");
    for case in cases() {
        assert!(
            here.contains(&format!("async fn {}()", case.name)),
            "{}",
            case.name
        );
    }
}

/// The other memory systems skip exactly the full memory's own cases.
#[tokio::test]
async fn only_the_full_memory_cases_are_skipped_for_other_memories() {
    let full: Vec<_> = cases()
        .into_iter()
        .filter(|c| c.requires == Requires::FullMemory)
        .map(|c| c.name)
        .collect();
    assert!(!full.is_empty());
    for subject in &subjects()[1..] {
        let report = run(subject).await;
        assert!(report.all_passed(), "{report}");
        let skipped = report.named(|v| matches!(v, Verdict::Skipped(_)));
        assert_eq!(skipped, full, "{report}");
    }
}

/// The suite has teeth: assemblies that break a promise fail it.
#[tokio::test]
async fn a_broken_assembly_fails() {
    let no_approvals = Subject::new("no approvals", |parts| async move {
        let preset = aidaemon_sdk::preset(parts.id())
            .approvals(ApprovalMode::Never)
            .undeclared_writes(aidaemon_sdk::policies::UndeclaredAccess::Allow);
        Ok(parts.finish(preset).build().await?)
    });
    assert!(run_case(&no_approvals, "approvals").await.is_failed());
    assert!(run_case(&no_approvals, "channel_round_trip")
        .await
        .is_failed());

    let no_memory = Subject::new("no memory", |parts| async move {
        let preset = aidaemon_sdk::preset(parts.id()).without_memory();
        Ok(parts.finish(preset).build().await?)
    });
    assert!(run_case(&no_memory, "memory_write").await.is_failed());
    assert!(run_case(&no_memory, "memory_recall").await.is_failed());

    // FullMemory's own cases do run, and fail, for a subject that claims it
    // but uses another memory.
    let claims_full = Subject::new("claims full", |p| assemble(p, Kind::Notebook)).full_memory();
    assert!(run_case(&claims_full, "full_memory_tools")
        .await
        .is_failed());
}
