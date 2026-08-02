pub(crate) use super::*;

#[path = "runtime/heartbeat_tests.rs"]
mod heartbeat_tests;

#[path = "runtime/group_session_tests.rs"]
mod group_session_tests;

#[path = "runtime/goal_delivery_tests.rs"]
mod goal_delivery_tests;

#[path = "runtime/goal_failure_tests.rs"]
mod goal_failure_tests;

#[path = "response_analysis_tests.rs"]
mod response_analysis_tests;

#[path = "policy/tool_scoping_tests.rs"]
mod tool_scoping_tests;

#[path = "tools/file_path_extraction_tests.rs"]
mod file_path_extraction_tests;

#[test]
fn nested_scheduled_agents_do_not_clear_the_shared_run_budget() {
    assert!(agent_return_owns_scheduled_run_cleanup(0, true, true));
    assert!(!agent_return_owns_scheduled_run_cleanup(1, true, true));
    assert!(!agent_return_owns_scheduled_run_cleanup(2, true, true));
    assert!(agent_return_owns_scheduled_run_cleanup(1, false, true));
    assert!(!agent_return_owns_scheduled_run_cleanup(0, true, false));
}
