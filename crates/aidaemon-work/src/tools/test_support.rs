//! Fixtures the tool tests share: a work store on a fresh database file.

use std::sync::Arc;

use crate::sqlite::{SqliteWorkStore, open_for_tests};

/// A host keeps chat-to-channel mappings beside the work tables; the tools
/// never read them, so the test store keeps none.
impl aidaemon_channels::hub::SessionChannelStore for SqliteWorkStore {}

/// A store on a fresh database file, deleted when the test process exits.
pub(crate) async fn test_store() -> Arc<SqliteWorkStore> {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let store = open_for_tests(db_file.path().to_str().unwrap()).await;
    // The pool may reopen the file; delete it when the test process exits.
    aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
    Arc::new(store)
}

/// The state a tool test runs against.
pub(crate) struct TestHarness {
    pub(crate) state: Arc<SqliteWorkStore>,
}

impl TestHarness {
    pub(crate) async fn new() -> Self {
        Self {
            state: test_store().await,
        }
    }
}
