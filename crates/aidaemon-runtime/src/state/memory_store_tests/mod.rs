//! The memory store exercised through the state store: the same contract
//! callers rely on, over a real state database.

use super::*;
use aidaemon_core::ChannelVisibility;
/// Any model id: these tests write and search the index under it directly.
const EMBEDDING_MODEL_ID: &str = "synthetic/test-model";
use aidaemon_memory::records::*;
use aidaemon_memory::stores::GraphSearchScope;

mod claims;
mod facts;
mod structured;
