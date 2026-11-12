//! The memory store exercised through the state store: the same contract
//! callers rely on, over a real state database.

use super::*;
use crate::memory::embeddings::EMBEDDING_MODEL_ID;
use aidaemon_core::ChannelVisibility;
use aidaemon_memory::records::*;
use aidaemon_memory::stores::GraphSearchScope;

mod claims;
mod facts;
mod structured;
