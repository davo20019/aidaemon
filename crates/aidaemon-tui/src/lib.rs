//! # aidaemon-tui
//!
//! A terminal client for the AIdaemon Client Gateway, built on the client
//! protocol alone (`aidaemon-client-protocol`): the sessions list, chat
//! with streaming and live status, approval cards from the typed card
//! facts, the running-work panel with cancel, and a connection that
//! reconnects and resumes from its last event.

pub mod app;
pub mod link;
pub mod state;
