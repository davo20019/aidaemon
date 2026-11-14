//! First-run setup. With no config the daemon runs [`mode`]: the Client
//! Gateway with no agent behind it, so the app (on this computer, or on
//! another device with the printed code) connects a model, the one step
//! that must pass ([`model`]). The daemon then starts with that model and
//! its app offers the optional chat step and posts the first conversation's
//! greeting ([`channel`]). `aidaemon setup` takes the model step in this
//! terminal instead ([`text`]).

pub(crate) mod channel;
pub(crate) mod mode;
pub(crate) mod model;
pub(crate) mod progress;
pub(crate) mod text;
