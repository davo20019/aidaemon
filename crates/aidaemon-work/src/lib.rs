//! Goals, mandates, tasks, work coordination and notifications: the domain
//! types, the store traits and their SQLite implementation.

pub mod goals;
pub mod mandates;
pub mod stores;

/// Every domain type and store trait, under one path.
pub mod traits {
    pub use crate::goals::*;
    pub use crate::mandates::*;
    pub use crate::stores::*;
}
