//! The work domain's tools: what the model calls to manage mandates, goals
//! and their tasks.

pub mod goal_ids;
pub mod goal_trace;
pub mod manage_goal_tasks;
pub mod manage_mandates;
pub mod mandate_review;
pub mod objective_status;
pub mod scheduled_goal_runs;
pub mod track_requirements;

#[cfg(test)]
mod goal_trace_tests;
#[cfg(test)]
mod manage_goal_tasks_tests;
#[cfg(test)]
mod manage_mandates_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod track_requirements_tests;
