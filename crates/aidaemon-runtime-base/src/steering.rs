//! Messages the owner sent while a run worked, taken into the run.

/// A message the same sender sent while the run was going, taken into it
/// (`sdk_take_steering`). It is answered once a model call that carried it,
/// with tools offered, returned; one never answered goes back to the channel
/// queue at run end (`sdk_return_unanswered_steering`).
pub struct AbsorbedSteering {
    pub message: aidaemon_channels::tasks::QueuedMessage,
    pub answered: bool,
}
