//! Live surfaces: screens the agent works on that the owner can watch and
//! take over from a paired app.
//!
//! A surface is anything with pixels that takes input: a browser tab now, a
//! desktop or a window later. Each kind has a [`SurfaceBackend`]. The
//! [`SurfaceRegistry`] owns the backends, one control lease per surface
//! ([`Controller`]), the live sessions watching each surface and the
//! handoffs the agent asked for.
//!
//! This crate knows nothing about browsers, the client gateway or its wire
//! format: backends depend on it, and it depends on none of them.
//!
//! Invariants:
//! - Exactly one controller per surface; the registry is the only authority
//!   on who may act there.
//! - Frames and input are never stored, logged or shown to a model. A
//!   stream keeps only its newest frame.
//! - Only the device holding the lease sends input.

mod registry;
mod types;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

/// `hand_over`, the tool an agent asks the owner to act on a surface with.
#[cfg(feature = "tool")]
pub mod tool;

pub use registry::{
    LiveMode, LiveReceivers, LiveSession, SurfaceChange, SurfaceNotifier, SurfaceRegistry,
    SurfacesConfig,
};
pub use types::{
    Codec, Controller, ExclusiveGuard, Frame, FrameSink, HandoffPrep, InputEvent, KeyAction,
    MAX_TEXT_CHARS, MAX_TITLE_CHARS, Modifier, PointedAt, PointerAction, PointerButton, Size,
    StreamHandle, StreamParams, SurfaceCaps, SurfaceDescriptor, SurfaceError, SurfaceId,
    SurfaceInfo, SurfaceKind, SurfaceObservation, Typing, clean_title, now_ms,
};

use async_trait::async_trait;

/// One kind of surface: lists them, streams them, applies input to them.
#[async_trait]
pub trait SurfaceBackend: Send + Sync {
    fn kind(&self) -> SurfaceKind;

    /// Every surface of this kind that exists now.
    async fn list(&self) -> Vec<SurfaceDescriptor>;

    /// Start producing frames into `sink` until the handle is dropped.
    async fn start_stream(
        &self,
        id: &SurfaceId,
        params: StreamParams,
        sink: FrameSink,
    ) -> Result<StreamHandle, SurfaceError>;

    /// Apply one input event from the owner. The registry has already
    /// checked who sent it.
    async fn input(&self, id: &SurfaceId, event: &InputEvent) -> Result<(), SurfaceError>;

    /// Where the surface is now, in typed facts.
    async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError>;

    /// Keep the agent's own actions on the surface waiting until the guard
    /// is dropped. Waits for an action already running to finish.
    async fn exclusive(&self, id: &SurfaceId) -> Result<ExclusiveGuard, SurfaceError>;

    /// What is at `(x, y)` (fractions of the frame), for the owner to point
    /// the agent at. Default: this surface cannot say.
    async fn describe_point(
        &self,
        _id: &SurfaceId,
        _x: f64,
        _y: f64,
    ) -> Result<PointedAt, SurfaceError> {
        Err(SurfaceError::NotSupported(
            "this screen cannot say what is there",
        ))
    }

    /// What has the keyboard on the surface now (the field the owner's
    /// typing goes to), described like a point. `None`: nothing does.
    /// Default: this surface cannot say.
    async fn focused(&self, _id: &SurfaceId) -> Result<Option<PointedAt>, SurfaceError> {
        Err(SurfaceError::NotSupported(
            "this screen cannot say what has focus",
        ))
    }

    /// Lay the surface out for the controlling viewer's room (`Some`), or
    /// back at its own size (`None`) before the agent acts again. Default:
    /// this surface keeps its size.
    async fn fit(&self, _id: &SurfaceId, _room: Option<Size>) -> Result<(), SurfaceError> {
        Ok(())
    }

    /// Get the surface ready before the owner is asked to act on it (a
    /// browser moves to a profile that keeps sign-ins). May reopen it under
    /// another id.
    async fn prepare_handoff(&self, id: &SurfaceId) -> Result<HandoffPrep, SurfaceError> {
        Ok(HandoffPrep {
            id: id.clone(),
            keeps_sign_in: None,
        })
    }
}
