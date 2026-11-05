//! Typed call effects and owner approvals.
//!
//! A tool declares what one call does and to which target
//! ([`Tool::effects`](crate::Tool::effects)), so policies judge effects
//! rather than tool names. A hook that wants the owner to decide returns
//! [`InterruptKind::OwnerApproval`]; when the resolver denies it, the kernel
//! remembers the request's effects in [`crate::RunState::declined`] for the
//! rest of the run.

use serde::{Deserialize, Serialize};

/// Prefix of the reason recorded when the owner declines an approval.
pub const OWNER_DECLINED: &str = "[OWNER DECLINED]";

/// What kind of thing a call does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EffectClass {
    /// Sends data or acts on a host outside the owner's machine.
    Egress,
    /// Deletes a path.
    Delete,
    /// Creates or modifies a path.
    Write,
    /// Changes configuration stored at a path.
    Configuration,
    /// Uses a stored credential.
    Credential,
    /// Starts a program.
    Process,
}

impl EffectClass {
    /// Whether targets of this class are paths, so a target covers the
    /// paths below it.
    fn is_path(self) -> bool {
        matches!(self, Self::Delete | Self::Write | Self::Configuration)
    }
}

/// One effect of a call on one target.
///
/// The producer normalizes `target`: lowercase `host[:port]` for
/// [`EffectClass::Egress`]; an absolute, normalized path for
/// [`EffectClass::Delete`], [`EffectClass::Write`] and
/// [`EffectClass::Configuration`]; the profile name for
/// [`EffectClass::Credential`]; the program name for [`EffectClass::Process`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectKey {
    pub class: EffectClass,
    pub target: String,
}

impl EffectKey {
    /// Same class, and the same target or, for path classes, one target is
    /// the other or below it.
    pub fn overlaps(&self, other: &EffectKey) -> bool {
        if self.class != other.class {
            return false;
        }
        if self.target == other.target {
            return true;
        }
        self.class.is_path()
            && (path_covers(&self.target, &other.target)
                || path_covers(&other.target, &self.target))
    }
}

/// Whether `inner` is `outer` or a path below it, compared by component.
fn path_covers(outer: &str, inner: &str) -> bool {
    let outer = outer.trim_end_matches('/');
    let inner = inner.trim_end_matches('/');
    if outer == inner {
        return true;
    }
    // After trimming, the root is empty and covers every absolute path.
    inner
        .strip_prefix(outer)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// The effects of one call, declared by its tool from the arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEffects {
    pub keys: Vec<EffectKey>,
    /// `keys` lists everything the call does. `false` means the tool could
    /// not tell (an opaque script), so policies must not read the list as
    /// exhaustive.
    pub complete: bool,
}

impl CallEffects {
    /// Effects not declared: the call may do anything.
    pub fn unknown() -> Self {
        Self {
            keys: Vec::new(),
            complete: false,
        }
    }

    /// The call has no effects (a read).
    pub fn none() -> Self {
        Self {
            keys: Vec::new(),
            complete: true,
        }
    }
}

/// How much is at stake in an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ApprovalRisk {
    Medium,
    High,
    Critical,
}

/// What the owner is asked to approve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OwnerApprovalRequest {
    /// The effects the call would have. A denial records them in
    /// [`crate::RunState::declined`].
    pub effects: Vec<EffectKey>,
    /// What the owner sees.
    pub summary: String,
    /// The exact action string that stored grants are matched against (for
    /// example a command line, or `METHOD url`).
    pub subject: String,
    pub risk: ApprovalRisk,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Who an interruption is waiting for.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InterruptKind {
    /// A check of a call made after the run read untrusted content (and,
    /// for interruptions saved before kinds existed, any interruption).
    #[default]
    Untrusted,
    /// The owner must approve the call.
    OwnerApproval(OwnerApprovalRequest),
}
