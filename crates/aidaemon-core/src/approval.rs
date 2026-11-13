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

/// Prefix of the reason for a call whose owner card got no answer (it
/// expired, could not be delivered, or the turn was cancelled). Not a
/// refusal: nothing is remembered as declined.
pub const OWNER_UNANSWERED: &str = "[NO ANSWER]";

/// Prefix of the reason an outbound secret scan (the
/// [`crate::Safeguard::Egress`] hook) gives when it refuses a call.
pub const EGRESS_BLOCKED: &str = "[EGRESS BLOCKED]";

/// Prefix of the reason the [`crate::AuthorityHook`] gives when a call is
/// outside what the run may do.
pub const AUTHORITY_BLOCKED: &str = "[AUTHORITY BLOCKED]";

/// Prefix of the refusal a rejected egress call returns to the model after
/// the run read untrusted content.
pub const UNTRUSTED_BLOCKED: &str = "[UNTRUSTED BLOCKED]";

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
    /// One exact call, named by its approval subject. Recorded when the
    /// owner declines a request that declared no effects, so the identical
    /// request is not put to the owner again.
    Call,
}

impl std::fmt::Display for EffectClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Egress => "egress",
            Self::Delete => "delete",
            Self::Write => "write",
            Self::Configuration => "configuration",
            Self::Credential => "credential",
            Self::Process => "process",
            Self::Call => "call",
        })
    }
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
/// The producer normalizes `target`: lowercase `host:port` for
/// [`EffectClass::Egress`]; an absolute, normalized path for
/// [`EffectClass::Delete`], [`EffectClass::Write`] and
/// [`EffectClass::Configuration`]; the profile name for
/// [`EffectClass::Credential`]; the program name for [`EffectClass::Process`].
/// Build keys with [`EffectKey::path`], [`EffectKey::egress`] and
/// [`EffectKey::named`] so every producer uses the same convention.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectKey {
    pub class: EffectClass,
    pub target: String,
}

impl EffectKey {
    /// A path effect (`Delete`, `Write`, `Configuration`) on an absolute
    /// path, normalized lexically: empty and `.` components dropped, `..`
    /// resolved, no trailing slash. `None` for a relative path or one that
    /// climbs above `/`. The filesystem is not consulted (symlinks are not
    /// resolved).
    pub fn path(class: EffectClass, raw: &str) -> Option<EffectKey> {
        if !raw.starts_with('/') {
            return None;
        }
        let mut parts: Vec<&str> = Vec::new();
        for part in raw.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop()?;
                }
                part => parts.push(part),
            }
        }
        Some(EffectKey {
            class,
            target: format!("/{}", parts.join("/")),
        })
    }

    /// An egress effect on `host`, with target `host:port`: the host
    /// lowercased and without a trailing dot (`h.example.` is `h.example`),
    /// and the port given or else the scheme's default (http
    /// 80, https 443, ssh 22; 0 for an unknown scheme).
    pub fn egress(scheme: &str, host: &str, port: Option<u16>) -> EffectKey {
        let port = port.unwrap_or_else(|| match scheme.to_ascii_lowercase().as_str() {
            "http" => 80,
            "https" => 443,
            "ssh" => 22,
            _ => 0,
        });
        EffectKey {
            class: EffectClass::Egress,
            target: format!("{}:{port}", host.to_ascii_lowercase().trim_end_matches('.')),
        }
    }

    /// An effect named by an identifier rather than a path or host: the
    /// profile name for [`EffectClass::Credential`], the program name for
    /// [`EffectClass::Process`].
    pub fn named(class: EffectClass, name: &str) -> EffectKey {
        EffectKey {
            class,
            target: name.to_string(),
        }
    }

    /// Same class, and the same target or, for path classes, one target is
    /// the other or below it. Targets are compared as given, so both keys
    /// must be normalized (built with [`EffectKey::path`],
    /// [`EffectKey::egress`] or [`EffectKey::named`]): `h.example.invalid`
    /// does not overlap `h.example.invalid:443`, and `/a/../b` does not
    /// overlap `/b`.
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
    /// Every part of the call was read and `keys` name what it changes,
    /// though the tool does not vouch for the list as `complete` (a shell
    /// command made only of programs it models: a copy into the work area,
    /// a listing, `date`). After the owner declines a step, such a call that
    /// deletes nothing and reaches nothing outside is not taken for the
    /// declined action by another route.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub understood: bool,
}

impl CallEffects {
    /// Effects not declared: the call may do anything.
    pub fn unknown() -> Self {
        Self {
            keys: Vec::new(),
            complete: false,
            understood: false,
        }
    }

    /// The call has no effects (a read).
    pub fn none() -> Self {
        Self {
            keys: Vec::new(),
            complete: true,
            understood: true,
        }
    }
}

/// How much is at stake in an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ApprovalRisk {
    /// A call new to the owner with nothing known to be at stake.
    Low,
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
    /// What the card says, as typed facts each channel renders. Plain
    /// words only, never authority: grants match `subject`. `None` shows
    /// `summary` and `warnings` as they are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card: Option<Box<ApprovalCard>>,
}

/// An approval card's content as facts, so every channel says the same
/// thing in its own formatting: who asks, what will happen, why the owner
/// is asked, and the one high-impact effect worth a warning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalCard {
    /// The helper asking ("Drupal install"); `None` for the assistant
    /// itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked_by: Option<String>,
    pub action: CardAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<AskReason>,
    /// The effect that earns a warning. `None` for ordinary work, which
    /// gets no warning however the request's risk is rated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<HighImpact>,
}

impl ApprovalCard {
    pub fn new(action: CardAction) -> Self {
        Self {
            asked_by: None,
            action,
            reason: None,
            impact: None,
        }
    }
}

/// What allowing the call does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum CardAction {
    /// Run a command on the owner's computer.
    Command {
        /// The exact line that runs (secrets hidden); shown collapsed,
        /// never as the headline.
        line: String,
        /// The few plain words the call gave for itself (`description`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// The folders a read-only command only looks inside, by name.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        looks_inside: Vec<String>,
    },
    /// Let the browser open a site on this computer.
    OpenLocalSite {
        site: String,
        /// How long a one-time allowance lasts.
        once_minutes: u64,
    },
    /// Use a tool; `what` completes "wants to …".
    Use {
        what: String,
        /// The call's arguments as they would run, shown collapsed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

/// Why the owner is asked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum AskReason {
    /// The work read content from outside, and the clean check could not
    /// confirm this step fits the request.
    ReadOutsideContent {
        /// The check's one sentence, addressed to the owner.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        check_says: Option<String>,
    },
    /// The owner said no to a step earlier in this run, and what this call
    /// does cannot be fully told, so it is not run unasked.
    AfterADecline {
        /// What the owner said no to, in a few plain words (the declined
        /// step's own title, or what it would have changed).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        declined: Option<String>,
    },
}

/// An effect that deserves a warning on the card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum HighImpact {
    /// Deletes files: the targets when they can be read.
    Deletes {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        targets: Vec<String>,
    },
    /// Sends or posts something to another service.
    Sends,
    /// Spends money.
    Spends,
    /// Writes outside the work area.
    WritesOutsideWorkArea,
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

/// Owner declines shared by a family of runs: a run and the child runs it
/// starts (and their children). A decline recorded in any of them reaches
/// every other one before its next call, so a parent is refused an effect
/// the owner declined inside its child, and a running child is refused one
/// the owner declined in its parent after the child started.
///
/// Attach it with [`crate::RunState::share_declines`]. The kernel exchanges
/// it with the run's own [`crate::RunState::declined`] when a turn starts,
/// before each call is judged, when the owner declines, and when the run
/// ends; the run's own list stays the serialized source of truth (so a
/// parent whose child was declined something never ends `Succeeded`). The
/// handle itself is not serialized: a run resumed from saved state keeps
/// its own declines and shares them again once the application reattaches
/// a set.
///
/// It only grows. Clones share one set.
#[derive(Clone, Default)]
pub struct SharedDeclines(std::sync::Arc<std::sync::Mutex<Vec<EffectKey>>>);

impl SharedDeclines {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<EffectKey>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Add effects (duplicates are ignored).
    pub fn record(&self, effects: &[EffectKey]) {
        let mut shared = self.lock();
        for effect in effects {
            if !shared.contains(effect) {
                shared.push(effect.clone());
            }
        }
    }

    /// Every effect declined so far, in the order first recorded.
    pub fn snapshot(&self) -> Vec<EffectKey> {
        self.lock().clone()
    }

    /// Publish `local`'s effects and take in the others'. Returns whether
    /// `local` grew.
    pub(crate) fn exchange(&self, local: &mut Vec<EffectKey>) -> bool {
        let mut shared = self.lock();
        for effect in local.iter() {
            if !shared.contains(effect) {
                shared.push(effect.clone());
            }
        }
        let before = local.len();
        for effect in shared.iter() {
            if !local.contains(effect) {
                local.push(effect.clone());
            }
        }
        local.len() > before
    }

    /// Both handles share one set.
    pub fn same_set(&self, other: &SharedDeclines) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Two handles are equal when they share one set.
impl PartialEq for SharedDeclines {
    fn eq(&self, other: &Self) -> bool {
        self.same_set(other)
    }
}

impl std::fmt::Debug for SharedDeclines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SharedDeclines")
            .field(&self.snapshot())
            .finish()
    }
}
