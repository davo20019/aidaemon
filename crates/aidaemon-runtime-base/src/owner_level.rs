//! The owner's permission level, below the authority that resolves it so
//! every layer can name it.

/// How much the owner's turns may do: AIdaemon's permission level
/// (`[authority] owner_default`, kept live by [`crate::permissions`]), or a
/// chat's `/mode`.
///
/// - `read_only`: tools that only read; anything else is refused.
/// - `full` (shown as Standard; `standard` is accepted): every tool, with
///   risky terminal commands asked first.
/// - `full_access`: as `full`, with every terminal command run unasked.
/// - `unrestricted` (a chat's `/mode` only): as `full`, and outward actions
///   taken after the turn read untrusted content are recorded instead of
///   checked.
///
/// Ordered by what it allows: a child takes the lower of its parent's and
/// its own.
#[derive(
    Debug,
    serde::Deserialize,
    serde::Serialize,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
)]
#[serde(rename_all = "snake_case")]
pub enum OwnerLevel {
    ReadOnly,
    #[default]
    #[serde(alias = "standard")]
    Full,
    FullAccess,
    Unrestricted,
}

impl OwnerLevel {
    /// Its name in `config.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Full => "full",
            Self::FullAccess => "full_access",
            Self::Unrestricted => "unrestricted",
        }
    }

    /// Its name in Settings.
    pub fn title(self) -> &'static str {
        match self {
            Self::ReadOnly => "Read only",
            Self::Full => "Standard",
            Self::FullAccess => "Full access",
            Self::Unrestricted => "Unrestricted",
        }
    }
}
