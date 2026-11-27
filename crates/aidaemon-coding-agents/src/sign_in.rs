//! How a coding agent signs in to its own service when the owner can
//! choose: its own saved sign-in (the default), or credentials the owner
//! gave AIdaemon to hand it at launch.
//!
//! An agent offers its choices ([`CodingAgent::sign_ins`]) and turns the
//! chosen one into what its launch adds ([`CodingAgent::signed_in`]):
//! arguments after its own, and variables set or removed. Its own sign-in
//! adds nothing, so a run is exactly what it was before the choice.
//!
//! [`CodingAgent::sign_ins`]: crate::CodingAgent::sign_ins
//! [`CodingAgent::signed_in`]: crate::CodingAgent::signed_in

/// A way an agent can sign in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SignInMethod {
    /// Its own saved sign-in, whatever it is.
    #[default]
    Own,
    /// A Gemini API key AIdaemon holds and hands it.
    GeminiApiKey,
}

impl SignInMethod {
    /// Its name in `config.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::GeminiApiKey => "gemini_api_key",
        }
    }

    /// The method a config value names.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "own" => Some(Self::Own),
            "gemini_api_key" => Some(Self::GeminiApiKey),
            _ => None,
        }
    }
}

/// The sign-in a launch uses, with its credential.
#[derive(Clone, Default, PartialEq, Eq)]
pub enum SignIn {
    /// Its own saved sign-in: the launch adds nothing.
    #[default]
    Own,
    /// This Gemini API key.
    GeminiApiKey(String),
}

impl SignIn {
    pub fn method(&self) -> SignInMethod {
        match self {
            Self::Own => SignInMethod::Own,
            Self::GeminiApiKey(_) => SignInMethod::GeminiApiKey,
        }
    }
}

/// Never prints a credential.
impl std::fmt::Debug for SignIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.method().as_str())
    }
}

/// What a sign-in adds to an agent's launch.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct SignedIn {
    /// After the agent's own arguments.
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub env_remove: Vec<String>,
}

/// Never prints a credential (the values it sets).
impl std::fmt::Debug for SignedIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignedIn")
            .field("args", &self.args)
            .field(
                "env",
                &self.env.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            )
            .field("env_remove", &self.env_remove)
            .finish()
    }
}

/// A short word about its own sign-in, for the owner, with where to read
/// more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignInNote {
    pub text: &'static str,
    pub link: &'static str,
}

/// Whether the embedding agent may hand an agent work on a sign-in: the
/// agent maker's terms for its consumer plans, declared once by its adapter
/// and checked by every delegated launch. A run the owner drives directly
/// (their own prompts, passed through) is never refused by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delegation {
    /// Its maker allows it.
    Allowed,
    /// Allowed, with a word for the owner (a grey area in its maker's terms).
    Warned(SignInNote),
    /// Its maker's terms prohibit it: only the owner drives it on this sign-in.
    Refused(SignInNote),
}

impl Delegation {
    /// The word for the owner, when there is one.
    pub fn note(self) -> Option<SignInNote> {
        match self {
            Self::Allowed => None,
            Self::Warned(note) | Self::Refused(note) => Some(note),
        }
    }

    pub fn is_refused(self) -> bool {
        matches!(self, Self::Refused(_))
    }
}

/// The Google credential variables a Gemini-based agent reads besides
/// `GEMINI_API_KEY` (which a key sign-in sets; a removal wins over a set):
/// removed so the key AIdaemon hands it is the only credential it sees.
pub(crate) const GOOGLE_CREDENTIAL_VARS: &[&str] = &[
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI",
];
