//! The project a turn works in.

use std::path::PathBuf;

/// The project a turn works in: a unit of work the host keeps (an app's
/// project, a bot placed in one), with the folder it lives in when it has
/// one. Memory and tools key project-scoped state on [`ProjectRef::id`],
/// which stays the same when the project is renamed or its folder moves.
///
/// The host resolves it from its own typed state (which project a
/// conversation is placed in, which folder a turn is bound to), never from
/// what a message says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectRef {
    /// Stable id. Only ASCII letters, digits, `-` and `_` (see
    /// [`ProjectRef::is_valid_id`]), so it is safe as a directory name.
    pub id: String,
    /// The name the owner gave it.
    pub name: String,
    /// The folder, absolute, when the project has one.
    pub root: Option<PathBuf>,
}

impl ProjectRef {
    /// A project, when `id` is a valid project id.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        root: Option<PathBuf>,
    ) -> Option<Self> {
        let id = id.into();
        Self::is_valid_id(&id).then(|| Self {
            id,
            name: name.into(),
            root,
        })
    }

    /// 1–128 ASCII letters, digits, `-` or `_`, not starting with `-`: a
    /// name that cannot leave the directory it is joined to.
    pub fn is_valid_id(id: &str) -> bool {
        !id.is_empty()
            && id.len() <= 128
            && !id.starts_with('-')
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_cannot_leave_their_directory() {
        assert!(ProjectRef::is_valid_id(
            "project-0a1b2c3d-0000-4000-8000-000000000001"
        ));
        assert!(ProjectRef::is_valid_id("acme_site"));
        for bad in [
            "",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "-rf",
            "a b",
            "é",
            &"x".repeat(129),
        ] {
            assert!(!ProjectRef::is_valid_id(bad), "{bad:?}");
        }
        assert!(ProjectRef::new("../etc", "x", None).is_none());
    }
}
