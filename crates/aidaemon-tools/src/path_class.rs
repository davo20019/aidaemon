//! Which paths hold secrets or must not be touched by the file tools.

use std::path::{Path, PathBuf};

/// Sensitive path patterns that should not be exposed through file tools.
pub const SENSITIVE_PATTERNS: &[&str] = &[
    ".git",
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".docker",
    ".kube",
    ".pulumi",
    ".terraform",
    ".env",
    ".envrc",
    ".dev.vars",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "credentials",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".htpasswd",
    ".git-credentials",
    "secrets",
];

/// Entries that hold secrets; a file read refuses these. Narrower than
/// [`SENSITIVE_PATTERNS`], which also guards non-secret writes such as
/// `.git`.
pub const SECRET_READ_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".env",
    ".envrc",
    ".dev.vars",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "credentials",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".htpasswd",
    ".git-credentials",
    "secrets",
];

/// Classifies paths for the file tools: secrets a read refuses, and the
/// wider set of sensitive paths a write refuses. Beyond the built-in
/// patterns, an app adds its own files (its config, its env file) as
/// `protected`; they count as both.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PathClassifier {
    protected: Vec<PathBuf>,
}

impl PathClassifier {
    pub fn new(protected: Vec<PathBuf>) -> Self {
        Self { protected }
    }

    /// A classifier with no app files protected (usable in a `static`).
    pub const fn empty() -> Self {
        Self {
            protected: Vec::new(),
        }
    }

    /// The app's own files, as given.
    pub fn protected_paths(&self) -> &[PathBuf] {
        &self.protected
    }

    /// One of the app's own files (compared after resolving symlinks).
    pub fn is_protected(&self, path: &Path) -> bool {
        self.protected.iter().any(|own| same_file(own, path))
    }

    /// A file that holds secrets: keys, credentials, env files, the app's
    /// own files.
    pub fn is_secret(&self, path: &Path) -> bool {
        self.is_secret_ignoring_name_shape(path) || has_secret_name_shape(path)
    }

    /// [`PathClassifier::is_secret`] without the `.env.*` / `*.key`
    /// name-shape rules, so a template file is judged by where it lives.
    pub fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool {
        matches_patterns(path, SECRET_READ_PATTERNS) || self.is_protected(path)
    }

    /// A path the file tools must not write: everything secret, plus
    /// repository internals and infrastructure state.
    pub fn is_sensitive(&self, path: &Path) -> bool {
        matches_patterns(path, SENSITIVE_PATTERNS)
            || self.is_protected(path)
            || has_secret_name_shape(path)
    }
}

/// Whether two paths name the same file, directly or after resolving
/// symlinks.
pub fn same_file(a: &Path, b: &Path) -> bool {
    a == b
        || matches!(
            (a.canonicalize(), b.canonicalize()),
            (Ok(x), Ok(y)) if x == y
        )
}

fn matches_patterns(path: &Path, patterns: &[&str]) -> bool {
    let path_lower = path.to_string_lossy().to_ascii_lowercase();
    patterns.iter().any(|pattern| {
        if pattern.contains('/') {
            // A multi-component pattern (".aws/credentials") is specific
            // enough to match as a substring.
            path_lower.contains(&pattern.to_ascii_lowercase())
        } else {
            // A single component matches a whole path component or the
            // file name, never part of one.
            path.components().any(|c| {
                c.as_os_str()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(pattern)
            }) || path
                .file_name()
                .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(pattern))
        }
    })
}

/// Secret-shaped names: env variants, key and keystore extensions, and
/// Terraform state.
fn has_secret_name_shape(path: &Path) -> bool {
    path.components().any(|component| {
        let component = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        component.starts_with(".env.")
            || component.starts_with(".dev.vars.")
            || component.ends_with(".key")
            || component.ends_with(".pem")
            || component.ends_with(".p12")
            || component.ends_with(".pfx")
            || component.ends_with(".jks")
            || component.ends_with(".keystore")
            || component == "terraform.tfstate"
            || component.starts_with("terraform.tfstate.")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(p: &str) -> &Path {
        Path::new(p)
    }

    #[test]
    fn secrets_and_sensitive_paths() {
        let classifier = PathClassifier::default();
        for secret in [
            "/home/synthetic/.ssh/id_ed25519",
            "/srv/app/.env",
            "/srv/app/.env.production",
            "/srv/app/server.key",
            "/home/synthetic/.aws/credentials",
        ] {
            assert!(classifier.is_secret(path(secret)), "{secret}");
            assert!(classifier.is_sensitive(path(secret)), "{secret}");
        }
        // Sensitive to write, but not a secret to read.
        assert!(classifier.is_sensitive(path("/srv/app/.git/config")));
        assert!(!classifier.is_secret(path("/srv/app/.git/config")));
        assert!(classifier.is_sensitive(path("/srv/infra/terraform.tfstate")));
        // Ordinary files.
        assert!(!classifier.is_sensitive(path("/srv/app/src/main.rs")));
        // A component name, not a substring: "monkey" is not ".key".
        assert!(!classifier.is_sensitive(path("/srv/app/monkey.txt")));
    }

    #[test]
    fn name_shape_rules_can_be_ignored_for_templates() {
        let classifier = PathClassifier::default();
        let template = path("/srv/app/config/.env.example");
        assert!(classifier.is_secret(template));
        assert!(!classifier.is_secret_ignoring_name_shape(template));
    }

    #[test]
    fn the_apps_own_files_are_protected() {
        let own = PathBuf::from("/srv/synthetic-app/app.toml");
        let classifier = PathClassifier::new(vec![own.clone()]);
        assert!(classifier.is_protected(&own));
        assert!(classifier.is_secret(&own));
        assert!(classifier.is_sensitive(&own));
        assert!(!PathClassifier::default().is_sensitive(&own));
    }
}
