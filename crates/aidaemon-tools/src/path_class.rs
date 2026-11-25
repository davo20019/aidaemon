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
/// `.git`. Multi-component entries name token stores whose file names do
/// not look secret (a CLI's login file, a shell's history).
pub const SECRET_READ_PATTERNS: &[&str] = &[
    ".config/gh/hosts.yml",
    ".codex/auth.json",
    ".claude/.credentials.json",
    ".config/gcloud/credentials.db",
    ".config/gcloud/access_tokens.db",
    ".config/gcloud/application_default_credentials.json",
    ".docker/config.json",
    ".kube/config",
    ".bash_history",
    ".zsh_history",
    ".python_history",
    ".node_repl_history",
    ".psql_history",
    ".mysql_history",
    ".sqlite_history",
    ".lesshst",
    ".local/share/fish/fish_history",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Application Support/Microsoft Edge",
    "Library/Application Support/BraveSoftware",
    "Library/Safari",
    "Library/Cookies",
    "Library/Keychains",
    ".mozilla/firefox",
    ".config/google-chrome",
    ".config/chromium",
    ".config/BraveSoftware",
    ".password-store",
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

/// Shell start-up files: whatever they hold runs in every new shell.
const SHELL_STARTUP_FILES: &[&str] = &[
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".bash_logout",
    ".profile",
    ".zshrc",
    ".zshenv",
    ".zprofile",
    ".zlogin",
    ".zlogout",
    ".cshrc",
    ".tcshrc",
    ".kshrc",
    ".mkshrc",
    ".xprofile",
    ".xinitrc",
    ".xsessionrc",
    "microsoft.powershell_profile.ps1",
    "profile.ps1",
];

/// Folders whose files the system starts on its own (at login, on a
/// schedule, as a service), as consecutive path components.
const STARTUP_FOLDERS: &[&[&str]] = &[
    &["Library", "LaunchAgents"],
    &["Library", "LaunchDaemons"],
    &["Library", "StartupItems"],
    &["systemd", "user"],
    &["systemd", "system"],
    &[".config", "autostart"],
    &[".config", "environment.d"],
    &[".config", "fish", "conf.d"],
    &["etc", "cron.d"],
    &["etc", "cron.daily"],
    &["etc", "cron.hourly"],
    &["etc", "cron.weekly"],
    &["etc", "cron.monthly"],
    &["var", "spool", "cron"],
    &["etc", "profile.d"],
    &["Start Menu", "Programs", "Startup"],
    &[".git", "hooks"],
    &[".husky"],
];

/// Files under the daemon's own folder that later turns load as
/// instructions: the owner's `AGENTS.md` (and its override), specialist
/// prompts and skills.
const DAEMON_INSTRUCTION_ENTRIES: &[&str] =
    &["AGENTS.md", "AGENTS.override.md", "specialists", "skills"];

/// The daemon's own folder name under the home directory.
const DAEMON_DIR_NAME: &str = ".aidaemon";

/// Whether writing `path` plants something that runs or is obeyed later
/// without anyone asking: a shell start-up file, a login item, launch
/// agent, service unit, cron entry or git hook, or instructions the daemon
/// loads into every later prompt (`~/.aidaemon/AGENTS.md`, specialists,
/// skills). `daemon_dirs` are the daemon's own folders (where its config
/// lives) besides the default `~/.aidaemon`.
pub fn is_persistence_path(path: &Path, daemon_dirs: &[PathBuf]) -> bool {
    let components: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let eq = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
    if path.file_name().is_some_and(|name| {
        let name = name.to_string_lossy();
        SHELL_STARTUP_FILES.iter().any(|file| eq(&name, file))
            || eq(&name, "crontab")
            || (eq(&name, "config.fish")
                && components
                    .iter()
                    .rev()
                    .nth(1)
                    .is_some_and(|dir| eq(dir, "fish")))
    }) {
        return true;
    }
    let holds_run = |run: &[&str]| {
        components
            .windows(run.len())
            .any(|window| window.iter().zip(run).all(|(c, r)| eq(c, r)))
    };
    if STARTUP_FOLDERS.iter().any(|run| holds_run(run)) {
        return true;
    }
    // `~/.aidaemon/<entry>` and `<daemon dir>/<entry>`.
    let under_entry = |dir_len: usize| {
        components
            .get(dir_len)
            .is_some_and(|entry| DAEMON_INSTRUCTION_ENTRIES.iter().any(|e| eq(entry, e)))
    };
    if components
        .iter()
        .enumerate()
        .any(|(i, c)| eq(c, DAEMON_DIR_NAME) && under_entry(i + 1))
    {
        return true;
    }
    daemon_dirs.iter().any(|dir| {
        let dir_len = dir.components().count();
        (path.starts_with(dir)
            || dunce::canonicalize(dir)
                .ok()
                .is_some_and(|dir| path.starts_with(&dir)))
            && under_entry(dir_len)
    })
}

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
            (dunce::canonicalize(a), dunce::canonicalize(b)),
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

/// How the daemon classifies paths: the SDK's patterns, its own config and
/// env files (fixed at startup), and any env file its environment names.
///
/// The default protects no daemon files; only tests use it.
#[derive(Debug, Clone, Default)]
pub struct DaemonPaths {
    classifier: std::sync::Arc<PathClassifier>,
}

impl DaemonPaths {
    /// `protected` holds the daemon's own config and env files. They change
    /// only through `manage_config`, never through the file tools.
    pub fn new(protected: Vec<PathBuf>) -> Self {
        Self {
            classifier: std::sync::Arc::new(PathClassifier::new(protected)),
        }
    }

    /// True for files that hold secrets (keys, credentials, env files, the
    /// daemon's own config). Used by `read_file`.
    pub fn is_secret(&self, path: &Path) -> bool {
        self.classifier.is_secret(path) || is_configured_env_file(path)
    }

    /// Like [`DaemonPaths::is_secret`] but without the `.env.*` / `*.key`
    /// name-shape rules, so a template file is judged by where it lives alone.
    pub fn is_secret_ignoring_name_shape(&self, path: &Path) -> bool {
        self.classifier.is_secret_ignoring_name_shape(path) || is_configured_env_file(path)
    }

    /// True for paths the file tools must not write.
    pub fn is_sensitive(&self, path: &Path) -> bool {
        self.classifier.is_sensitive(path) || is_configured_env_file(path)
    }

    /// True for the daemon's own config and env files given at startup.
    pub fn is_own_file(&self, path: &Path) -> bool {
        self.classifier.is_protected(path)
    }

    /// True for a path whose content runs or is obeyed later without
    /// anyone asking ([`is_persistence_path`]), the daemon's own folders
    /// being where its config files live.
    pub fn is_persistence(&self, path: &Path) -> bool {
        let dirs = self
            .classifier
            .protected_paths()
            .iter()
            .filter_map(|file| file.parent().map(Path::to_path_buf))
            .collect::<Vec<_>>();
        is_persistence_path(path, &dirs)
    }
}

/// Env files named by the daemon's own environment variables
/// (`AIDAEMON_ENV_FILE`, `AIDAEMON_RUNTIME_ENV_FILE`), wherever they live.
fn is_configured_env_file(path: &Path) -> bool {
    ["AIDAEMON_ENV_FILE", "AIDAEMON_RUNTIME_ENV_FILE"]
        .iter()
        .filter_map(std::env::var_os)
        .filter(|value| !value.is_empty())
        .any(|value| same_file(Path::new(&value), path))
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
    fn token_stores_and_histories_are_secrets() {
        let classifier = PathClassifier::default();
        for secret in [
            "/home/synthetic/.config/gh/hosts.yml",
            "/home/synthetic/.codex/auth.json",
            "/home/synthetic/.claude/.credentials.json",
            "/home/synthetic/.zsh_history",
            "/home/synthetic/.bash_history",
            "/home/synthetic/.netrc",
            "/home/synthetic/.docker/config.json",
            "/home/synthetic/.kube/config",
            "/Users/synthetic/Library/Application Support/Google/Chrome/Default/Cookies",
            "/home/synthetic/.mozilla/firefox/abc.default/logins.json",
        ] {
            assert!(classifier.is_secret(path(secret)), "{secret}");
        }
        assert!(!classifier.is_secret(path("/srv/app/.config/settings.yml")));
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

    // `DaemonPaths` as the daemon's file tools use it (moved from the
    // daemon's `tools::file_policy`).
    #[test]
    fn sensitive_paths_are_classified() {
        let paths = DaemonPaths::default();
        for sensitive in [
            "/home/user/.ssh/id_rsa",
            "/home/user/.env",
            "/repo/.env.production/value",
            "/repo/.dev.vars.local",
            "/repo/.git/config",
            "/repo/certificates/app.PEM",
            "/repo/terraform.tfstate.backup",
            "/home/user/.gnupg/key",
        ] {
            assert!(paths.is_sensitive(Path::new(sensitive)), "{sensitive}");
        }
        assert!(!paths.is_sensitive(Path::new("/home/user/code/main.rs")));
        assert!(!paths.is_sensitive(Path::new("/repo/src/environment.rs")));
    }

    #[test]
    fn persistence_paths_are_classified() {
        let daemon = PathBuf::from("/srv/synthetic-daemon");
        let paths = DaemonPaths::new(vec![daemon.join("config.toml")]);
        for persistent in [
            "/home/synthetic/.zshrc",
            "/home/synthetic/.bash_profile",
            "/Users/synthetic/Library/LaunchAgents/com.example.agent.plist",
            "/Library/LaunchDaemons/com.example.plist",
            "/home/synthetic/.config/systemd/user/example.service",
            "/home/synthetic/.config/autostart/example.desktop",
            "/home/synthetic/.config/fish/config.fish",
            "/etc/cron.d/example",
            "/var/spool/cron/crontabs/synthetic",
            "/srv/app/.git/hooks/pre-commit",
            "/srv/app/.husky/pre-push",
            "/home/synthetic/.aidaemon/AGENTS.md",
            "/home/synthetic/.aidaemon/specialists/coder.md",
            "/home/synthetic/.aidaemon/skills/example/SKILL.md",
            "/srv/synthetic-daemon/AGENTS.md",
            "/srv/synthetic-daemon/specialists/x.md",
        ] {
            assert!(paths.is_persistence(Path::new(persistent)), "{persistent}");
        }
        for ordinary in [
            "/srv/app/src/main.rs",
            "/srv/app/AGENTS.md",
            "/home/synthetic/.aidaemon/files/inbox/report.pdf",
            "/srv/app/profile.rs",
            "/srv/app/docs/launchagents.md",
        ] {
            assert!(!paths.is_persistence(Path::new(ordinary)), "{ordinary}");
        }
    }

    #[test]
    fn the_daemons_own_files_are_protected_only_when_given() {
        let own = Path::new("/synthetic/daemon/config.toml");
        assert!(!DaemonPaths::default().is_sensitive(own));
        let paths = DaemonPaths::new(vec![own.to_path_buf()]);
        assert!(paths.is_sensitive(own));
        assert!(paths.is_secret(own));
    }
}
