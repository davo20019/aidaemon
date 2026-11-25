//! Running git in a repository the daemon does not own: one an agent, a peer
//! or a cloned project can write to.
//!
//! The repository's own configuration (`.git/config`, a worktree's
//! `config.worktree`, and the files they include) can name programs git then
//! runs on its own: hooks, an fsmonitor, a signing program, and the drivers
//! `.gitattributes` selects (`filter.<name>.clean` / `smudge` / `process`
//! on `add`, `status`, `diff` and checkout; `diff.<name>.textconv` /
//! `command` and `diff.external` on `diff`; `merge.<name>.driver` on merge).
//! Git has no switch that turns repository drivers off, and a `-c` override
//! cannot unset a key, only replace its value. So:
//!
//! - fixed keys get a fixed harmless value ([`FIXED_OVERRIDES`]);
//! - every driver the repository's own scopes define is listed first
//!   ([`DRIVER_LISTING_ARGS`], reading configuration runs nothing) and each
//!   of its commands is replaced with an identity (`cat`), a no-op (`true`,
//!   `false`) or, for `process`, the empty value git treats as "none".
//!
//! Drivers from the owner's global or system configuration (git-lfs, say)
//! are the owner's own and stay as they are. Overriding by value rather than
//! pointing `GIT_CONFIG_*` at an empty file is deliberate: those variables
//! only cover the global and system files, never the repository's.

/// Overrides for keys that name a program, whatever the repository.
pub const FIXED_OVERRIDES: &[&str] = &[
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "commit.gpgSign=false",
    "tag.gpgSign=false",
    "log.showSignature=false",
];

/// `git <these>`: the repository's driver definitions, NUL-separated with
/// the scope each came from (`git config --show-scope -z`).
pub const DRIVER_LISTING_ARGS: &[&str] = &[
    "config",
    "--show-scope",
    "-z",
    "--get-regexp",
    r"^(filter|diff|merge)\.",
];

/// The `-c` arguments (`["-c", "key=value", ...]`) to put before a git
/// subcommand run in an untrusted repository. `listing` is the stdout of
/// `git` with [`DRIVER_LISTING_ARGS`] in that repository (empty when it
/// failed, as it does with no driver defined). `Err` when a driver name
/// cannot be overridden on a command line (it holds `=` or a newline):
/// the caller must not run git there.
pub fn untrusted_repo_config_args(listing: &[u8]) -> Result<Vec<String>, String> {
    let mut overrides: Vec<String> = FIXED_OVERRIDES.iter().map(|s| s.to_string()).collect();
    let text = String::from_utf8_lossy(listing);
    // Machine format: `scope NUL key LF value NUL`, repeated.
    let mut fields = text.split('\0');
    while let (Some(scope), Some(entry)) = (fields.next(), fields.next()) {
        // Only the repository's own scopes; `command` is ours.
        if !matches!(scope, "local" | "worktree") {
            continue;
        }
        let key = entry.split('\n').next().unwrap_or_default();
        let Some((section, rest)) = key.split_once('.') else {
            continue;
        };
        // A repository's own external diff program: replaced with a no-op
        // (an empty value would make git try to run ""). Only when the
        // repository sets one, so an ordinary diff keeps its output.
        if section.eq_ignore_ascii_case("diff") && rest.eq_ignore_ascii_case("external") {
            let arg = "diff.external=true".to_string();
            if !overrides.contains(&arg) {
                overrides.push(arg);
            }
            continue;
        }
        // The subsection is everything between the first and last dot.
        let Some((name, _variable)) = rest.rsplit_once('.') else {
            continue;
        };
        let replacements: &[(&str, &str)] = match section.to_ascii_lowercase().as_str() {
            "filter" => &[
                ("clean", "cat"),
                ("smudge", "cat"),
                ("process", ""),
                ("required", "false"),
            ],
            "diff" => &[("textconv", "cat"), ("command", "true")],
            "merge" => &[("driver", "false")],
            _ => continue,
        };
        if name.contains('=') || name.contains('\n') {
            return Err(format!(
                "the repository defines a git {section} driver named {name:?}, which cannot \
                 be turned off; git was not run there"
            ));
        }
        for (variable, value) in replacements {
            let arg = format!("{section}.{name}.{variable}={value}");
            if !overrides.contains(&arg) {
                overrides.push(arg);
            }
        }
    }
    Ok(overrides
        .into_iter()
        .flat_map(|arg| ["-c".to_string(), arg])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(args: &[String]) -> Vec<&str> {
        args.chunks(2)
            .map(|pair| {
                assert_eq!(pair[0], "-c");
                pair[1].as_str()
            })
            .collect()
    }

    #[test]
    fn repository_drivers_are_replaced_and_the_owners_are_kept() {
        let listing = b"local\0filter.evil.clean\ntouch pwned; cat\0\
            local\0filter.a.b.smudge\nx\0\
            worktree\0diff.Evil.textconv\nx\0\
            local\0merge.m.driver\nx\0\
            global\0filter.lfs.process\ngit-lfs filter-process\0\
            local\0diff.external\nevil\0\
            local\0core.editor\nvi\0";
        let args = untrusted_repo_config_args(listing).unwrap();
        let set = pairs(&args);
        for expected in [
            "core.hooksPath=/dev/null",
            "filter.evil.clean=cat",
            "filter.evil.smudge=cat",
            "filter.evil.process=",
            "filter.evil.required=false",
            "filter.a.b.clean=cat",
            "diff.Evil.textconv=cat",
            "diff.Evil.command=true",
            "merge.m.driver=false",
            "diff.external=true",
        ] {
            assert!(set.contains(&expected), "{expected}: {set:?}");
        }
        assert!(!set.iter().any(|arg| arg.contains("lfs")), "{set:?}");
        // One override set per driver, however many of its keys are listed.
        assert_eq!(
            set.iter().filter(|a| a.starts_with("filter.evil.")).count(),
            4
        );
    }

    #[test]
    fn no_listing_still_gets_the_fixed_overrides() {
        let args = untrusted_repo_config_args(b"").unwrap();
        assert_eq!(pairs(&args), FIXED_OVERRIDES);
        // An ordinary diff keeps its output: no external diff is forced.
        assert!(
            !pairs(&args)
                .iter()
                .any(|arg| arg.starts_with("diff.external"))
        );
    }

    #[test]
    fn a_name_that_cannot_be_overridden_refuses() {
        assert!(untrusted_repo_config_args(b"local\0filter.a=b.clean\nx\0").is_err());
    }

    /// End to end with the real git, when it is installed: a repository
    /// whose filter, textconv and diff command would each leave a mark runs
    /// none of them under the overrides.
    #[test]
    fn real_git_runs_no_repository_driver() {
        use std::process::Command;
        if Command::new("git").arg("--version").output().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("r");
        let marks = dir.path().join("marks");
        std::fs::create_dir_all(&marks).unwrap();
        let git = |args: &[&str]| {
            Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(args)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", dir.path().join("none"))
                .output()
                .unwrap()
        };
        std::fs::create_dir_all(&repo).unwrap();
        assert!(git(&["init", "-q"]).status.success());
        let mark = |name: &str| format!("touch '{}'; cat", marks.join(name).display());
        for (key, value) in [
            ("filter.evil.clean", mark("clean")),
            ("filter.evil.smudge", mark("smudge")),
            ("filter.evil.process", mark("process")),
            ("diff.evil.textconv", mark("textconv")),
            ("diff.evil.command", mark("command")),
            ("user.name", "Synthetic".to_string()),
            ("user.email", "synthetic@example.com".to_string()),
        ] {
            assert!(git(&["config", key, &value]).status.success());
        }
        std::fs::write(repo.join(".gitattributes"), "*.txt filter=evil diff=evil\n").unwrap();
        std::fs::write(repo.join("a.txt"), "one\n").unwrap();

        let listing = git(DRIVER_LISTING_ARGS).stdout;
        let guard = untrusted_repo_config_args(&listing).unwrap();
        let guarded = |args: &[&str]| {
            let mut full: Vec<&str> = guard.iter().map(String::as_str).collect();
            full.extend(args);
            git(&full)
        };
        assert!(guarded(&["add", "-A"]).status.success());
        assert!(guarded(&["commit", "-qm", "one"]).status.success());
        std::fs::write(repo.join("a.txt"), "two\n").unwrap();
        assert!(guarded(&["status", "--porcelain"]).status.success());
        assert!(guarded(&["diff"]).status.success());
        // A repository's diff command is a no-op under the overrides; the
        // daemon's own diffs pass `--no-ext-diff` and keep their output.
        let diff = guarded(&["diff", "--no-ext-diff"]);
        assert!(diff.status.success());
        assert!(
            String::from_utf8_lossy(&diff.stdout).contains("a.txt"),
            "the diff still shows the change"
        );
        assert!(
            guarded(&["restore", "--source=HEAD", "--worktree", "--", "a.txt"])
                .status
                .success()
        );
        let left: Vec<_> = std::fs::read_dir(&marks).unwrap().collect();
        assert!(left.is_empty(), "a repository driver ran: {left:?}");
    }
}
