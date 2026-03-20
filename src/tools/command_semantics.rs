use crate::traits::{ToolCallSemantics, ToolVerificationMode};

fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| text.starts_with(prefix))
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

/// Strip a leading `cd <dir> &&` or `cd <dir>;` prefix from a lowercased
/// command string.  Returns a `Cow` so we avoid allocating when there is
/// nothing to strip.
fn strip_leading_cd(cmd: &str) -> std::borrow::Cow<'_, str> {
    let trimmed = cmd.trim_start();
    if !trimmed.starts_with("cd ") {
        return std::borrow::Cow::Borrowed(cmd);
    }
    // Find the separator (`&&` or `;`) that ends the `cd` invocation.
    for sep in &[" && ", "; "] {
        if let Some(pos) = trimmed.find(sep) {
            let rest = trimmed[pos + sep.len()..].trim_start();
            if !rest.is_empty() {
                return std::borrow::Cow::Owned(rest.to_string());
            }
        }
    }
    // No separator found — the whole command is just `cd <dir>`.
    std::borrow::Cow::Borrowed(cmd)
}

pub(crate) fn classify_shell_command(command: &str) -> ToolCallSemantics {
    let lower = command.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return ToolCallSemantics::administrative();
    }

    // Strip leading `cd <dir> &&` or `cd <dir>;` prefix so the classification
    // is based on the actual command, not the directory change.  The terminal
    // tool auto-injects `cd /project_dir && <command>` for path context, which
    // would otherwise cause every injected command to fall through to the
    // default mutation classification.
    let lower = strip_leading_cd(&lower);

    if lower.starts_with("curl ") || lower.starts_with("wget ") {
        let mutating_request = contains_any(
            &lower,
            &[
                " -x post",
                " --request post",
                " -x put",
                " --request put",
                " -x patch",
                " --request patch",
                " -x delete",
                " --request delete",
                " -d ",
                " --data",
                " --upload-file",
            ],
        );
        if mutating_request {
            return ToolCallSemantics::mutation();
        }
        return ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
    }

    if starts_with_any(
        &lower,
        &[
            "ls",
            "pwd",
            "cat ",
            "head ",
            "tail ",
            "find ",
            "rg ",
            "grep ",
            "stat ",
            "wc ",
            "date",
            "uname",
            "whoami",
            "hostname",
            "uptime",
            "ps ",
            "env",
            "printenv",
            "echo ",
            "test ",
            "git status",
            "git remote",
            "git log",
            "git diff",
            "git show",
            "git branch",
            "git tag",
            "git rev-parse",
            "git shortlog",
            "git blame",
            "cargo tree",
            "cargo metadata",
            "npm audit",
            "npm outdated",
            "npm ls",
            "tree",
            "du ",
            "df ",
            "file ",
            "diff ",
            "sort ",
            "uniq ",
        ],
    ) {
        return ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
    }

    if starts_with_any(
        &lower,
        &[
            "cargo test",
            "cargo check",
            "cargo clippy",
            "cargo fmt --check",
            "pytest",
            "python -m pytest",
            "python3 -m pytest",
            "python ",
            "python3 ",
            "jest",
            "vitest",
            "go test",
            "npm test",
            "yarn test",
            "bun test",
        ],
    ) {
        return ToolCallSemantics::observation_and_mutation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
    }

    if starts_with_any(&lower, &["npm run ", "yarn run ", "bun run "]) {
        if contains_any(
            &lower,
            &[
                " test",
                " lint",
                " check",
                " typecheck",
                " audit",
                " verify",
            ],
        ) {
            return ToolCallSemantics::observation_and_mutation()
                .with_verification_mode(ToolVerificationMode::ResultContent);
        }
        return ToolCallSemantics::mutation();
    }

    if starts_with_any(
        &lower,
        &[
            "cargo build",
            "cargo run",
            "cargo fmt",
            "cargo bench",
            "cargo doc",
            "npm install",
            "yarn add",
            "bun add",
            "go build",
            "go generate",
            "make ",
            "cmake",
            "gradle",
            "mvn",
        ],
    ) {
        return ToolCallSemantics::mutation();
    }

    ToolCallSemantics::mutation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_leading_cd() {
        // No cd prefix — returned as-is
        assert_eq!(
            strip_leading_cd("python3 foo.py").as_ref(),
            "python3 foo.py"
        );

        // cd with &&
        assert_eq!(
            strip_leading_cd("cd /home/user/project && python3 foo.py").as_ref(),
            "python3 foo.py"
        );

        // cd with ;
        assert_eq!(strip_leading_cd("cd /tmp; ls -la").as_ref(), "ls -la");

        // Just cd, no following command
        assert_eq!(strip_leading_cd("cd /home/user").as_ref(), "cd /home/user");

        // Nested cd (only strips first)
        assert_eq!(
            strip_leading_cd("cd /a && cd /b && echo hi").as_ref(),
            "cd /b && echo hi"
        );
    }

    #[test]
    fn test_classify_with_cd_prefix() {
        // Without cd prefix — observation
        let sem = classify_shell_command("python3 -c 'print(1)'");
        assert!(sem.observes_state());

        // With cd prefix — should still be classified correctly
        let sem = classify_shell_command("cd /home/user/project && python3 -c 'print(1)'");
        assert!(
            sem.observes_state(),
            "cd-prefixed python3 should be observation"
        );

        let sem = classify_shell_command("cd /tmp && ls -la");
        assert!(sem.observes_state(), "cd-prefixed ls should be observation");
        assert!(
            !sem.mutates_state(),
            "cd-prefixed ls should NOT be mutation"
        );

        let sem = classify_shell_command("cd /project && cargo test");
        assert!(
            sem.observes_state(),
            "cd-prefixed cargo test should observe"
        );

        let sem = classify_shell_command("cd /project && cargo build");
        assert!(
            sem.mutates_state(),
            "cd-prefixed cargo build should be mutation"
        );
    }
}
