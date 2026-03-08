use crate::traits::{ToolCallSemantics, ToolVerificationMode};

fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| text.starts_with(prefix))
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

pub(crate) fn classify_shell_command(command: &str) -> ToolCallSemantics {
    let lower = command.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return ToolCallSemantics::administrative();
    }

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
