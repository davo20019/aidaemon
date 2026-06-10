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

#[derive(Debug)]
struct ShellStructure {
    segments: Vec<String>,
    has_output_redirection: bool,
}

fn parse_shell_structure(command: &str) -> Option<ShellStructure> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut has_output_redirection = false;

    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            current.push(ch);
            continue;
        }
        if let Some(active_quote) = quote {
            current.push(ch);
            if ch == active_quote {
                quote = None;
            }
            continue;
        }
        if matches!(ch, '\'' | '"') {
            quote = Some(ch);
            current.push(ch);
            continue;
        }

        let is_separator = match ch {
            ';' => true,
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                true
            }
            '|' => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                true
            }
            '>' => {
                has_output_redirection = true;
                if matches!(chars.peek(), Some('>') | Some('|')) {
                    chars.next();
                }
                false
            }
            _ => false,
        };
        if is_separator {
            let segment = current.trim();
            if segment.is_empty() {
                return None;
            }
            segments.push(segment.to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if escaped || quote.is_some() {
        return None;
    }
    let tail = current.trim();
    if !tail.is_empty() {
        segments.push(tail.to_string());
    }
    if segments.is_empty() {
        return None;
    }
    Some(ShellStructure {
        segments,
        has_output_redirection,
    })
}

/// Remove redirections that discard or duplicate streams without touching
/// the filesystem (`2>/dev/null`, `>/dev/null`, `&>/dev/null`, `2>&1`).
/// They are pure noise suppression, not mutations.
fn strip_non_mutating_redirections(command: &str) -> String {
    static NON_MUTATING_REDIRECT: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| {
            regex::Regex::new(r"(?:[0-9]+|&)?>{1,2}\s*/dev/null|[0-9]+>&[0-9]+").unwrap()
        });
    NON_MUTATING_REDIRECT.replace_all(command, " ").into_owned()
}

pub(crate) fn classify_shell_command(command: &str) -> ToolCallSemantics {
    let command = strip_non_mutating_redirections(command);
    let Some(structure) = parse_shell_structure(&command) else {
        return ToolCallSemantics::mutation();
    };
    let mut observes = false;
    let mut mutates = structure.has_output_redirection;
    for segment in structure.segments {
        let semantics = classify_simple_shell_command(&segment);
        observes |= semantics.observes_state();
        mutates |= semantics.mutates_state();
    }
    match (observes, mutates) {
        (true, true) => ToolCallSemantics::observation_and_mutation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        (true, false) => ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        (false, true) => ToolCallSemantics::mutation(),
        (false, false) => ToolCallSemantics::administrative(),
    }
}

fn classify_simple_shell_command(command: &str) -> ToolCallSemantics {
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

    if lower == "cd" || lower.starts_with("cd ") {
        return ToolCallSemantics::administrative();
    }

    // Document text extractors recommended by read_file's PDF/Word stubs.
    // `pdftotext <in> -` streams to stdout (pure observation); without the
    // trailing `-` it writes a .txt next to the input (mutation).
    if lower.starts_with("pdftotext ") {
        return if lower.ends_with(" -") {
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent)
        } else {
            ToolCallSemantics::mutation()
        };
    }

    if matches!(
        lower.as_ref(),
        "ls" | "pwd"
            | "cat"
            | "head"
            | "tail"
            | "find"
            | "rg"
            | "grep"
            | "stat"
            | "wc"
            | "date"
            | "uname"
            | "whoami"
            | "hostname"
            | "uptime"
            | "env"
            | "printenv"
            | "echo"
            | "tree"
            | "du"
            | "df"
            | "file"
            | "diff"
            | "sort"
            | "uniq"
    ) {
        return ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
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
            "mdls ",
            "mdfind ",
            "locate ",
            "strings ",
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
    fn document_extractors_classify_as_observation() {
        // These are the exact commands read_file's PDF/Word stubs recommend —
        // they must classify as observation or the text-only gate blocks the
        // very command we told the model to run.
        for cmd in [
            "pdftotext -layout \"/Users/x/Downloads/Offer Letter (1).pdf\" -",
            "mdls -raw -name kMDItemTextContent \"/Users/x/file.pdf\"",
            "mdfind -name \"Offer Letter\"",
            "strings /tmp/blob.bin",
        ] {
            let semantics = classify_shell_command(cmd);
            assert!(
                semantics.observes_state() && !semantics.mutates_state(),
                "expected pure observation for: {cmd}"
            );
        }
    }

    #[test]
    fn pdftotext_without_stdout_target_still_mutates() {
        // `pdftotext report.pdf` writes report.txt next to the input.
        let semantics = classify_shell_command("pdftotext report.pdf");
        assert!(semantics.mutates_state());
    }

    #[test]
    fn dev_null_redirect_does_not_count_as_mutation() {
        // `find ... 2>/dev/null` is the canonical noise-suppressed lookup;
        // discarding output mutates nothing.
        let semantics = classify_shell_command("find ~ -name \"*Offer Letter*.pdf\" 2>/dev/null");
        assert!(semantics.observes_state());
        assert!(!semantics.mutates_state());
    }

    #[test]
    fn stderr_to_stdout_dup_does_not_count_as_mutation() {
        let semantics = classify_shell_command("ls -la /tmp 2>&1");
        assert!(semantics.observes_state());
        assert!(!semantics.mutates_state());
    }

    #[test]
    fn redirect_to_real_file_still_mutates() {
        let semantics = classify_shell_command("ls -la > files.txt");
        assert!(semantics.mutates_state());
    }

    #[test]
    fn append_to_dev_null_does_not_mutate() {
        let semantics = classify_shell_command("grep -r foo . >> /dev/null");
        assert!(!semantics.mutates_state());
    }

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

    #[test]
    fn output_redirection_is_state_mutating() {
        for command in [
            "echo value > file",
            "cat > file",
            "echo value >> file",
            "echo value 2> errors.log",
            "echo value &> all.log",
        ] {
            assert!(
                classify_shell_command(command).mutates_state(),
                "{command} must be mutating"
            );
        }
    }

    #[test]
    fn compound_and_pipeline_semantics_include_every_segment() {
        let mixed = classify_shell_command("ls && rm file");
        assert!(mixed.observes_state());
        assert!(mixed.mutates_state());

        let observation = classify_shell_command("rg pattern file | head");
        assert!(observation.observes_state());
        assert!(!observation.mutates_state());

        let tee = classify_shell_command("rg pattern file | tee output.txt");
        assert!(tee.observes_state());
        assert!(tee.mutates_state());
    }

    #[test]
    fn quoted_and_escaped_operators_are_not_shell_structure() {
        for command in [
            "echo 'value > file'",
            "echo \"a | b && c; d\"",
            r"echo value \> file",
        ] {
            let semantics = classify_shell_command(command);
            assert!(semantics.observes_state(), "{command} should observe");
            assert!(
                !semantics.mutates_state(),
                "{command} should not be classified as a mutation"
            );
        }
    }

    #[test]
    fn ambiguous_shell_expression_is_conservatively_mutating() {
        let semantics = classify_shell_command("echo 'unterminated");
        assert!(semantics.mutates_state());
        assert!(!semantics.observes_state());
    }
}
