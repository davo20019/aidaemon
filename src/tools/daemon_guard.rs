use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
use base64::Engine;
use std::collections::HashSet;
use std::path::Path;

/// Daemonization-related commands that can intentionally detach execution.
const DETACH_COMMANDS: &[(&str, &str)] = &[
    ("nohup", "nohup"),
    ("setsid", "setsid"),
    ("disown", "disown"),
    ("systemd-run", "systemd-run"),
    ("launchctl", "launchctl"),
    ("crontab", "crontab"),
    ("at", "at"),
    ("screen", "screen"),
    ("tmux", "tmux"),
    ("daemonize", "daemonize"),
];

const MAX_RECURSION_DEPTH: usize = 3;
const MIN_BASE64_LEN: usize = 16;
const MAX_BASE64_LEN: usize = 8192;
const SHELL_C_WRAPPERS: &[&str] = &["bash", "sh", "zsh", "fish", "dash", "ksh", "csh", "tcsh"];
const INLINE_CODE_WRAPPERS: &[&str] = &["python", "python3", "node", "perl", "ruby", "php"];

/// Return daemonization primitives detected in input text/command.
///
/// Results are user-facing primitive labels (deduplicated, stable order).
pub fn detect_daemonization_primitives(input: &str) -> Vec<&'static str> {
    let mut hits: Vec<&'static str> = Vec::new();
    let mut visited = HashSet::new();
    detect_recursive(input, 0, &mut hits, &mut visited);
    hits
}

fn detect_recursive(
    input: &str,
    depth: usize,
    hits: &mut Vec<&'static str>,
    visited: &mut HashSet<String>,
) {
    if depth > MAX_RECURSION_DEPTH {
        return;
    }

    let normalized = input.trim().to_ascii_lowercase();
    if normalized.is_empty() || !visited.insert(normalized) {
        return;
    }

    detect_direct_primitives(input, hits);

    if depth == MAX_RECURSION_DEPTH {
        return;
    }

    for nested in extract_nested_payloads(input) {
        detect_recursive(&nested, depth + 1, hits, visited);
    }
    for decoded in extract_decoded_payloads(input) {
        detect_recursive(&decoded, depth + 1, hits, visited);
    }
}

fn detect_direct_primitives(input: &str, hits: &mut Vec<&'static str>) {
    let lower = input.to_ascii_lowercase();

    // Fast-path: shell backgrounding operator
    if contains_unquoted_background_operator(input) {
        push_unique(hits, "background operator (&)");
    }

    // Command-segment aware detection: catches "x | crontab -", "foo && nohup ...", etc.
    for (segment, _) in crate::tools::command_risk::split_by_operators(input) {
        if segment.is_empty() {
            continue;
        }
        let parts = match shell_words::split(&segment) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if parts.is_empty() {
            continue;
        }
        let base = Path::new(&parts[0])
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&parts[0])
            .to_ascii_lowercase();
        for (token, label) in DETACH_COMMANDS {
            if base == *token {
                push_unique(hits, label);
            }
        }
    }

    // Fallback lexical scan catches prompts like "run nohup python app.py &".
    for (token, label) in DETACH_COMMANDS {
        if *token == "at" {
            // Avoid excessive false positives on normal prose ("look at this").
            continue;
        }
        if contains_word_ascii_ci(&lower, token) || contains_obfuscated_word_ascii_ci(input, token)
        {
            push_unique(hits, label);
        }
    }
}

fn push_unique(out: &mut Vec<&'static str>, value: &'static str) {
    if !out.contains(&value) {
        out.push(value);
    }
}

/// Detect '&' used as a background operator outside quotes.
/// Ignores logical AND (&&) and redirection forms like 2>&1 / &>.
fn contains_unquoted_background_operator(input: &str) -> bool {
    let chars: Vec<char> = input.chars().collect();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escaped = false;

    for (i, ch) in chars.iter().enumerate() {
        let ch = *ch;
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && !in_single_quote {
            escaped = true;
            continue;
        }
        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            continue;
        }
        if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            continue;
        }
        if in_single_quote || in_double_quote || ch != '&' {
            continue;
        }

        let prev = chars.get(i.saturating_sub(1)).copied();
        let next = chars.get(i + 1).copied();
        if prev == Some('&') || next == Some('&') {
            continue; // logical AND
        }

        let prev_non_ws = chars[..i]
            .iter()
            .rev()
            .find(|c| !c.is_whitespace())
            .copied();
        let next_non_ws = chars[i + 1..].iter().find(|c| !c.is_whitespace()).copied();

        if matches!(prev_non_ws, Some('>') | Some('<'))
            || matches!(next_non_ws, Some('>') | Some('<'))
        {
            continue; // redirect operators like 2>&1 and &>
        }

        return true;
    }

    false
}

fn contains_word_ascii_ci(lower_haystack: &str, lower_word: &str) -> bool {
    if lower_word.is_empty() {
        return false;
    }
    let bytes = lower_haystack.as_bytes();
    let needle = lower_word.as_bytes();
    if needle.len() > bytes.len() {
        return false;
    }
    for i in 0..=bytes.len() - needle.len() {
        if &bytes[i..i + needle.len()] != needle {
            continue;
        }
        let left_ok = i == 0 || !is_word_char(bytes[i - 1] as char);
        let right_ok =
            i + needle.len() == bytes.len() || !is_word_char(bytes[i + needle.len()] as char);
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
}

fn contains_obfuscated_word_ascii_ci(input: &str, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let haystack: Vec<char> = input.to_ascii_lowercase().chars().collect();
    let needle: Vec<char> = token.chars().collect();
    if needle.len() > haystack.len() {
        return false;
    }

    for start in 0..haystack.len() {
        if haystack[start] != needle[0] {
            continue;
        }

        let left_sig = haystack[..start]
            .iter()
            .rev()
            .find(|c| !is_obfuscation_separator(**c))
            .copied();
        if matches!(left_sig, Some(ch) if is_word_char(ch)) {
            continue;
        }

        let mut i = start + 1;
        let mut j = 1;
        while j < needle.len() && i < haystack.len() {
            let ch = haystack[i];
            if ch == needle[j] {
                j += 1;
                i += 1;
                continue;
            }
            if is_obfuscation_separator(ch) {
                i += 1;
                continue;
            }
            break;
        }

        if j == needle.len() {
            let right_sig = haystack[i..]
                .iter()
                .find(|c| !is_obfuscation_separator(**c))
                .copied();
            if !matches!(right_sig, Some(ch) if is_word_char(ch)) {
                return true;
            }
        }
    }
    false
}

fn is_obfuscation_separator(ch: char) -> bool {
    matches!(
        ch,
        '\\' | '\'' | '"' | '`' | '$' | '{' | '}' | '(' | ')' | '[' | ']' | '+' | '.'
    )
}

fn extract_nested_payloads(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for (segment, _) in crate::tools::command_risk::split_by_operators(input) {
        if segment.is_empty() {
            continue;
        }
        let parts = match shell_words::split(&segment) {
            Ok(parts) => parts,
            Err(_) => continue,
        };
        if parts.is_empty() {
            continue;
        }

        let Some((base, cmd_index)) = primary_command(&parts) else {
            continue;
        };

        if base == "eval" {
            if cmd_index + 1 < parts.len() {
                let payload = parts[cmd_index + 1..].join(" ");
                push_unique_owned(&mut out, &mut seen, payload);
            }
            continue;
        }

        if SHELL_C_WRAPPERS.contains(&base.as_str()) {
            if let Some(payload) = extract_flag_payload(&parts, cmd_index + 1, PayloadMode::ShellC)
            {
                push_unique_owned(&mut out, &mut seen, payload);
            }
            continue;
        }

        if INLINE_CODE_WRAPPERS.contains(&base.as_str()) {
            if let Some(payload) =
                extract_flag_payload(&parts, cmd_index + 1, PayloadMode::InlineCode)
            {
                push_unique_owned(&mut out, &mut seen, payload);
            }
        }
    }

    out
}

#[derive(Clone, Copy)]
enum PayloadMode {
    ShellC,
    InlineCode,
}

fn extract_flag_payload(parts: &[String], start: usize, mode: PayloadMode) -> Option<String> {
    let mut i = start;
    while i < parts.len() {
        let token = parts[i].as_str();

        match mode {
            PayloadMode::ShellC => {
                if token == "-c" {
                    return parts.get(i + 1).cloned();
                }
                if token.starts_with("-c") && token.len() > 2 {
                    return Some(token[2..].to_string());
                }
                if has_short_flag(token, 'c') {
                    return parts.get(i + 1).cloned();
                }
            }
            PayloadMode::InlineCode => {
                if token == "-e" || token == "-c" || token == "--eval" {
                    return parts.get(i + 1).cloned();
                }
                if (token.starts_with("-e") || token.starts_with("-c")) && token.len() > 2 {
                    return Some(token[2..].to_string());
                }
                if token == "-pe" || token == "-ep" {
                    return parts.get(i + 1).cloned();
                }
            }
        }
        i += 1;
    }
    None
}

fn has_short_flag(token: &str, flag: char) -> bool {
    if !token.starts_with('-') || token.starts_with("--") || token.len() <= 2 {
        return false;
    }
    token.chars().skip(1).any(|c| c == flag)
}

fn primary_command(parts: &[String]) -> Option<(String, usize)> {
    let mut i = 0;
    while i < parts.len() {
        let base = command_basename(&parts[i]);
        match base.as_str() {
            "env" => {
                i += 1;
                while i < parts.len() {
                    let token = parts[i].as_str();
                    if token == "--" {
                        i += 1;
                        break;
                    }
                    if token.starts_with('-') || is_env_assignment(token) {
                        i += 1;
                        continue;
                    }
                    break;
                }
            }
            "sudo" | "doas" => {
                i += 1;
                while i < parts.len() {
                    let option = parts[i].as_str();
                    if option == "--" {
                        i += 1;
                        break;
                    }
                    if !option.starts_with('-') {
                        break;
                    }
                    i += 1;
                    if option_takes_value(base.as_str(), option) && i < parts.len() {
                        i += 1;
                    }
                }
            }
            "command" | "builtin" | "time" => {
                i += 1;
                while i < parts.len() && parts[i].starts_with('-') {
                    i += 1;
                }
            }
            "nice" => {
                i += 1;
                while i < parts.len() && parts[i].starts_with('-') {
                    let option = parts[i].as_str();
                    i += 1;
                    if (option == "-n" || option == "--adjustment") && i < parts.len() {
                        i += 1;
                    }
                }
            }
            _ => return Some((base, i)),
        }
    }
    None
}

fn command_basename(token: &str) -> String {
    Path::new(token)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(token)
        .to_ascii_lowercase()
}

fn is_env_assignment(token: &str) -> bool {
    !token.starts_with('-') && token.contains('=') && !token.starts_with('=')
}

fn option_takes_value(wrapper: &str, option: &str) -> bool {
    match wrapper {
        "sudo" => matches!(
            option,
            "-u" | "-g" | "-h" | "-C" | "-p" | "-r" | "-t" | "-T"
        ),
        "doas" => option == "-u",
        _ => false,
    }
}

fn extract_decoded_payloads(input: &str) -> Vec<String> {
    let lower = input.to_ascii_lowercase();
    if !has_decode_hint(&lower) {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for candidate in extract_base64_candidates(input) {
        if let Some(decoded) = decode_base64_candidate(&candidate) {
            push_unique_owned(&mut out, &mut seen, decoded);
        }
    }
    out
}

fn has_decode_hint(lower_input: &str) -> bool {
    lower_input.contains("base64 -d")
        || lower_input.contains("base64 --decode")
        || lower_input.contains("base64 -decode")
        || lower_input.contains("b64decode")
        || lower_input.contains("base64decode")
        || lower_input.contains("frombase64string")
        || lower_input.contains("decode64")
        || lower_input.contains("openssl enc -d -base64")
}

fn extract_base64_candidates(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for (segment, _) in crate::tools::command_risk::split_by_operators(input) {
        if segment.is_empty() {
            continue;
        }
        let parts = match shell_words::split(&segment) {
            Ok(parts) => parts,
            Err(_) => continue,
        };
        for part in parts {
            maybe_push_base64_candidate(&part, &mut out, &mut seen);
        }
    }

    let mut run = String::new();
    for ch in input.chars() {
        if is_base64_char(ch) {
            run.push(ch);
        } else if !run.is_empty() {
            maybe_push_base64_candidate(&run, &mut out, &mut seen);
            run.clear();
        }
    }
    if !run.is_empty() {
        maybe_push_base64_candidate(&run, &mut out, &mut seen);
    }

    out
}

fn maybe_push_base64_candidate(token: &str, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    let cleaned = token.trim_matches(|c: char| {
        matches!(
            c,
            '"' | '\'' | '`' | '$' | '(' | ')' | '{' | '}' | '[' | ']' | ';' | ','
        )
    });
    if !is_plausible_base64(cleaned) {
        return;
    }
    if seen.insert(cleaned.to_string()) {
        out.push(cleaned.to_string());
    }
}

fn is_base64_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '-' | '_')
}

fn is_plausible_base64(s: &str) -> bool {
    if s.len() < MIN_BASE64_LEN || s.len() > MAX_BASE64_LEN {
        return false;
    }
    if s.starts_with('-') || s.len() % 4 == 1 {
        return false;
    }
    if !s.chars().all(is_base64_char) {
        return false;
    }

    let mut has_lower = false;
    let mut has_upper = false;
    let mut has_digit = false;
    let mut has_symbol = false;

    for ch in s.chars() {
        if ch.is_ascii_lowercase() {
            has_lower = true;
        } else if ch.is_ascii_uppercase() {
            has_upper = true;
        } else if ch.is_ascii_digit() {
            has_digit = true;
        } else {
            has_symbol = true;
        }
    }

    if !(has_digit && (has_lower || has_upper)) {
        return false;
    }

    // For shorter tokens, require mixed case+digits to avoid treating common words as base64.
    if s.len() < 24 && !(has_lower && has_upper && has_digit) {
        return false;
    }

    // Allow symbol-free payloads if length is sufficiently high.
    has_symbol || s.len() >= 20
}

fn decode_base64_candidate(candidate: &str) -> Option<String> {
    if !is_plausible_base64(candidate) {
        return None;
    }

    let mut attempts = vec![candidate.to_string()];
    if candidate.len() % 4 != 0 {
        let pad_len = (4 - (candidate.len() % 4)) % 4;
        if pad_len > 0 {
            attempts.push(format!("{}{}", candidate, "=".repeat(pad_len)));
        }
    }

    for encoded in attempts {
        for decoded in [
            STANDARD.decode(encoded.as_bytes()),
            STANDARD_NO_PAD.decode(encoded.as_bytes()),
            URL_SAFE.decode(encoded.as_bytes()),
            URL_SAFE_NO_PAD.decode(encoded.as_bytes()),
        ] {
            let Ok(bytes) = decoded else {
                continue;
            };
            if bytes.is_empty() || bytes.len() > MAX_BASE64_LEN {
                continue;
            }
            let Ok(text) = String::from_utf8(bytes) else {
                continue;
            };
            if looks_like_text_payload(&text) {
                return Some(text);
            }
        }
    }

    None
}

fn looks_like_text_payload(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let total = text.chars().count();
    if total == 0 {
        return false;
    }
    let printable = text
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
        .count();
    printable * 100 / total >= 85
}

fn push_unique_owned(out: &mut Vec<String>, seen: &mut HashSet<String>, value: String) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return;
    }
    if seen.insert(trimmed.to_string()) {
        out.push(trimmed.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_background_operator() {
        let hits = detect_daemonization_primitives("python app.py &");
        assert!(hits.contains(&"background operator (&)"));
    }

    #[test]
    fn ignores_logical_and_and_redirection_ampersands() {
        let hits = detect_daemonization_primitives("echo ok && echo done 2>&1");
        assert!(!hits.contains(&"background operator (&)"));
    }

    #[test]
    fn detects_detach_commands() {
        let hits = detect_daemonization_primitives("nohup ./server; systemd-run --user myjob");
        assert!(hits.contains(&"nohup"));
        assert!(hits.contains(&"systemd-run"));
    }

    #[test]
    fn avoids_at_false_positive_in_plain_text() {
        let hits = detect_daemonization_primitives("look at this output please");
        assert!(!hits.contains(&"at"));
    }

    #[test]
    fn detects_eval_wrapped_obfuscated_detach() {
        let hits = detect_daemonization_primitives(r#"eval "no\hup sleep 5 &""#);
        assert!(hits.contains(&"nohup"));
        assert!(hits.contains(&"background operator (&)"));
    }

    #[test]
    fn detects_base64_decoded_detach_payload() {
        let hits =
            detect_daemonization_primitives(r#"eval "$(echo bm9odXAgc2xlZXAgNSAm | base64 -d)""#);
        assert!(hits.contains(&"nohup"));
        assert!(hits.contains(&"background operator (&)"));
    }

    #[test]
    fn detects_python_b64decode_detach_payload() {
        let hits = detect_daemonization_primitives(
            r#"python3 -c "import base64;exec(base64.b64decode('bm9odXAgc2xlZXAgNSAm').decode())""#,
        );
        assert!(hits.contains(&"nohup"));
        assert!(hits.contains(&"background operator (&)"));
    }

    #[test]
    fn does_not_flag_eval_without_detach() {
        let hits = detect_daemonization_primitives(r#"eval "echo safe""#);
        assert!(!hits.contains(&"nohup"));
        assert!(!hits.contains(&"background operator (&)"));
    }
}
