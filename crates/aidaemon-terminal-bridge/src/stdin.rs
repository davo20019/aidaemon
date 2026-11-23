//! Decoding escaped control sequences in client stdin for the PTY.

pub(crate) fn escaped_control_token_len(data: &str) -> Option<usize> {
    let mut chars = data.chars();
    if chars.next() != Some('\\') {
        return None;
    }
    match chars.next()? {
        'r' | 'n' | 't' | 'b' | 'f' => Some(2),
        'u' => {
            let mut len = 2usize;
            for _ in 0..4 {
                let ch = chars.next()?;
                if !ch.is_ascii_hexdigit() {
                    return None;
                }
                len += 1;
            }
            Some(len)
        }
        'x' => {
            let a = chars.next()?;
            let b = chars.next()?;
            if a.is_ascii_hexdigit() && b.is_ascii_hexdigit() {
                Some(4)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn count_escaped_control_tokens(data: &str) -> usize {
    let mut count = 0usize;
    let mut idx = 0usize;
    while idx < data.len() {
        let remaining = &data[idx..];
        if let Some(token_len) = escaped_control_token_len(remaining) {
            count = count.saturating_add(1);
            idx = idx.saturating_add(token_len);
            continue;
        }
        let Some(ch) = remaining.chars().next() else {
            break;
        };
        idx = idx.saturating_add(ch.len_utf8());
    }
    count
}

pub(crate) fn should_decode_stdin_control_escapes(data: &str) -> bool {
    let trimmed = data.trim();
    if trimmed.is_empty() {
        return false;
    }

    if let Some(token_len) = escaped_control_token_len(trimmed)
        && token_len == trimmed.len()
    {
        return true;
    }

    if trimmed.ends_with("\\r") || trimmed.ends_with("\\n") {
        return true;
    }

    if trimmed.contains("\\u001b") || trimmed.contains("\\x1b") {
        return true;
    }

    count_escaped_control_tokens(trimmed) >= 2
}

pub(crate) fn decode_stdin_control_escapes(data: &str) -> String {
    if !data.contains('\\') {
        return data.to_string();
    }

    let mut out = String::with_capacity(data.len());
    let mut chars = data.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            out.push('\\');
            break;
        };
        match next {
            'r' => out.push('\r'),
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000c}'),
            'u' => {
                let mut lookahead = chars.clone();
                let mut code = 0u32;
                let mut digits = ['\0'; 4];
                let mut ok = true;
                for slot in &mut digits {
                    let Some(hex) = lookahead.next() else {
                        ok = false;
                        break;
                    };
                    let Some(value) = hex.to_digit(16) else {
                        ok = false;
                        break;
                    };
                    *slot = hex;
                    code = (code << 4) | value;
                }
                if ok {
                    for _ in 0..4 {
                        let _ = chars.next();
                    }
                    if let Some(decoded) = char::from_u32(code) {
                        out.push(decoded);
                    } else {
                        out.push('\\');
                        out.push('u');
                        for hex in digits {
                            out.push(hex);
                        }
                    }
                } else {
                    out.push('\\');
                    out.push('u');
                }
            }
            'x' => {
                let mut lookahead = chars.clone();
                let first = lookahead.next();
                let second = lookahead.next();
                if let (Some(a), Some(b)) = (first, second) {
                    if let (Some(hi), Some(lo)) = (a.to_digit(16), b.to_digit(16)) {
                        let _ = chars.next();
                        let _ = chars.next();
                        let byte = ((hi << 4) | lo) as u8;
                        if byte.is_ascii() {
                            out.push(byte as char);
                        } else {
                            out.push('\\');
                            out.push('x');
                            out.push(a);
                            out.push(b);
                        }
                    } else {
                        out.push('\\');
                        out.push('x');
                    }
                } else {
                    out.push('\\');
                    out.push('x');
                }
            }
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    out
}

pub(crate) fn normalize_stdin_for_pty(data: &str) -> String {
    if should_decode_stdin_control_escapes(data) {
        decode_stdin_control_escapes(data)
    } else {
        data.to_string()
    }
}
