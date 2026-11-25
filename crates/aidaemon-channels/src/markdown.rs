//! Inline Markdown as the models write it, parsed once for every chat's
//! formatter (Slack mrkdwn, Telegram HTML, plain text): each renders the
//! same [`Inline`] tree its own way, so the rules cannot drift apart.
//!
//! The rules: `` `code` ``; `**bold**` and `__bold__`; `~~strike~~`;
//! `*italic*` (no space just inside either star) and `_italic_` (only at
//! word boundaries, so `snake_case` stays text); `[text](url)` with a
//! non-empty URL holding no whitespace. Anything else is text.

/// One piece of a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inline {
    Text(String),
    Code(String),
    Bold(Vec<Inline>),
    Italic(Vec<Inline>),
    Strike(Vec<Inline>),
    Link {
        text: Vec<Inline>,
        url: String,
    },
    /// A `<...>` token the caller asked to keep whole (see
    /// [`parse_with`]), without its brackets.
    Angle(String),
}

/// Parse one line of inline Markdown.
pub fn parse(text: &str) -> Vec<Inline> {
    parse_with(text, |_| false)
}

/// [`parse`], keeping `<inner>` whole as [`Inline::Angle`] wherever
/// `keep_angle(inner)` is true (Slack's mentions and autolinks).
pub fn parse_with(text: &str, keep_angle: impl Fn(&str) -> bool + Copy) -> Vec<Inline> {
    let chars: Vec<char> = text.chars().collect();
    parse_chars(&chars, keep_angle)
}

/// The text of `inlines` with every mark dropped (a link keeps only its
/// text).
pub fn plain_text(inlines: &[Inline]) -> String {
    let mut out = String::new();
    for inline in inlines {
        match inline {
            Inline::Text(t) | Inline::Code(t) => out.push_str(t),
            Inline::Angle(t) => {
                out.push('<');
                out.push_str(t);
                out.push('>');
            }
            Inline::Bold(inner) | Inline::Italic(inner) | Inline::Strike(inner) => {
                out.push_str(&plain_text(inner));
            }
            Inline::Link { text, .. } => out.push_str(&plain_text(text)),
        }
    }
    out
}

fn parse_chars(chars: &[char], keep_angle: impl Fn(&str) -> bool + Copy) -> Vec<Inline> {
    let mut out = Vec::new();
    let mut text = String::new();
    let flush = |text: &mut String, out: &mut Vec<Inline>| {
        if !text.is_empty() {
            out.push(Inline::Text(std::mem::take(text)));
        }
    };
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let c = chars[i];
        let emit = |node: Inline, next: usize, text: &mut String, out: &mut Vec<Inline>| {
            flush(text, out);
            out.push(node);
            next
        };

        if c == '`'
            && let Some(end) = find(chars, '`', i + 1)
        {
            i = emit(
                Inline::Code(collect(&chars[i + 1..end])),
                end + 1,
                &mut text,
                &mut out,
            );
            continue;
        }

        // `**x**`, `__x__`, `~~x~~`.
        if matches!(c, '*' | '_' | '~')
            && chars.get(i + 1) == Some(&c)
            && let Some(end) = find_double(chars, c, i + 2)
            && end > i + 2
        {
            let inner = parse_chars(&chars[i + 2..end], keep_angle);
            let node = if c == '~' {
                Inline::Strike(inner)
            } else {
                Inline::Bold(inner)
            };
            i = emit(node, end + 2, &mut text, &mut out);
            continue;
        }

        if c == '['
            && let Some((close, url_end)) = link(chars, i)
        {
            let node = Inline::Link {
                text: parse_chars(&chars[i + 1..close], keep_angle),
                url: collect(&chars[close + 2..url_end]),
            };
            i = emit(node, url_end + 1, &mut text, &mut out);
            continue;
        }

        if c == '<'
            && let Some(end) = find(chars, '>', i + 1)
        {
            let inner = collect(&chars[i + 1..end]);
            if keep_angle(&inner) {
                i = emit(Inline::Angle(inner), end + 1, &mut text, &mut out);
                continue;
            }
        }

        // `_x_` only at word boundaries.
        if c == '_'
            && (i == 0 || !chars[i - 1].is_alphanumeric())
            && let Some(end) = find(chars, '_', i + 1)
            && end > i + 1
            && !chars[i + 1].is_whitespace()
            && (end + 1 >= len || !chars[end + 1].is_alphanumeric())
        {
            let inner = parse_chars(&chars[i + 1..end], keep_angle);
            i = emit(Inline::Italic(inner), end + 1, &mut text, &mut out);
            continue;
        }

        // `*x*`: the stars touch the words they hold.
        if c == '*'
            && chars
                .get(i + 1)
                .is_some_and(|next| *next != '*' && !next.is_whitespace())
            && let Some(end) = find_single_star(chars, i + 1)
            && !chars[end - 1].is_whitespace()
        {
            let inner = parse_chars(&chars[i + 1..end], keep_angle);
            i = emit(Inline::Italic(inner), end + 1, &mut text, &mut out);
            continue;
        }

        text.push(c);
        i += 1;
    }
    flush(&mut text, &mut out);
    out
}

fn collect(chars: &[char]) -> String {
    chars.iter().collect()
}

fn find(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == c)
}

fn find_double(chars: &[char], c: char, start: usize) -> Option<usize> {
    (start..chars.len().saturating_sub(1)).find(|&j| chars[j] == c && chars[j + 1] == c)
}

/// A lone closing `*` (not part of `**`).
fn find_single_star(chars: &[char], start: usize) -> Option<usize> {
    (start..chars.len()).find(|&j| {
        chars[j] == '*' && chars.get(j + 1) != Some(&'*') && (j == 0 || chars[j - 1] != '*')
    })
}

/// `[text](url)`: (index of `]`, index of `)`), the URL non-empty and
/// without whitespace.
fn link(chars: &[char], start: usize) -> Option<(usize, usize)> {
    let close = find(chars, ']', start + 1)?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let paren = find(chars, ')', close + 2)?;
    if paren == close + 2 || chars[close + 2..paren].iter().any(|c| c.is_whitespace()) {
        return None;
    }
    Some((close, paren))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Inline {
        Inline::Text(s.to_string())
    }

    #[test]
    fn marks_nest_and_snake_case_stays_text() {
        assert_eq!(
            parse("**bold _it_** and my_var_name `c*d`"),
            vec![
                Inline::Bold(vec![text("bold "), Inline::Italic(vec![text("it")])]),
                text(" and my_var_name "),
                Inline::Code("c*d".into()),
            ]
        );
    }

    #[test]
    fn a_star_needs_words_inside_it() {
        assert_eq!(parse("2 * 3 * 4"), vec![text("2 * 3 * 4")]);
        assert_eq!(parse("*a*"), vec![Inline::Italic(vec![text("a")])]);
    }

    #[test]
    fn links_need_a_url_without_spaces() {
        assert_eq!(
            parse("[docs](https://example.com) [x](not a url)"),
            vec![
                Inline::Link {
                    text: vec![text("docs")],
                    url: "https://example.com".into()
                },
                text(" [x](not a url)"),
            ]
        );
    }

    #[test]
    fn kept_angle_tokens_stay_whole() {
        assert_eq!(
            parse_with("hi <@U1> <b>", |inner| inner.starts_with('@')),
            vec![text("hi "), Inline::Angle("@U1".into()), text(" <b>")]
        );
    }
}
