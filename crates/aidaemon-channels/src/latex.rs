//! LaTeX math cleanup for chat platforms that cannot render it.

use std::sync::LazyLock;

use regex::Regex;

/// Rewrite LaTeX math as readable plain text (`\frac{a}{b}` → `(a)/(b)`,
/// `x^2` → `x²`, `\times` → `×`) and drop math delimiters. Fenced code
/// blocks are left untouched.
///
/// Inline `$...$` follows Pandoc's rule: the opening `$` must be followed by
/// a non-space character, and the closing `$` preceded by one and not
/// followed by a digit. So prices such as `$50 and $60` stay as written.
///
/// ```
/// use aidaemon_channels::latex::strip_latex;
///
/// assert_eq!(strip_latex("$247 \\times 38$"), "247 × 38");
/// assert_eq!(strip_latex("Price is $50 and $60"), "Price is $50 and $60");
/// ```
pub fn strip_latex(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find("```") {
        result.push_str(&strip_latex_outside_code(&remaining[..start]));
        let after_opening = &remaining[start + 3..];
        if let Some(end_offset) = after_opening.find("```") {
            let end = start + 3 + end_offset + 3;
            result.push_str(&remaining[start..end]);
            remaining = &remaining[end..];
        } else {
            // An unclosed fence: the rest is code.
            result.push_str(&remaining[start..]);
            return result;
        }
    }
    result.push_str(&strip_latex_outside_code(remaining));
    result
}

/// [`strip_latex`] for everything except code: fenced blocks and inline
/// `` `code` `` spans (of any backtick run length) keep every character, so a
/// shell variable or a formula quoted as code stays exactly as written.
///
/// This is the formatter on the outbound path of [`crate::ChannelRunner`];
/// a channel opts out with [`crate::Channel::strips_latex`].
///
/// ```
/// use aidaemon_channels::latex::strip_latex_keeping_code;
///
/// assert_eq!(
///     strip_latex_keeping_code("$x^2$ but not `$x^2$`"),
///     "x² but not `$x^2$`"
/// );
/// ```
pub fn strip_latex_keeping_code(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut plain_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let run = bytes[i..].iter().take_while(|b| **b == b'`').count();
        // A backslash-escaped backtick opens nothing.
        if i > 0 && bytes[i - 1] == b'\\' {
            i += 1;
            continue;
        }
        match code_end(text, i + run, run) {
            Some(end) => {
                out.push_str(&strip_latex_outside_code(&text[plain_start..i]));
                out.push_str(&text[i..end]);
                i = end;
                plain_start = end;
            }
            None if run >= 3 => {
                // An unclosed fence: the rest is code.
                out.push_str(&strip_latex_outside_code(&text[plain_start..i]));
                out.push_str(&text[i..]);
                return out;
            }
            None => i += run,
        }
    }
    out.push_str(&strip_latex_outside_code(&text[plain_start..]));
    out
}

/// Where the code opened by `run` backticks before `from` ends: after a run
/// of at least as many for a fence, of exactly as many for a span, which
/// cannot cross a blank line.
fn code_end(text: &str, from: usize, run: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let fence = run >= 3;
    let mut i = from;
    while i < bytes.len() {
        if !fence && bytes[i] == b'\n' {
            let next_line = &text[i + 1..];
            let line_end = next_line.find('\n').unwrap_or(next_line.len());
            if next_line[..line_end].trim().is_empty() {
                return None;
            }
        }
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let closing = bytes[i..].iter().take_while(|b| **b == b'`').count();
        i += closing;
        if closing == run || (fence && closing > run) {
            return Some(i);
        }
    }
    None
}

static SQRT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\sqrt\{([^}]*)\}").unwrap());
static FRAC: LazyLock<Regex> = LazyLock::new(|| {
    // One level of nested braces in each argument.
    Regex::new(r"\\frac\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}").unwrap()
});
static SUPERSCRIPT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\^\{([^}]*)\}").unwrap());
static SUBSCRIPT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_\{([^}]*)\}").unwrap());
static TEXT_COMMAND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\(?:text|mathrm|mathbf|mathit|textbf)\{([^}]*)\}").unwrap());
static DISPLAY_DOLLARS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\$\$(.+?)\$\$").unwrap());
static DISPLAY_BRACKETS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\\\[(.+?)\\\]").unwrap());

const SYMBOLS: &[(&str, &str)] = &[
    ("\\times", "×"),
    ("\\div", "÷"),
    ("\\pm", "±"),
    ("\\leq", "≤"),
    ("\\geq", "≥"),
    ("\\neq", "≠"),
    ("\\approx", "≈"),
    ("\\infty", "∞"),
    ("\\cdot", "·"),
    ("\\alpha", "α"),
    ("\\beta", "β"),
    ("\\gamma", "γ"),
    ("\\delta", "δ"),
    ("\\theta", "θ"),
    ("\\lambda", "λ"),
    ("\\sigma", "σ"),
    ("\\pi", "π"),
    ("\\left(", "("),
    ("\\right)", ")"),
    ("\\left[", "["),
    ("\\right]", "]"),
    ("\\left|", "|"),
    ("\\right|", "|"),
    ("\\left\\{", "{"),
    ("\\right\\}", "}"),
];

const FUNCTIONS: &[&str] = &["sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min"];

fn strip_latex_outside_code(text: &str) -> String {
    // Delimiters first, judged on the text as written.
    let mut s = DISPLAY_DOLLARS.replace_all(text, "$1").into_owned();
    s = DISPLAY_BRACKETS.replace_all(&s, "$1").into_owned();
    s = strip_inline_dollars(&s);

    for (command, symbol) in SYMBOLS {
        s = s.replace(command, symbol);
    }
    for function in FUNCTIONS {
        s = s.replace(&format!("\\{function}"), function);
    }
    // `\sqrt` before `\frac`, so a root inside a fraction is resolved first;
    // a few passes resolve nesting.
    for (pattern, replacement) in [(&*SQRT, "√($1)"), (&*FRAC, "($1)/($2)")] {
        for _ in 0..3 {
            let next = pattern.replace_all(&s, replacement).into_owned();
            if next == s {
                break;
            }
            s = next;
        }
    }
    s = s.replace("^2", "²").replace("^3", "³").replace("^n", "ⁿ");
    s = SUPERSCRIPT.replace_all(&s, "^($1)").into_owned();
    s = SUBSCRIPT.replace_all(&s, "_($1)").into_owned();
    TEXT_COMMAND.replace_all(&s, "$1").into_owned()
}

/// Remove `$` delimiters around inline math, by Pandoc's rule (see
/// [`strip_latex`]). Math never spans lines.
fn strip_inline_dollars(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$'
            && let Some(close) = inline_math_close(&chars, i)
        {
            out.extend(&chars[i + 1..close]);
            i = close + 1;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// The index of the `$` closing inline math opened at `open`, if any.
fn inline_math_close(chars: &[char], open: usize) -> Option<usize> {
    let first = *chars.get(open + 1)?;
    if first.is_whitespace() || first == '$' {
        return None;
    }
    let mut j = open + 2;
    while j < chars.len() {
        match chars[j] {
            '\n' => return None,
            '$' if !chars[j - 1].is_whitespace()
                && !chars.get(j + 1).is_some_and(char::is_ascii_digit) =>
            {
                return Some(j);
            }
            _ => {}
        }
        j += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{strip_latex, strip_latex_keeping_code};

    #[test]
    fn code_keeps_every_character() {
        for code in [
            "`$x^2$`",
            "``a ` $x^2$ b``",
            "```\n$\\frac{a}{b}$ \\times\n```",
            "```sh\necho $HOME^2\n```",
        ] {
            let text = format!("$y^2$ before {code} after $y^2$");
            assert_eq!(
                strip_latex_keeping_code(&text),
                format!("y² before {code} after y²")
            );
        }
    }

    #[test]
    fn unclosed_code_marks_are_handled_like_markdown() {
        // A fence never closed runs to the end.
        assert_eq!(
            strip_latex_keeping_code("$x^2$\n```\n$y^2$"),
            "x²\n```\n$y^2$"
        );
        // A lone backtick is just a backtick.
        assert_eq!(strip_latex_keeping_code("a ` $x^2$"), "a ` x²");
        // A span cannot cross a blank line.
        assert_eq!(
            strip_latex_keeping_code("a ` b\n\n$x^2$ ` c"),
            "a ` b\n\nx² ` c"
        );
        // An escaped backtick opens nothing.
        assert_eq!(strip_latex_keeping_code("\\` $x^2$ `y`"), "\\` x² `y`");
    }

    #[test]
    fn plain_text_matches_the_fence_only_formatter() {
        for text in [
            "$247 \\times 38$",
            "Price is $50 and $60",
            "$$x = \\frac{1}{2}$$",
            "```\n$x$\n``` and $y$",
        ] {
            assert_eq!(strip_latex_keeping_code(text), strip_latex(text));
        }
    }

    #[test]
    fn inline_math_becomes_plain_text() {
        assert_eq!(strip_latex("$247 \\times 38$"), "247 × 38");
        let frac = strip_latex("$\\frac{a}{b}$");
        assert!(frac.contains('a') && frac.contains('b') && frac.contains('/'));
        assert!(!frac.contains("\\frac"));
        let sqrt = strip_latex("$\\sqrt{x}$");
        assert!(sqrt.contains('√') && sqrt.contains('x') && !sqrt.contains("\\sqrt"));
    }

    #[test]
    fn display_math_loses_its_delimiters() {
        assert_eq!(strip_latex("$$x + y$$"), "x + y");
        assert_eq!(strip_latex("\\[x + y\\]"), "x + y");
    }

    #[test]
    fn prices_are_not_math() {
        assert_eq!(strip_latex("costs $50"), "costs $50");
        assert_eq!(strip_latex("Price is $50 and $60"), "Price is $50 and $60");
        assert_eq!(strip_latex("$5 + $10"), "$5 + $10");
        assert_eq!(strip_latex("from $5 to $10."), "from $5 to $10.");
    }

    #[test]
    fn inline_math_closes_at_the_first_valid_dollar_on_its_line() {
        assert_eq!(strip_latex("$x$ and $y$"), "x and y");
        // A `$` followed by a digit cannot close math.
        assert_eq!(strip_latex("$x$5"), "$x$5");
        assert_eq!(strip_latex("a $x\ny$ b"), "a $x\ny$ b");
    }

    #[test]
    fn the_quadratic_formula_reads_cleanly() {
        let result = strip_latex(r"$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$");
        for leftover in ["\\frac", "\\sqrt", "\\pm", "$$"] {
            assert!(!result.contains(leftover), "{leftover} left in: {result}");
        }
        for wanted in ['√', '±', '/'] {
            assert!(result.contains(wanted), "missing {wanted}: {result}");
        }
    }

    #[test]
    fn superscripts_and_functions() {
        assert_eq!(strip_latex("$x^2$"), "x²");
        assert_eq!(strip_latex("$x^3$"), "x³");
        let braced = strip_latex("$x^{n+1}$");
        assert!(braced.contains("n+1") && !braced.contains('{'));
        let trig = strip_latex(r"$\sin(x) + \cos(y)$");
        assert!(trig.contains("sin(x)") && trig.contains("cos(y)") && !trig.contains('\\'));
        let delimiters = strip_latex(r"$\left(\frac{a}{b}\right)$");
        assert!(!delimiters.contains("\\left") && !delimiters.contains("\\right"));
        assert!(delimiters.contains('(') && delimiters.contains(')'));
    }

    #[test]
    fn code_blocks_are_untouched() {
        let input = "Here is math: $x^2$\n```\n$\\frac{a}{b}$\n```\nMore math: $y^3$";
        let result = strip_latex(input);
        assert!(result.contains("x²") && result.contains("y³"), "{result}");
        assert!(result.contains("\\frac{a}{b}"), "{result}");
    }
}
