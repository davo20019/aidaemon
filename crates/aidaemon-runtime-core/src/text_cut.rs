//! Long text cut to a byte budget, keeping its head and its end
//! (terminal output, coding-agent transcripts, helper results).

/// `text` cut to at most `max_bytes`: its first quarter and its end, joined
/// by a marker. Build and test tools print what decides the outcome last
/// (the test summary, the compiler's errors, stderr after stdout), so the
/// end is kept in preference to the middle. Cuts land on line breaks when
/// one is near, and always on char boundaries.
pub fn keep_head_and_tail(text: &str, max_bytes: usize) -> String {
    keep_head_and_tail_marked(text, max_bytes, |_| GENERIC_OMITTED_MARKER.to_string())
}

pub const GENERIC_OMITTED_MARKER: &str = "\n[... middle of the output omitted ...]\n";

/// What a head-and-tail cut left out, for its marker.
pub struct OmittedRange {
    /// First and last omitted line, 1-based.
    pub first_line: usize,
    pub last_line: usize,
    pub total_lines: usize,
    pub omitted_chars: usize,
}

/// [`keep_head_and_tail`] with a marker built from what was cut; the result
/// still fits `max_bytes`.
pub fn keep_head_and_tail_marked(
    text: &str,
    max_bytes: usize,
    marker: impl Fn(&OmittedRange) -> String,
) -> String {
    // The marker's longest form: every number it can name is at most the
    // text's length, so a marker built from those fits the real one.
    let marker_len = marker(&OmittedRange {
        first_line: text.len(),
        last_line: text.len(),
        total_lines: text.len(),
        omitted_chars: text.len(),
    })
    .len();
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let floor = |at: usize| text.floor_char_boundary(at);
    if max_bytes <= marker_len * 2 {
        // Too small for a marker: the end alone.
        let mut start = text.ceil_char_boundary(text.len() - max_bytes);
        if start == text.len() {
            // A limit smaller than the last character: show that character.
            start = text.char_indices().last().map_or(0, |(at, _)| at);
        }
        return text[start..].to_string();
    }
    let budget = max_bytes - marker_len;
    let head_end = floor(budget / 4);
    // Prefer ending the head at a line break in its last third.
    let head_end = text[..head_end]
        .rfind('\n')
        .filter(|at| *at >= head_end * 2 / 3)
        .map_or(head_end, |at| at + 1);
    let tail_len = budget - head_end;
    let tail_start = text.ceil_char_boundary(text.len() - tail_len);
    // Prefer starting the tail on a fresh line in its first third.
    let window_end = floor(tail_start + tail_len / 3);
    let tail_start = text[tail_start..window_end]
        .find('\n')
        .map_or(tail_start, |at| tail_start + at + 1);
    let omitted = OmittedRange {
        first_line: text[..head_end].matches('\n').count() + 1,
        last_line: text[..tail_start].matches('\n').count(),
        total_lines: text.matches('\n').count() + usize::from(!text.ends_with('\n')),
        omitted_chars: text[head_end..tail_start].chars().count(),
    };
    format!(
        "{}{}{}",
        &text[..head_end],
        marker(&omitted),
        &text[tail_start..]
    )
}
