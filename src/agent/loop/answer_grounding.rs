//! Grounding check for enumeration-style replies.
//!
//! The agent loop validates tool *actions* extensively but historically never
//! validated answer *content*: a final reply could enumerate a "complete"
//! roster of names that appear in no tool output (assembled from search
//! snippets plus invention) and sail through every guard. This module detects
//! that signature — a bullet/numbered list of name-like entities where
//! multiple entries are absent from everything the model observed this turn —
//! so the completion phase can force a grounded rewrite.

/// Minimum name-like list entries before the check applies — short lists are
/// usually prose, not a data enumeration. Shared with the corroboration gate.
pub(in crate::agent) const MIN_LIST_ENTITIES: usize = 5;
/// Minimum ungrounded entries to flag. A single miss can be a spelling or
/// abbreviation mismatch; several misses is fabrication.
const MIN_UNGROUNDED: usize = 2;
/// Words shorter than this (initials, particles) are skipped when grounding.
const MIN_WORD_LEN: usize = 3;
/// Cap on reported entities — keeps the injected directive message bounded.
const MAX_REPORTED: usize = 10;

/// Return list entities in `reply` that are not grounded in any of the
/// `evidence` texts (tool outputs, the user's message). Empty when the reply
/// has too few name-like list entries for the check to be meaningful, or when
/// the misses are below the fabrication threshold.
pub(in crate::agent) fn find_ungrounded_list_entities(
    reply: &str,
    evidence: &[&str],
) -> Vec<String> {
    let entities = extract_list_name_entities(reply);
    if entities.len() < MIN_LIST_ENTITIES {
        return Vec::new();
    }
    let corpus = fold_for_match(&evidence.join("\n"));
    let mut ungrounded: Vec<String> = entities
        .into_iter()
        .filter(|entity| {
            // Grounded only when every significant word of the entity appears
            // somewhere in the evidence. Substring (not word-boundary) match
            // is deliberate: it tolerates punctuation/inflection around the
            // name and only errs toward NOT flagging.
            !entity
                .split_whitespace()
                .map(fold_for_match)
                .filter(|w| w.chars().count() >= MIN_WORD_LEN)
                .all(|w| corpus.contains(&w))
        })
        .collect();
    ungrounded.dedup();
    if ungrounded.len() < MIN_UNGROUNDED {
        return Vec::new();
    }
    ungrounded.truncate(MAX_REPORTED);
    ungrounded
}

/// Number of name-like list entities in `reply` — used by the corroboration
/// gate to decide whether the reply is an enumeration-style answer.
pub(in crate::agent) fn count_list_name_entities(reply: &str) -> usize {
    extract_list_name_entities(reply).len()
}

/// Extract name-like entities from bullet/numbered list items: a leading run
/// of 2-4 titlecase words, cut before any annotation (club, role, dash note).
fn extract_list_name_entities(reply: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in reply.lines() {
        let Some(item) = strip_list_marker(line) else {
            continue;
        };
        let item = item
            .trim_matches(|c| c == '*' || c == '_' || c == '`')
            .trim();
        // Cut before annotations: "(PSG)", ": role", "— note".
        let cut = item.find(['(', ':', '—', '–']).unwrap_or(item.len());
        let head = &item[..cut];
        let mut words: Vec<String> = Vec::new();
        for raw in head.split_whitespace() {
            let w = raw.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | '*' | '_' | '`'));
            if words.len() == 4 || !is_titlecase_name_word(w) {
                break;
            }
            words.push(w.to_string());
        }
        if words.len() >= 2 && !is_leading_stopword(&words[0]) {
            out.push(words.join(" "));
        }
    }
    out
}

/// Strip a bullet or numbered list marker, returning the item text.
fn strip_list_marker(line: &str) -> Option<&str> {
    let t = line.trim_start();
    for marker in ["• ", "- ", "* ", "– ", "· ", "◦ "] {
        if let Some(rest) = t.strip_prefix(marker) {
            return Some(rest);
        }
    }
    // "1. Name" / "12) Name"
    let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && digits <= 3 {
        let rest = &t[digits..];
        return rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "));
    }
    None
}

/// A titlecase word: uppercase first letter, the rest lowercase (apostrophes
/// and hyphens allowed). ALL-CAPS, mid-caps (McX), digits, and bare initials
/// are rejected — rejection only ends the name run, which fails open.
fn is_titlecase_name_word(w: &str) -> bool {
    let mut chars = w.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_uppercase() {
        return false;
    }
    let mut has_lower = false;
    for c in chars {
        if c.is_lowercase() {
            has_lower = true;
        } else if c != '\'' && c != '’' && c != '-' {
            return false;
        }
    }
    has_lower
}

/// Leading words that mark a prose/action bullet rather than a name
/// ("Fixed the parser bug", "The output is clean").
fn is_leading_stopword(w: &str) -> bool {
    matches!(
        w,
        "The"
            | "This"
            | "That"
            | "These"
            | "Those"
            | "There"
            | "Then"
            | "They"
            | "When"
            | "Where"
            | "While"
            | "After"
            | "Before"
            | "Added"
            | "Fixed"
            | "Updated"
            | "Removed"
            | "Changed"
            | "Created"
            | "Implemented"
            | "Improved"
            | "Renamed"
            | "Moved"
            | "Deleted"
            | "Note"
            | "Notes"
            | "Step"
            | "Option"
            | "Key"
            | "New"
            | "Use"
            | "Used"
            | "Using"
            | "Run"
            | "Running"
            | "Check"
            | "Checked"
            | "Make"
            | "Made"
            | "Ensure"
            | "Verify"
            | "Verified"
            | "Set"
            | "Get"
            | "Write"
            | "Read"
            | "Open"
            | "Closed"
            | "Install"
            | "Installed"
            | "Build"
            | "Built"
            | "Test"
            | "Tested"
            | "Deploy"
            | "Deployed"
            | "Review"
            | "Each"
            | "Every"
            | "Some"
            | "Most"
            | "Many"
            | "Your"
            | "Their"
            | "Our"
            | "His"
            | "Her"
            | "Its"
    )
}

/// Entities the `reply` makes a claim/denial about that appear nowhere in
/// `evidence` (tool outputs + user message). The classifier supplies the
/// candidate `entities`; this confirms the reply actually addresses them and
/// that they were not grounded this turn. Substring (folded) match, like the
/// list gate — errs toward NOT flagging.
#[allow(dead_code)]
pub(in crate::agent) fn find_unsearched_denials(
    reply: &str,
    entities: &[String],
    evidence: &[&str],
) -> Vec<String> {
    if entities.is_empty() {
        return Vec::new();
    }
    let reply_f = fold_for_match(reply);
    let corpus = fold_for_match(&evidence.join("\n"));
    entities
        .iter()
        .filter(|e| {
            let ef = fold_for_match(e);
            // The reply addresses the entity, but evidence does not contain it.
            ef.split_whitespace()
                .filter(|w| w.chars().count() >= 3)
                .any(|w| reply_f.contains(w))
                && !ef
                    .split_whitespace()
                    .filter(|w| w.chars().count() >= 3)
                    .all(|w| corpus.contains(w))
        })
        .cloned()
        .collect()
}

/// Lowercase and fold common Latin diacritics to ASCII so accent differences
/// between the model's spelling and the source text don't read as misses.
fn fold_for_match(s: &str) -> String {
    s.chars()
        .flat_map(char::to_lowercase)
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            'ý' | 'ÿ' => 'y',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROSTER_EVIDENCE: &str = "Ecuador squad preview: Moisés Caicedo (Chelsea) anchors \
         the midfield, with captain Enner Valencia up front and Willian Pacho \
         marshalling the defence. Kendry Páez and Pervis Estupiñán complete the spine.";

    #[test]
    fn fabricated_roster_entries_are_flagged() {
        let reply = "Here is the squad:\n\
             • Moisés Caicedo (Chelsea)\n\
             • Enner Valencia (Captain)\n\
             • Willian Pacho (PSG)\n\
             • Denis Segovia (LDU Quito)\n\
             • Alex Granda (Emelec)\n\
             • Yholen Pichenda (Independiente)\n";
        let ungrounded = find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE]);
        assert_eq!(
            ungrounded,
            vec!["Denis Segovia", "Alex Granda", "Yholen Pichenda"]
        );
    }

    #[test]
    fn fully_grounded_list_passes() {
        let reply = "Squad:\n\
             • Moisés Caicedo\n\
             • Enner Valencia\n\
             • Willian Pacho\n\
             • Kendry Páez\n\
             • Pervis Estupiñán\n";
        assert!(find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE]).is_empty());
    }

    #[test]
    fn short_lists_are_not_checked() {
        let reply = "• Denis Segovia\n• Alex Granda\n• Yholen Pichenda\n";
        assert!(find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE]).is_empty());
    }

    #[test]
    fn single_miss_is_tolerated() {
        // One ungrounded entry can be a spelling/abbreviation mismatch.
        let reply = "• Moisés Caicedo\n\
             • Enner Valencia\n\
             • Willian Pacho\n\
             • Kendry Páez\n\
             • Denis Segovia\n";
        assert!(find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE]).is_empty());
    }

    #[test]
    fn diacritic_differences_do_not_count_as_misses() {
        // Model wrote unaccented forms; evidence has the accented originals.
        let reply = "• Moises Caicedo\n\
             • Enner Valencia\n\
             • Willian Pacho\n\
             • Kendry Paez\n\
             • Pervis Estupinan\n";
        assert!(find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE]).is_empty());
    }

    #[test]
    fn user_text_counts_as_evidence() {
        let user_text = "Tell me about Denis Segovia, Alex Granda and Yholen Pichenda";
        let reply = "• Moisés Caicedo\n\
             • Enner Valencia\n\
             • Denis Segovia\n\
             • Alex Granda\n\
             • Yholen Pichenda\n";
        assert!(find_ungrounded_list_entities(reply, &[ROSTER_EVIDENCE, user_text]).is_empty());
    }

    #[test]
    fn numbered_items_with_club_annotations_extract_names() {
        let reply = "1. Willian Pacho (PSG)\n2. Piero Hincapié (Arsenal)\n";
        assert_eq!(
            extract_list_name_entities(reply),
            vec!["Willian Pacho", "Piero Hincapié"]
        );
    }

    #[test]
    fn prose_bullets_are_not_treated_as_names() {
        // Action/sentence bullets must not register as name entities, even
        // when they start with capitalized words.
        let reply = "- Fixed the parser bug\n\
             - Added a regression test\n\
             - THE OUTPUT is clean\n\
             - run cargo fmt\n";
        assert!(extract_list_name_entities(reply).is_empty());
    }

    #[test]
    fn bold_markers_are_stripped_from_items() {
        let reply = "• **Enner Valencia** (Captain)\n";
        assert_eq!(extract_list_name_entities(reply), vec!["Enner Valencia"]);
    }

    #[test]
    fn flags_denial_of_unsearched_entity() {
        let reply = "I don't have information about Conchi's spouse.";
        let entities = vec!["Conchi".to_string()];
        let evidence = vec!["partner_name: Aracely Zambrano"]; // Conchi absent
        let out = find_unsearched_denials(reply, &entities, &evidence);
        assert_eq!(out, vec!["Conchi".to_string()]);
    }

    #[test]
    fn does_not_flag_when_entity_is_in_evidence() {
        let reply = "I don't have Conchi's phone number.";
        let entities = vec!["Conchi".to_string()];
        let evidence = vec!["mother_name: Consuelo (Conchi) Montesdeoca"]; // present
        assert!(find_unsearched_denials(reply, &entities, &evidence).is_empty());
    }

    #[test]
    fn does_not_flag_when_no_entities() {
        assert!(find_unsearched_denials("anything", &[], &["x"]).is_empty());
    }
}
