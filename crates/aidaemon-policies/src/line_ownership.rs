//! Who last changed each line of a file that several agents write.
//!
//! Pure bookkeeping, with no model and no I/O: a host records every change
//! it sees ([`Ownership::record`]) and asks, before a write, whether the
//! write would change lines another writer holds ([`Ownership::contested`]).
//! Owners and rulings are the host's own small integers.

use std::ops::Range;

/// A writer, as the host numbers them.
pub type OwnerId = u32;

/// Nobody the host tracks: the file as it was found, or a change made from
/// outside.
pub const NOBODY: OwnerId = 0;

/// Who holds a line, and the host's ruling about it, if any (`0`: none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Claim {
    pub owner: OwnerId,
    pub ruling: u32,
}

impl Claim {
    pub fn by(owner: OwnerId) -> Self {
        Self { owner, ruling: 0 }
    }

    fn is_held(self) -> bool {
        self.owner != NOBODY
    }

    fn is_foreign_to(self, writer: OwnerId) -> bool {
        self.is_held() && self.owner != writer
    }
}

/// One run of lines that differs between two versions of a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    /// Lines of the earlier version it replaces (empty: an insertion).
    pub before: Range<usize>,
    /// Lines of the later version it puts there (empty: a removal).
    pub after: Range<usize>,
}

/// The lines of `bytes`, each with its line ending.
pub fn lines(bytes: &[u8]) -> Vec<&[u8]> {
    bytes.split_inclusive(|byte| *byte == b'\n').collect()
}

/// Edits past which two versions are compared as one changed block.
const MAX_EDIT_DISTANCE: usize = 512;

/// The runs of lines that differ between `before` and `after`, in order.
/// Versions further apart than `MAX_EDIT_DISTANCE` line edits come back as
/// one hunk covering everything between their common start and end.
pub fn diff_lines(before: &[u8], after: &[u8]) -> Vec<Hunk> {
    let (a, b) = (lines(before), lines(after));
    let prefix = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suffix = a[prefix..]
        .iter()
        .rev()
        .zip(b[prefix..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (a_mid, b_mid) = (&a[prefix..a.len() - suffix], &b[prefix..b.len() - suffix]);
    if a_mid.is_empty() && b_mid.is_empty() {
        return Vec::new();
    }
    let whole = || {
        vec![Hunk {
            before: prefix..prefix + a_mid.len(),
            after: prefix..prefix + b_mid.len(),
        }]
    };
    if a_mid.is_empty() || b_mid.is_empty() {
        return whole();
    }
    let Some(matches) = matched_lines(a_mid, b_mid) else {
        return whole();
    };
    let mut hunks = Vec::new();
    let (mut x, mut y) = (0usize, 0usize);
    let end = (a_mid.len(), b_mid.len());
    for (mx, my) in matches.into_iter().chain(std::iter::once(end)) {
        if mx > x || my > y {
            hunks.push(Hunk {
                before: prefix + x..prefix + mx,
                after: prefix + y..prefix + my,
            });
        }
        (x, y) = (mx + 1, my + 1);
    }
    hunks
}

/// The line pairs a shortest edit script keeps, in order (Myers). `None`
/// when the versions are more than `MAX_EDIT_DISTANCE` edits apart.
fn matched_lines(a: &[&[u8]], b: &[&[u8]]) -> Option<Vec<(usize, usize)>> {
    let (n, m) = (a.len() as isize, b.len() as isize);
    let limit = MAX_EDIT_DISTANCE.min(a.len() + b.len()) as isize;
    let offset = limit + 1;
    let mut v = vec![0isize; (2 * limit + 3) as usize];
    let mut trace: Vec<Vec<isize>> = Vec::new();
    let at = |k: isize| (k + offset) as usize;
    let mut reached = None;
    'search: for d in 0..=limit {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let mut x = if k == -d || (k != d && v[at(k - 1)] < v[at(k + 1)]) {
                v[at(k + 1)]
            } else {
                v[at(k - 1)] + 1
            };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[at(k)] = x;
            if x >= n && y >= m {
                reached = Some(d);
                break 'search;
            }
            k += 2;
        }
    }
    let reached = reached?;
    let mut matches = Vec::new();
    let (mut x, mut y) = (n, m);
    for d in (0..=reached).rev() {
        let v = &trace[d as usize];
        let k = x - y;
        let previous_k = if k == -d || (k != d && v[at(k - 1)] < v[at(k + 1)]) {
            k + 1
        } else {
            k - 1
        };
        let previous_x = if d == 0 { 0 } else { v[at(previous_k)] };
        let previous_y = if d == 0 { 0 } else { previous_x - previous_k };
        while x > previous_x && y > previous_y {
            x -= 1;
            y -= 1;
            matches.push((x as usize, y as usize));
        }
        (x, y) = (previous_x, previous_y);
    }
    matches.reverse();
    Some(matches)
}

/// What a write would change that someone else holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contested {
    /// The other writers' claims it would change, each once.
    pub claims: Vec<Claim>,
    /// The changes that touch them (`before`: the current file, `after`: the
    /// proposed one). Empty when [`whole_file`](Self::whole_file).
    pub hunks: Vec<Hunk>,
    /// Lines could not be told apart, so the whole file counts.
    pub whole_file: bool,
}

/// Who holds what in one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ownership {
    /// By line of the file's current content.
    Lines {
        lines: Vec<Claim>,
        /// A removal recorded before line `i`; one longer than `lines`.
        gaps: Vec<Claim>,
    },
    /// The lines could not be followed (a version was not kept): each of
    /// these holds the whole file.
    Whole(Vec<Claim>),
}

impl Ownership {
    /// A file nobody has changed. `None`: its content is not available, so
    /// its lines cannot be followed.
    pub fn unclaimed(content: Option<&[u8]>) -> Self {
        match content {
            Some(content) => {
                let count = lines(content).len();
                Self::Lines {
                    lines: vec![Claim::default(); count],
                    gaps: vec![Claim::default(); count + 1],
                }
            }
            None => Self::Whole(Vec::new()),
        }
    }

    /// Every claim someone holds, each once.
    pub fn claims(&self) -> Vec<Claim> {
        let mut found: Vec<Claim> = Vec::new();
        let mut add = |claim: &Claim| {
            if claim.is_held() && !found.contains(claim) {
                found.push(*claim);
            }
        };
        match self {
            Self::Lines { lines, gaps } => lines.iter().chain(gaps).for_each(&mut add),
            Self::Whole(claims) => claims.iter().for_each(&mut add),
        }
        found
    }

    fn degrade(&mut self, claim: Claim) {
        let mut claims = self.claims();
        if claim.is_held() {
            claims.retain(|held| held.owner != claim.owner);
            claims.push(claim);
        }
        *self = Self::Whole(claims);
    }

    /// The file went from `before` to `after`, and `claim` made the change.
    /// A version that is not available (`None`) ends line tracking for this
    /// file: from then on every holder holds all of it.
    pub fn record(&mut self, before: Option<&[u8]>, after: Option<&[u8]>, claim: Claim) {
        let (Self::Lines { lines: held, gaps }, Some(before), Some(after)) =
            (&mut *self, before, after)
        else {
            return self.degrade(claim);
        };
        if held.len() != lines(before).len() {
            return self.degrade(claim);
        }
        let count = lines(after).len();
        let hunks = diff_lines(before, after);
        let mut new_lines = Vec::with_capacity(count);
        let mut new_gaps = vec![Claim::default(); count + 1];
        let (mut x, mut y) = (0usize, 0usize);
        let end = Hunk {
            before: held.len()..held.len(),
            after: count..count,
        };
        for hunk in hunks.iter().chain(std::iter::once(&end)) {
            // The unchanged run before this hunk keeps its claims.
            // The gap right after a hunk is that hunk's to set.
            let kept = hunk.before.start - x;
            let skip = usize::from(x > 0 || y > 0);
            new_lines.extend_from_slice(&held[x..x + kept]);
            if skip <= kept {
                new_gaps[y + skip..=y + kept].copy_from_slice(&gaps[x + skip..=x + kept]);
            }
            if hunk.after.is_empty() && !hunk.before.is_empty() {
                new_gaps[hunk.after.start] = claim;
            } else if !hunk.after.is_empty() {
                new_gaps[hunk.after.start..=hunk.after.end].fill(Claim::default());
            }
            new_lines.extend(std::iter::repeat_n(claim, hunk.after.len()));
            (x, y) = (hunk.before.end, hunk.after.end);
        }
        *held = new_lines;
        *gaps = new_gaps;
    }

    /// The same change as [`record`](Self::record), for a file already
    /// recorded as holding `after` under someone else's name: only the
    /// changed lines move to `claim`.
    pub fn reclaim(&mut self, before: Option<&[u8]>, after: Option<&[u8]>, claim: Claim) {
        let (Self::Lines { lines: held, gaps }, Some(before), Some(after)) =
            (&mut *self, before, after)
        else {
            return self.degrade(claim);
        };
        if held.len() != lines(after).len() {
            return self.degrade(claim);
        }
        for hunk in diff_lines(before, after) {
            if hunk.after.is_empty() {
                gaps[hunk.after.start] = claim;
            }
            held[hunk.after].fill(claim);
        }
    }

    /// What changing the file from `current` to `proposed` would alter that
    /// a writer other than `writer` holds. `None`: nothing. A `proposed` of
    /// `None` is a write whose result is not known, which counts as changing
    /// every line.
    pub fn contested(
        &self,
        current: Option<&[u8]>,
        proposed: Option<&[u8]>,
        writer: OwnerId,
    ) -> Option<Contested> {
        if let (Self::Lines { lines: held, gaps }, Some(current), Some(proposed)) =
            (self, current, proposed)
            && held.len() == lines(current).len() {
                let mut claims: Vec<Claim> = Vec::new();
                let mut hunks = Vec::new();
                for hunk in diff_lines(current, proposed) {
                    let touched: Vec<Claim> = held[hunk.before.clone()]
                        .iter()
                        .chain(&gaps[hunk.before.start..=hunk.before.end])
                        .filter(|claim| claim.is_foreign_to(writer))
                        .copied()
                        .collect();
                    let foreign = !touched.is_empty();
                    for claim in touched {
                        if !claims.contains(&claim) {
                            claims.push(claim);
                        }
                    }
                    if foreign {
                        hunks.push(hunk);
                    }
                }
                return (!hunks.is_empty()).then_some(Contested {
                    claims,
                    hunks,
                    whole_file: false,
                });
            }
        if let (Some(current), Some(proposed)) = (current, proposed)
            && current == proposed {
                return None;
            }
        let claims: Vec<Claim> = self
            .claims()
            .into_iter()
            .filter(|claim| claim.is_foreign_to(writer))
            .collect();
        (!claims.is_empty()).then_some(Contested {
            claims,
            hunks: Vec::new(),
            whole_file: true,
        })
    }

    /// Put a ruling on the claims `contested` found, where `writer` asked to
    /// change them: `ruling_for` names the ruling each claim now carries.
    pub fn stamp(
        &mut self,
        contested: &Contested,
        writer: OwnerId,
        ruling_for: &mut dyn FnMut(Claim) -> u32,
    ) {
        let mut rule = |claim: &mut Claim| {
            if claim.is_foreign_to(writer) {
                claim.ruling = ruling_for(*claim);
            }
        };
        match self {
            Self::Lines { lines, gaps } if !contested.whole_file => {
                for hunk in &contested.hunks {
                    if hunk.before.end > lines.len() {
                        continue;
                    }
                    lines[hunk.before.clone()].iter_mut().for_each(&mut rule);
                    gaps[hunk.before.start..=hunk.before.end]
                        .iter_mut()
                        .for_each(&mut rule);
                }
            }
            Self::Lines { lines, gaps } => lines.iter_mut().chain(gaps).for_each(&mut rule),
            Self::Whole(claims) => claims.iter_mut().for_each(&mut rule),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: OwnerId = 2;
    const B: OwnerId = 3;

    fn hunk(before: Range<usize>, after: Range<usize>) -> Hunk {
        Hunk { before, after }
    }

    fn owners(ownership: &Ownership) -> Vec<OwnerId> {
        match ownership {
            Ownership::Lines { lines, .. } => lines.iter().map(|claim| claim.owner).collect(),
            Ownership::Whole(_) => panic!("lines are not followed"),
        }
    }

    #[test]
    fn a_diff_names_each_changed_run_separately() {
        let before = b"one\ntwo\nthree\nfour\nfive\n";
        let after = b"ONE\ntwo\nthree\nfour\nFIVE\nsix\n";
        assert_eq!(
            diff_lines(before, after),
            vec![hunk(0..1, 0..1), hunk(4..5, 4..6)]
        );
        assert_eq!(diff_lines(before, before), Vec::new());
        assert_eq!(diff_lines(b"", b"new\n"), vec![hunk(0..0, 0..1)]);
        assert_eq!(
            diff_lines(b"a\nb\nc\n", b"a\nc\n"),
            vec![hunk(1..2, 1..1)],
            "a removal"
        );
        assert_eq!(
            diff_lines(b"a\nc\n", b"a\nb\nc\n"),
            vec![hunk(1..1, 1..2)],
            "an insertion"
        );
    }

    #[test]
    fn a_missing_final_newline_is_a_change_to_the_last_line_only() {
        assert_eq!(diff_lines(b"a\nb", b"a\nb\n"), vec![hunk(1..2, 1..2)]);
    }

    #[test]
    fn versions_too_far_apart_compare_as_one_block() {
        let before: Vec<u8> = (0..2_000)
            .flat_map(|i| format!("old {i}\n").into_bytes())
            .collect();
        let after: Vec<u8> = (0..2_000)
            .flat_map(|i| format!("new {i}\n").into_bytes())
            .collect();
        let mut framed_before = b"head\n".to_vec();
        framed_before.extend(&before);
        framed_before.extend(b"tail\n");
        let mut framed_after = b"head\n".to_vec();
        framed_after.extend(&after);
        framed_after.extend(b"tail\n");
        assert_eq!(
            diff_lines(&framed_before, &framed_after),
            vec![hunk(1..2_001, 1..2_001)]
        );
    }

    #[test]
    fn changed_lines_go_to_the_writer_and_the_rest_keep_their_holder() {
        let v0 = b"one\ntwo\nthree\n";
        let v1 = b"one\nTWO\nthree\n";
        let v2 = b"zero\none\nTWO\nthree\nfour\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(B));
        assert_eq!(owners(&file), vec![NOBODY, B, NOBODY]);
        file.record(Some(v1), Some(v2), Claim::by(A));
        assert_eq!(owners(&file), vec![A, NOBODY, B, NOBODY, A]);
    }

    #[test]
    fn a_write_is_contested_only_where_it_changes_anothers_lines() {
        let v0 = b"one\ntwo\nthree\n";
        let v1 = b"one\nTWO\nthree\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(B));

        // Its own line, and lines nobody holds.
        assert_eq!(
            file.contested(Some(v1), Some(b"one\nTWO!\nthree\n"), B),
            None
        );
        assert_eq!(
            file.contested(Some(v1), Some(b"ONE\nTWO\nthree\nfour\n"), A),
            None
        );
        // The sibling's line, by an edit or by replacing the file.
        let contest = file
            .contested(Some(v1), Some(b"one\nmine\nthree\n"), A)
            .unwrap();
        assert_eq!(contest.claims, vec![Claim::by(B)]);
        assert_eq!(contest.hunks, vec![hunk(1..2, 1..2)]);
        assert!(!contest.whole_file);
        assert!(file.contested(Some(v1), Some(b"all new\n"), A).is_some());
        // A replacement that keeps the sibling's line as it is.
        assert_eq!(
            file.contested(Some(v1), Some(b"ONE\nTWO\nTHREE\n"), A),
            None
        );
    }

    #[test]
    fn a_removed_line_cannot_be_put_back_unnoticed() {
        let v0 = b"one\ntwo\nthree\n";
        let v1 = b"one\nthree\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(B));
        assert!(file.contested(Some(v1), Some(v0), A).is_some());
        assert_eq!(file.contested(Some(v1), Some(v0), B), None);
        assert_eq!(
            file.contested(Some(v1), Some(b"one\nthree\nfour\n"), A),
            None
        );
    }

    #[test]
    fn an_outside_change_clears_the_lines_it_rewrites() {
        let v0 = b"one\ntwo\n";
        let v1 = b"one\nTWO\n";
        let v2 = b"one\noutside\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(B));
        file.record(Some(v1), Some(v2), Claim::by(NOBODY));
        assert_eq!(owners(&file), vec![NOBODY, NOBODY]);
    }

    #[test]
    fn without_a_version_the_whole_file_is_held() {
        let v0 = b"one\ntwo\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), None, Claim::by(B));
        assert_eq!(file, Ownership::Whole(vec![Claim::by(B)]));
        let contest = file.contested(Some(b"x\n"), Some(b"y\n"), A).unwrap();
        assert!(contest.whole_file);
        assert_eq!(file.contested(Some(b"x\n"), Some(b"y\n"), B), None);
        assert_eq!(file.contested(Some(b"x\n"), Some(b"x\n"), A), None);
        // A write whose result is unknown counts as changing everything.
        assert!(file.contested(Some(b"x\n"), None, A).is_some());
    }

    #[test]
    fn a_write_with_no_known_result_is_contested_by_any_other_holder() {
        let v0 = b"one\ntwo\n";
        let v1 = b"one\nTWO\n";
        let mut file = Ownership::unclaimed(Some(v0));
        assert_eq!(file.contested(Some(v0), None, A), None);
        file.record(Some(v0), Some(v1), Claim::by(B));
        assert!(file.contested(Some(v1), None, A).unwrap().whole_file);
        assert_eq!(file.contested(Some(v1), None, B), None);
    }

    #[test]
    fn a_ruling_is_stamped_on_the_contested_claims_only() {
        let v0 = b"one\ntwo\nthree\n";
        let v1 = b"ONE\ntwo\nTHREE\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(B));
        let contest = file
            .contested(Some(v1), Some(b"mine\ntwo\nTHREE\n"), A)
            .unwrap();
        file.stamp(&contest, A, &mut |_| 7);
        let Ownership::Lines { lines, .. } = &file else {
            panic!()
        };
        assert_eq!(
            lines[0],
            Claim {
                owner: B,
                ruling: 7
            }
        );
        assert_eq!(lines[2], Claim::by(B));
    }

    #[test]
    fn a_change_recorded_under_another_name_can_be_reclaimed() {
        let v0 = b"one\ntwo\n";
        let v1 = b"one\nTWO\n";
        let mut file = Ownership::unclaimed(Some(v0));
        file.record(Some(v0), Some(v1), Claim::by(NOBODY));
        file.reclaim(Some(v0), Some(v1), Claim::by(B));
        assert_eq!(owners(&file), vec![NOBODY, B]);
    }
}
