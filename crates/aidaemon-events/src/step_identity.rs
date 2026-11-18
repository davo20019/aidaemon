//! Which step of a re-declared checklist is which step of the one before it.
//!
//! A checklist is re-declared as a full ordered list, so a step's index is
//! where it sits now, not what it is. The runs that took a step on, the
//! launches that named it and its proof belong to the step, wherever the
//! new list puts it (live 2026-11-14: the turn re-declared its checklist
//! right after launching two helpers; the launches' items pointed at other
//! rows, the parent's own install rows read "not attempted" and never
//! settled). A step is the same step by typed links only: the owner's units
//! it carries out (`owner_units`), else its own words unchanged. Never by
//! how alike two wordings are.

/// One declared step, as identity reads it.
#[derive(Debug, Clone, Copy)]
pub struct StepKey<'a> {
    pub description: &'a str,
    pub owner_units: &'a [usize],
}

impl StepKey<'_> {
    /// The same step: the same owner units (when it names any), or the
    /// same words.
    fn same_as(&self, other: &StepKey<'_>) -> bool {
        self.same_units(other) || self.description.trim() == other.description.trim()
    }

    fn same_units(&self, other: &StepKey<'_>) -> bool {
        !self.owner_units.is_empty() && unit_set(self.owner_units) == unit_set(other.owner_units)
    }

    fn shares_unit(&self, other: &StepKey<'_>) -> bool {
        self.owner_units
            .iter()
            .any(|unit| other.owner_units.contains(unit))
    }
}

/// One matching pass: (old index, old step, new index, new step).
type Pass<'p> = &'p dyn Fn(usize, &StepKey<'_>, usize, &StepKey<'_>) -> bool;

fn unit_set(units: &[usize]) -> std::collections::BTreeSet<usize> {
    units.iter().copied().collect()
}

/// For each step of `previous`, the index the same step has in `next`, or
/// `None` when `next` no longer lists it. Matched in order of strength: the
/// same position and identity, then the same owner units, then the same
/// words; each new step is matched once.
pub fn same_steps(previous: &[StepKey<'_>], next: &[StepKey<'_>]) -> Vec<Option<usize>> {
    let mut matched: Vec<Option<usize>> = vec![None; previous.len()];
    let mut taken = vec![false; next.len()];
    let passes: [Pass<'_>; 3] = [
        &|old, a, new, b| old == new && a.same_as(b),
        &|_, a, _, b| a.same_units(b),
        &|_, a, _, b| a.description.trim() == b.description.trim(),
    ];
    for pass in passes {
        for (old, key) in previous.iter().enumerate() {
            if matched[old].is_some() {
                continue;
            }
            if let Some(new) =
                (0..next.len()).find(|&new| !taken[new] && pass(old, key, new, &next[new]))
            {
                matched[old] = Some(new);
                taken[new] = true;
            }
        }
    }
    matched
}

/// Where the work a run took on in `previous` now lives in `next`: the same
/// step ([`same_steps`]), else every new step that carries one of the owner
/// units the old step did (a run doing a unit does each step of it the new
/// list names). Empty when `next` has no step for that work.
pub fn carried_work(previous: &[StepKey<'_>], next: &[StepKey<'_>]) -> Vec<Vec<usize>> {
    same_steps(previous, next)
        .into_iter()
        .zip(previous)
        .map(|(same, key)| match same {
            Some(index) => vec![index],
            None => next
                .iter()
                .enumerate()
                .filter(|(_, candidate)| key.shares_unit(candidate))
                .map(|(index, _)| index)
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key<'a>(description: &'a str, owner_units: &'a [usize]) -> StepKey<'a> {
        StepKey {
            description,
            owner_units,
        }
    }

    #[test]
    fn a_step_moved_by_a_re_declaration_is_found_by_its_units_or_words() {
        let before = [
            key("Install your Claude-company Drupal site", &[0]),
            key("Install your Claude-company WordPress site", &[1]),
            key("Report", &[]),
        ];
        let after = [
            key("Check DDEV", &[]),
            key("Install Claude company WordPress in folder B", &[1]),
            key("Install Claude company Drupal in folder A", &[0]),
            key("Report", &[]),
        ];
        assert_eq!(same_steps(&before, &after), vec![Some(2), Some(1), Some(3)]);
    }

    #[test]
    fn different_words_without_units_are_not_the_same_step() {
        let before = [key("Install Drupal", &[])];
        let after = [key("Install the Drupal site", &[])];
        assert_eq!(same_steps(&before, &after), vec![None]);
        assert_eq!(carried_work(&before, &after), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn work_whose_step_was_split_goes_to_every_step_of_its_unit() {
        let before = [key("Install and screenshot Drupal", &[0, 2])];
        let after = [
            key("Install Drupal", &[0]),
            key("Screenshot both sites", &[2, 3]),
            key("Install WordPress", &[1]),
        ];
        assert_eq!(carried_work(&before, &after), vec![vec![0, 1]]);
    }
}
