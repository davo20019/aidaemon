//! Sites on this computer that the owner's own work brought up.
//!
//! Opening a site on this computer in the browser asks the owner first: a
//! page there can be anything that listens on loopback. A site the current
//! work set up itself (a dev server it started, a `ddev` project it created)
//! or one the owner's request names is part of what was asked, so it needs
//! no card. The runtime notes such sites here, keyed by the session whose
//! approvals cover the work (the chat, shared by its sub-agents), from
//! typed sources only: the owner's own request, the output of a tool call
//! the run made whose output is typed as trusted (local, not content from
//! outside), and the output of a call that set something up on this
//! computer within the run's authority ([`set_up_locally`]): what a run
//! created within its scope is in its scope, whatever program it used. A
//! site served from inside the run's own write scope is in its scope too,
//! whoever set it up and whenever ([`served_within`]): a command that ran
//! in a folder the run may change reports the sites that folder serves (a
//! project's own `status` or `describe` output), so the site's project
//! folder, not whether this run's set-up command printed it, decides. A
//! call that only read elsewhere (a page, a file, a status query outside
//! the run's scope) adds nothing. Each
//! site is kept by the key the browser approves sites under
//! ([`crate::web::local_site_key`]), so only names that resolve to this
//! computer alone are ever noted; any other host stays gated.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

/// How long a noted site stays covered.
const KEEP_FOR: Duration = Duration::from_secs(6 * 60 * 60);
/// How many sites one session keeps (the oldest go first).
const MAX_PER_SESSION: usize = 64;
/// How many addresses one text is searched for (each costs a lookup).
const MAX_URLS_PER_TEXT: usize = 16;

static SITES: LazyLock<Mutex<HashMap<String, HashMap<String, Instant>>>> =
    LazyLock::new(Mutex::default);

/// The `http(s)://` addresses written in `text`, in order, without
/// duplicates.
fn urls_in(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut rest = text;
    while let Some(at) = ["http://", "https://"]
        .iter()
        .filter_map(|scheme| rest.find(scheme))
        .min()
    {
        let tail = &rest[at..];
        let end = tail
            .find(|c: char| {
                c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`' | '|' | '\\')
            })
            .unwrap_or(tail.len());
        let url = tail[..end].trim_end_matches(['.', ',', ';', ':', ')', ']', '}', '!', '?']);
        if !url.is_empty() && !urls.iter().any(|known| known == url) {
            urls.push(url.to_string());
            if urls.len() == MAX_URLS_PER_TEXT {
                break;
            }
        }
        rest = &tail[end.max(1)..];
    }
    urls
}

/// The site keys of the addresses in `text` that point at this computer
/// alone. Resolves names, so it blocks.
pub fn local_sites_in(text: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for url in urls_in(text) {
        if let Some(key) = crate::web::local_site_key(&url)
            && !keys.contains(&key)
        {
            keys.push(key);
        }
    }
    keys
}

/// The effects of work that stays on this computer: writing its files and
/// starting or changing its processes.
const LOCAL_SETUP: crate::semantics::ToolMutationEffects =
    crate::semantics::ToolMutationEffects::LOCAL_SOURCE_WRITE
        .union(crate::semantics::ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
        .union(crate::semantics::ToolMutationEffects::LOCAL_DERIVED_WRITE)
        .union(crate::semantics::ToolMutationEffects::PROCESS_STATE);

/// Whether a call set something up on this computer, so the local sites
/// its output names are ones the run created: it ran (it did not fail or
/// get refused; a launch still running counts, a non-zero exit after it
/// ran does too) and its typed effects are local changes only, at least
/// one of them. Any effect beyond this computer (a delivery, a remote
/// change, an unspecified one) or no effect at all: its output may carry
/// content from elsewhere, so it sets nothing up.
pub fn set_up_locally(
    status: crate::semantics::ToolOutcomeStatus,
    semantics: &crate::semantics::ToolCallSemantics,
) -> bool {
    let effects = semantics.mutation_effects;
    !status.is_failure() && !effects.is_empty() && LOCAL_SETUP.contains(effects)
}

/// Whether a call that ran in folder `cwd` ran inside one of a run's
/// `write_roots`, so the local sites its output reports are served from the
/// run's own write scope. Compared by path component, so a folder never
/// counts as inside a sibling that shares its name's start
/// (`claude-drupal-2` is not inside `claude-drupal`). A run without write
/// roots (one not confined to any) has no scope to serve sites from.
pub fn served_within(cwd: &str, write_roots: &[String]) -> bool {
    let normal = |path: &str| {
        let expanded = shellexpand::tilde(path.trim()).into_owned();
        let path = std::path::PathBuf::from(expanded);
        path.is_absolute().then(|| {
            path.components()
                .filter(|part| !matches!(part, std::path::Component::CurDir))
                .collect::<std::path::PathBuf>()
        })
    };
    let Some(cwd) = normal(cwd) else {
        return false;
    };
    if cwd
        .components()
        .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return false;
    }
    write_roots
        .iter()
        .filter_map(|root| normal(root))
        .any(|root| root.parent().is_some() && cwd.starts_with(&root))
}

/// Note `keys` as sites the work in `grant_session` set up or was asked for.
pub fn note(grant_session: &str, keys: &[String]) {
    if grant_session.trim().is_empty() || keys.is_empty() {
        return;
    }
    let Ok(mut sites) = SITES.lock() else {
        return;
    };
    let now = Instant::now();
    let session = sites.entry(grant_session.to_string()).or_default();
    session.retain(|_, at| now.duration_since(*at) < KEEP_FOR);
    for key in keys {
        session.insert(key.clone(), now);
    }
    while session.len() > MAX_PER_SESSION {
        let Some(oldest) = session
            .iter()
            .min_by_key(|(_, at)| **at)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        session.remove(&oldest);
    }
}

/// Note the local sites written in `text` (the owner's request, or a
/// trusted tool output) for `grant_session`. Name lookups run off the async
/// workers.
pub async fn note_sites_in(grant_session: &str, text: &str) {
    if grant_session.trim().is_empty() || !text.contains("http") {
        return;
    }
    let text = text.to_string();
    let keys = tokio::task::spawn_blocking(move || local_sites_in(&text))
        .await
        .unwrap_or_default();
    note(grant_session, &keys);
}

/// Whether the work in `grant_session` set up the local site `key`, or the
/// owner's request named it.
pub fn is_noted(grant_session: &str, key: &str) -> bool {
    let Ok(sites) = SITES.lock() else {
        return false;
    };
    sites
        .get(grant_session)
        .and_then(|session| session.get(key))
        .is_some_and(|at| at.elapsed() < KEEP_FOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run may open a local site it set up itself (its own install
    /// command printed it), never one a read, a page or a delivery named.
    #[test]
    fn only_a_call_that_set_something_up_locally_brings_its_sites() {
        use crate::semantics::{ToolCallSemantics, ToolMutationEffects, ToolOutcomeStatus};
        let changed = |effects| ToolCallSemantics::observation_and_mutation_with(effects);
        let install = changed(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::PROCESS_STATE),
        );
        for status in [
            ToolOutcomeStatus::Succeeded,
            ToolOutcomeStatus::CompletedWithNegativeResult,
            ToolOutcomeStatus::Backgrounded,
        ] {
            assert!(set_up_locally(status, &install), "{status:?}");
        }
        for status in [
            ToolOutcomeStatus::FailedPermanent,
            ToolOutcomeStatus::FailedRetryable,
            ToolOutcomeStatus::Blocked,
        ] {
            assert!(!set_up_locally(status, &install), "{status:?}");
        }
        let succeeded = ToolOutcomeStatus::Succeeded;
        assert!(!set_up_locally(
            succeeded,
            &ToolCallSemantics::observation()
        ));
        for elsewhere in [
            ToolMutationEffects::EXTERNAL_DELIVERY,
            ToolMutationEffects::REMOTE_MUTATION,
            ToolMutationEffects::UNSPECIFIED,
            ToolMutationEffects::PROCESS_STATE.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        ] {
            assert!(
                !set_up_locally(succeeded, &changed(elsewhere)),
                "{elsewhere:?}"
            );
        }
    }

    /// Live 2026-11-13: a helper scoped to `~/projects/claude-drupal` was
    /// asked to approve opening `claude-drupal.ddev.site`, which that folder
    /// serves: the site was there before the run, so no set-up command of
    /// its own printed it. A command run inside the run's write scope
    /// reports its sites; one run outside it, or in a sibling folder whose
    /// name only starts the same, does not.
    #[test]
    fn a_run_may_open_a_site_served_from_inside_its_own_scope_only() {
        let scope = vec!["/Users/someone/projects/claude-drupal".to_string()];
        assert!(served_within(
            "/Users/someone/projects/claude-drupal",
            &scope
        ));
        assert!(served_within(
            "/Users/someone/projects/claude-drupal/web",
            &scope
        ));
        assert!(served_within(
            "/Users/someone/projects/claude-drupal/./web",
            &scope
        ));
        assert!(!served_within(
            "/Users/someone/projects/claude-drupal-2",
            &scope
        ));
        assert!(!served_within(
            "/Users/someone/projects/claude-wordpress",
            &scope
        ));
        assert!(!served_within("/Users/someone/projects", &scope));
        assert!(!served_within(
            "/Users/someone/projects/claude-drupal/../claude-wordpress",
            &scope
        ));
        assert!(!served_within("relative/claude-drupal", &scope));
        // No write scope: nothing is served from it, and the root never
        // counts as a scope that holds everything.
        assert!(!served_within("/Users/someone/projects/claude-drupal", &[]));
        assert!(!served_within("/Users/someone", &["/".to_string()]));
    }

    #[test]
    fn addresses_are_read_out_of_command_output() {
        let output = "Successfully started claude-drupal-2\n\
                      Your project can be reached at https://claude-drupal-2.ddev.site \
                      and http://127.0.0.1:32770.\n(see \"https://claude-drupal-2.ddev.site\")";
        assert_eq!(
            urls_in(output),
            vec![
                "https://claude-drupal-2.ddev.site".to_string(),
                "http://127.0.0.1:32770".to_string(),
            ]
        );
    }

    #[test]
    fn only_sites_on_this_computer_are_noted() {
        let keys = local_sites_in(
            "dev server on http://localhost:5173/ and docs at https://docs.example.invalid/x \
             and http://shop.localhost:8080/admin and http://10.0.0.5/",
        );
        assert_eq!(
            keys,
            vec!["localhost:5173".to_string(), "shop.localhost".to_string()]
        );
    }

    #[tokio::test]
    async fn a_noted_site_counts_only_for_its_own_session() {
        note_sites_in(
            "test:local-sites-a",
            "Started: http://blog-synthetic.localhost/",
        )
        .await;
        assert!(is_noted("test:local-sites-a", "blog-synthetic.localhost"));
        assert!(!is_noted("test:local-sites-b", "blog-synthetic.localhost"));
        assert!(!is_noted("test:local-sites-a", "localhost:8080"));
        // Nothing is noted without a session.
        note("", &["other.localhost".to_string()]);
        assert!(!is_noted("", "other.localhost"));
    }
}
