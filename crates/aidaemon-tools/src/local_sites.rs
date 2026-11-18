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

/// The folders each grant session's request owns ([`note_folders`]): a
/// site one of them serves, by the project's own configuration, is the
/// request's own.
static FOLDERS: LazyLock<Mutex<HashMap<String, Vec<String>>>> = LazyLock::new(Mutex::default);

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
/// owner's request named it, or one of the folders its request owns serves
/// it ([`note_folders`], [`project_hosts`]).
pub fn is_noted(grant_session: &str, key: &str) -> bool {
    let noted = SITES.lock().is_ok_and(|sites| {
        sites
            .get(grant_session)
            .and_then(|session| session.get(key))
            .is_some_and(|at| at.elapsed() < KEEP_FOR)
    });
    noted || served_from_owned(grant_session, key)
}

/// Record the folders the request in `grant_session` owns (made by its own
/// work: [`crate::semantics::OwnedFolders`]). Whatever their projects
/// serve, now or once configured, is the request's own site; nothing else
/// is added by them.
pub fn note_folders(grant_session: &str, folders: &[String]) {
    if grant_session.trim().is_empty() || folders.is_empty() {
        return;
    }
    let Ok(mut known) = FOLDERS.lock() else {
        return;
    };
    let kept = known.entry(grant_session.to_string()).or_default();
    for folder in folders {
        if std::path::Path::new(folder.trim()).is_absolute() && !kept.contains(folder) {
            kept.push(folder.clone());
        }
    }
    let excess = kept.len().saturating_sub(MAX_PER_SESSION);
    kept.drain(..excess);
}

/// Whether one of the folders `grant_session`'s request owns serves the
/// site `key` by its project's own configuration.
fn served_from_owned(grant_session: &str, key: &str) -> bool {
    let folders = FOLDERS
        .lock()
        .ok()
        .and_then(|known| known.get(grant_session).cloned())
        .unwrap_or_default();
    !folders_serving(key, &folders).is_empty()
}

/// The folders among `folders` whose project configuration serves `host`
/// (compared without case). A folder with no project configuration serves
/// nothing.
pub fn folders_serving(host: &str, folders: &[String]) -> Vec<String> {
    let host = host.trim().trim_end_matches('.').to_lowercase();
    if host.is_empty() {
        return Vec::new();
    }
    folders
        .iter()
        .filter(|folder| project_hosts(std::path::Path::new(folder.trim())).contains(&host))
        .cloned()
        .collect()
}

/// Name, on an observation, the project folder (among `folders`) that
/// serves each site it observed, by that project's own configuration
/// ([`project_hosts`]): a check of the site is a check of what the folder
/// serves, so a step aimed at the folder and a step aimed at its site are
/// matched by what they resolve to, never by their text. A call that
/// observes nothing, or names no URL, is left as it is.
pub fn name_serving_folders(
    semantics: &mut crate::semantics::ToolCallSemantics,
    folders: &[String],
) {
    if !semantics.observes_state() || folders.is_empty() {
        return;
    }
    let hosts: Vec<String> = semantics
        .target_hints
        .iter()
        .filter(|hint| hint.kind == crate::semantics::ToolTargetHintKind::Url)
        .filter_map(|hint| reqwest::Url::parse(&hint.value).ok())
        .filter_map(|url| url.host_str().map(str::to_string))
        .collect();
    for host in hosts {
        for folder in folders_serving(&host, folders) {
            let named = semantics.target_hints.iter().any(|hint| {
                hint.kind == crate::semantics::ToolTargetHintKind::Path && hint.value == folder
            });
            if !named {
                *semantics = std::mem::take(semantics)
                    .with_target_hint(crate::semantics::ToolTargetHintKind::Path, folder);
            }
        }
    }
}

/// The host names a local project folder serves, read from the project's
/// own configuration: a `ddev` project's `.ddev/config.yaml` (and its
/// `config.*.yaml` overrides, applied in name order as `ddev` does) gives
/// its `name` (the folder's name when unset), `project_tld` (else the
/// global configuration's, else `ddev`'s default), `additional_hostnames`
/// (each under that domain) and `additional_fqdns`. Lowercase, without
/// duplicates; empty when the folder holds no such project.
pub fn project_hosts(folder: &std::path::Path) -> Vec<String> {
    const DDEV_DEFAULT_TLD: &str = "ddev.site";
    let config_dir = folder.join(".ddev");
    let Ok(base) = std::fs::read_to_string(config_dir.join("config.yaml")) else {
        return Vec::new();
    };
    let mut overrides: Vec<std::path::PathBuf> = std::fs::read_dir(&config_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("config.")
                                && name.ends_with(".yaml")
                                && name != "config.yaml"
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    overrides.sort();
    let mut config = TopLevel::default();
    config.apply(&base);
    for path in overrides {
        if let Ok(text) = std::fs::read_to_string(path) {
            config.apply(&text);
        }
    }
    let name = config
        .scalar("name")
        .or_else(|| {
            folder
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let tld = config
        .scalar("project_tld")
        .or_else(|| {
            let home = dirs::home_dir()?;
            let text = std::fs::read_to_string(home.join(".ddev/global_config.yaml")).ok()?;
            let mut global = TopLevel::default();
            global.apply(&text);
            global.scalar("project_tld")
        })
        .unwrap_or_else(|| DDEV_DEFAULT_TLD.to_string());
    let tld = tld.trim_matches('.');
    let mut hosts: Vec<String> = Vec::new();
    let mut add = |host: String| {
        let host = host.trim().trim_end_matches('.').to_lowercase();
        if !host.is_empty() && !hosts.contains(&host) {
            hosts.push(host);
        }
    };
    if !name.trim().is_empty() {
        add(format!("{}.{tld}", name.trim()));
    }
    for extra in config.list("additional_hostnames") {
        add(format!("{extra}.{tld}"));
    }
    for fqdn in config.list("additional_fqdns") {
        add(fqdn);
    }
    hosts
}

/// The top-level keys of a project's YAML configuration file: scalars and
/// lists of scalars (block `- item` or flow `[a, b]`), which is all a
/// project's served names are written as. Nested mappings are skipped.
#[derive(Default)]
struct TopLevel {
    values: HashMap<String, Vec<String>>,
}

impl TopLevel {
    /// Read `text`, its keys replacing earlier ones (an override file).
    fn apply(&mut self, text: &str) {
        let mut current: Option<String> = None;
        for line in text.lines() {
            let content = strip_comment(line);
            if content.trim().is_empty() {
                continue;
            }
            let indented = content.starts_with([' ', '\t']);
            let trimmed = content.trim();
            if indented || trimmed.starts_with('-') {
                if let (Some(key), Some(item)) = (current.as_ref(), trimmed.strip_prefix('-')) {
                    let item = unquote(item.trim());
                    if !item.is_empty() {
                        self.values.entry(key.clone()).or_default().push(item);
                    }
                }
                continue;
            }
            current = None;
            let Some((key, value)) = trimmed.split_once(':') else {
                continue;
            };
            let key = key.trim().to_string();
            let value = value.trim();
            let items: Vec<String> = match value.strip_prefix('[').and_then(|v| v.strip_suffix(']'))
            {
                Some(flow) => flow
                    .split(',')
                    .map(|item| unquote(item.trim()))
                    .filter(|item| !item.is_empty())
                    .collect(),
                None if value.is_empty() => {
                    current = Some(key.clone());
                    Vec::new()
                }
                None => vec![unquote(value)],
            };
            self.values.insert(key, items);
        }
    }

    fn scalar(&self, key: &str) -> Option<String> {
        match self.values.get(key).map(Vec::as_slice) {
            Some([one]) if !one.is_empty() => Some(one.clone()),
            _ => None,
        }
    }

    fn list(&self, key: &str) -> Vec<String> {
        self.values.get(key).cloned().unwrap_or_default()
    }
}

/// A YAML line without its trailing comment (a `#` outside quotes that
/// starts the line or follows a space).
fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut previous = ' ';
    for (at, c) in line.char_indices() {
        match quote {
            Some(open) if c == open => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' && previous.is_whitespace() => return &line[..at],
            None => {}
        }
        previous = c;
    }
    line
}

/// A YAML scalar without its surrounding quotes.
fn unquote(value: &str) -> String {
    let value = value.trim();
    for q in ['"', '\''] {
        if let Some(inner) = value.strip_prefix(q).and_then(|v| v.strip_suffix(q)) {
            return inner.to_string();
        }
    }
    value.to_string()
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

    fn ddev_project(root: &std::path::Path, folder: &str, config: &str) -> String {
        let folder = root.join(folder);
        std::fs::create_dir_all(folder.join(".ddev")).unwrap();
        std::fs::write(folder.join(".ddev/config.yaml"), config).unwrap();
        folder.display().to_string()
    }

    /// The names a project serves come from its own configuration, not
    /// from the shape of a host name.
    #[test]
    fn a_project_serves_the_names_its_configuration_gives() {
        let root = tempfile::tempdir().unwrap();
        let site = ddev_project(
            root.path(),
            "claude-company-wordpress-20261113",
            "name: claude-company-wordpress-20261113 # the project\n\
             type: wordpress\n\
             additional_hostnames:\n  - \"admin\"\n\
             additional_fqdns: [shop.example.test]\n\
             web_environment: []\n\
             hooks:\n  post-start:\n    - exec: echo hi\n",
        );
        assert_eq!(
            project_hosts(std::path::Path::new(&site)),
            vec![
                "claude-company-wordpress-20261113.ddev.site".to_string(),
                "admin.ddev.site".to_string(),
                "shop.example.test".to_string(),
            ]
        );
        // An override file changes the name; the project's own TLD wins.
        std::fs::write(
            std::path::Path::new(&site).join(".ddev/config.local.yaml"),
            "name: renamed\nproject_tld: test\nadditional_hostnames: []\n",
        )
        .unwrap();
        assert_eq!(
            project_hosts(std::path::Path::new(&site)),
            vec!["renamed.test".to_string(), "shop.example.test".to_string()]
        );
        // No project configuration: it serves nothing.
        let plain = root.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert!(project_hosts(&plain).is_empty());
    }

    /// Live 2026-11-13: a helper that created a `ddev` project in its own
    /// new folder was asked to approve opening that project's site. A site
    /// served from a folder the request owns is its own; another project's
    /// site stays gated.
    #[test]
    fn a_site_served_from_an_owned_folder_is_the_requests_own() {
        let root = tempfile::tempdir().unwrap();
        let own = ddev_project(root.path(), "own-site", "name: own-site\n");
        ddev_project(root.path(), "other-site", "name: other-site\n");
        let grant = "test:local-sites-owned";
        assert!(!is_noted(grant, "own-site.ddev.site"));
        note_folders(grant, std::slice::from_ref(&own));
        assert!(is_noted(grant, "own-site.ddev.site"));
        assert!(is_noted(grant, "OWN-SITE.ddev.site"));
        assert!(!is_noted(grant, "other-site.ddev.site"));
        assert!(!is_noted(
            "test:local-sites-elsewhere",
            "own-site.ddev.site"
        ));
    }

    /// A check of a site names the project folder that serves it, so a
    /// step aimed at that folder is matched by what the check observed.
    #[test]
    fn an_observation_of_a_site_names_the_folder_that_serves_it() {
        use crate::semantics::{ToolCallSemantics, ToolTargetHintKind};
        let root = tempfile::tempdir().unwrap();
        let site = ddev_project(root.path(), "site-a", "name: site-a\n");
        let other = ddev_project(root.path(), "site-b", "name: site-b\n");
        let folders = vec![site.clone(), other];
        let paths = |semantics: &ToolCallSemantics| -> Vec<String> {
            semantics
                .target_hints
                .iter()
                .filter(|hint| hint.kind == ToolTargetHintKind::Path)
                .map(|hint| hint.value.clone())
                .collect()
        };
        let mut check = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://site-a.ddev.site/");
        name_serving_folders(&mut check, &folders);
        assert_eq!(paths(&check), vec![site.clone()]);
        name_serving_folders(&mut check, &folders);
        assert_eq!(paths(&check), vec![site], "named once");
        // A change names no folder by it: only an observation checks.
        let mut change = ToolCallSemantics::mutation()
            .with_target_hint(ToolTargetHintKind::Url, "https://site-a.ddev.site/");
        name_serving_folders(&mut change, &folders);
        assert!(paths(&change).is_empty());
        // A site no folder serves names none.
        let mut elsewhere = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://site-c.ddev.site/");
        name_serving_folders(&mut elsewhere, &folders);
        assert!(paths(&elsewhere).is_empty());
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
