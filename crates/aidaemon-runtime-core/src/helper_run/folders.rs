//! Folders made fresh for a helper's work: creating one claims its name
//! for the request, and a helper that never started gives it back.

/// Make each of `folders` fresh and empty for a child's work, and return
/// them as absolute paths: the one place a spawn's `new_folders` become real.
/// Creating the folder is what claims its name, so two siblings can never
/// both start in it, and the child finds it there to work in (live
/// 2026-11-13: a helper sent to install into a folder that did not exist yet
/// found it missing and gave up). A folder that already exists, or a path
/// that is not absolute, is refused with the next free name, and nothing is
/// created (live 2026-11-13: "fresh" installs went into claude-drupal-3,
/// which held an earlier site, and preserved it).
pub fn claim_new_folders(folders: &[String]) -> anyhow::Result<Vec<String>> {
    let mut wanted = Vec::with_capacity(folders.len());
    let mut refused = Vec::new();
    for folder in folders {
        let path = std::path::PathBuf::from(shellexpand::tilde(folder.trim()).as_ref());
        if !path.is_absolute() {
            refused.push(format!("{} is not an absolute path", path.display()));
        } else if path.exists() {
            refused.push(format!(
                "{} already exists (the next free name is {})",
                path.display(),
                next_free_folder(&path).display()
            ));
        } else if !wanted.contains(&path) {
            wanted.push(path);
        }
    }
    let mut made: Vec<std::path::PathBuf> = Vec::new();
    if refused.is_empty() {
        for path in &wanted {
            let parent_ready = path.parent().map_or(Ok(()), std::fs::create_dir_all);
            match parent_ready.and_then(|()| std::fs::create_dir(path)) {
                Ok(()) => made.push(path.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    refused.push(format!(
                        "{} already exists (the next free name is {})",
                        path.display(),
                        next_free_folder(path).display()
                    ));
                    break;
                }
                Err(error) => {
                    refused.push(format!("{} could not be created: {error}", path.display()));
                    break;
                }
            }
        }
    }
    if !refused.is_empty() {
        release_new_folders(&made);
        anyhow::bail!(
            "No helper started: {}. New work needs an absolute folder that doesn't exist yet. \
             Call again with the free folder in title, task, scope and new_folders, or ask \
             the owner whether to use the existing one.",
            refused.join("; ")
        );
    }
    Ok(made
        .into_iter()
        .map(|path| path.display().to_string())
        .collect())
}

/// Give back folders [`claim_new_folders`] made for a child that did not
/// start or that failed: each is removed only while it is still empty, so
/// the name is free for the next try.
pub fn release_new_folders<P: AsRef<std::path::Path>>(folders: &[P]) {
    for folder in folders {
        let _ = std::fs::remove_dir(folder.as_ref());
    }
}

/// The request tree a call belongs to: the folders its work made for
/// itself are recorded there, for every run of the tree.
fn request_folders(
    facts: &aidaemon_tools::semantics::ToolCallContext,
) -> Option<&aidaemon_tools::semantics::OwnedFolders> {
    crate::authority::parent_authority(facts)
        .and_then(|parent| parent.spawn_family.as_ref())
        .map(crate::authority::SpawnFamily::owned_folders)
}

/// [`claim_new_folders`] for a helper call: the folders made are recorded
/// as the request's own ([`crate::authority::SpawnFamily::owned_folders`]),
/// a typed fact every run of the request reads, so a change or a delete
/// inside them is never taken for acting on something the work did not
/// make.
pub fn claim_request_folders(
    folders: &[String],
    facts: &aidaemon_tools::semantics::ToolCallContext,
) -> anyhow::Result<Vec<String>> {
    let made = claim_new_folders(folders)?;
    if let Some(owned) = request_folders(facts) {
        owned.claim(&made);
    }
    Ok(made)
}

/// [`release_new_folders`] for a helper call that did not start: a folder
/// removed again is no longer the request's.
pub fn release_request_folders(
    folders: &[String],
    facts: &aidaemon_tools::semantics::ToolCallContext,
) {
    release_new_folders(folders);
    if let Some(owned) = request_folders(facts) {
        let gone: Vec<&String> = folders
            .iter()
            .filter(|folder| !std::path::Path::new(folder.as_str()).exists())
            .collect();
        owned.release(&gone);
    }
}

/// The line a child's task gets for the folders made for it: they exist,
/// empty, and are where its work goes.
pub fn new_folders_note(folders: &[String]) -> String {
    match folders {
        [] => String::new(),
        [one] => format!(
            "\n\nThe folder for this work, {one}, was just created for you and is empty: \
             work in it. It is part of what you may change."
        ),
        many => format!(
            "\n\nThe folders for this work were just created for you and are empty: {}. \
             Work in them. They are part of what you may change.",
            many.join(", ")
        ),
    }
}

/// The first sibling of `path` named like it with a higher number
/// (`site-3` -> `site-4`, `site` -> `site-2`) that does not exist.
pub fn next_free_folder(path: &std::path::Path) -> std::path::PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let (stem, mut number) = match name.rsplit_once('-') {
        Some((stem, digits))
            if !stem.is_empty()
                && !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit()) =>
        {
            (stem.to_string(), digits.parse::<u64>().unwrap_or(1))
        }
        _ => (name.clone(), 1),
    };
    loop {
        number += 1;
        let candidate = path.with_file_name(format!("{stem}-{number}"));
        if !candidate.exists() {
            return candidate;
        }
    }
}

#[cfg(test)]
mod folder_tests {
    use super::*;

    /// The folders a helper call makes fresh are recorded as the request's
    /// own, for every run of its tree; a folder removed again because its
    /// helper never started is not.
    #[test]
    fn folders_a_helper_call_makes_are_the_requests_own() {
        let base = tempfile::tempdir().unwrap();
        let family = crate::authority::SpawnFamily::new(Default::default());
        let mut facts = aidaemon_tools::semantics::ToolCallContext::default();
        facts
            .extensions
            .insert(crate::authority::InheritedAuthority {
                authority: aidaemon_core::Authority::full(),
                owner_level: None,
                task_access: None,
                mode_session: None,
                untrusted_context: false,
                origin_request: None,
                origin_trusted: true,
                trusted_roots: Vec::new(),
                declined: Vec::new(),
                shared_declines: None,
                learning: None,
                spawn_family: Some(family.clone()),
                child_declaration: None,
                parent_run: None,
                known_resources: Default::default(),
                card_title: None,
                card_steps: Vec::new(),
                lineage: Vec::new(),
                parent_task: None,
            });
        let folder = base.path().join("claude-drupal-9").display().to_string();
        let made = claim_request_folders(std::slice::from_ref(&folder), &facts).unwrap();
        assert_eq!(made, vec![folder.clone()]);
        assert_eq!(family.owned_folders().list(), made);
        assert!(aidaemon_tools::semantics::within_owned(
            &format!("{folder}/web/pass-reset.php"),
            &family.owned_folders().list()
        ));
        release_request_folders(&made, &facts);
        assert!(family.owned_folders().list().is_empty());
    }
}
