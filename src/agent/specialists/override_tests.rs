#![cfg(test)]

use super::{SpecialistRegistry, SpecialistSource};
use crate::traits::SpecialistKind;
use std::fs;
use tempfile::TempDir;

#[test]
fn override_file_replaces_bundled_entry() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("code.md");
    fs::write(
        &path,
        "---\nkind: code\ndescription: My custom code specialist.\n---\nCustom body for {{mission}}.\n",
    )
    .unwrap();

    let registry = SpecialistRegistry::load(Some(dir.path()));
    let def = registry.get(SpecialistKind::Code);
    assert_eq!(def.description, "My custom code specialist.");
    assert!(matches!(def.source, SpecialistSource::UserOverride(_)));
}

#[test]
fn override_with_unknown_filename_is_ignored() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_a_kind.md");
    fs::write(&path, "---\nkind: code\ndescription: x.\n---\nBody.\n").unwrap();

    let registry = SpecialistRegistry::load(Some(dir.path()));
    let def = registry.get(SpecialistKind::Code);
    assert!(matches!(def.source, SpecialistSource::Bundled));
}

#[test]
fn override_with_kind_mismatch_falls_back_to_bundled() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("code.md");
    fs::write(&path, "---\nkind: research\ndescription: bad.\n---\nBody.\n").unwrap();

    let registry = SpecialistRegistry::load(Some(dir.path()));
    let def = registry.get(SpecialistKind::Code);
    assert!(matches!(def.source, SpecialistSource::Bundled));
}

#[test]
fn missing_override_dir_is_silent() {
    let registry = SpecialistRegistry::load(Some(std::path::Path::new(
        "/tmp/aidaemon-nonexistent-override-dir-for-tests",
    )));
    for kind in SpecialistKind::all() {
        assert!(matches!(registry.get(*kind).source, SpecialistSource::Bundled));
    }
}
