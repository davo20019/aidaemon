use crate::traits::SpecialistKind;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

use super::parse::parse_specialist;
use super::{SpecialistDef, SpecialistRegistry, SpecialistSource};

#[allow(dead_code)] // production wiring lands in Tasks 5-8; covered by tests now
const BUNDLED: &[(SpecialistKind, &str)] = &[
    (
        SpecialistKind::TaskLead,
        include_str!("../../../specialists/task_lead.md"),
    ),
    (
        SpecialistKind::Executor,
        include_str!("../../../specialists/executor.md"),
    ),
    (
        SpecialistKind::Research,
        include_str!("../../../specialists/research.md"),
    ),
    (
        SpecialistKind::ArtifactWriter,
        include_str!("../../../specialists/artifact_writer.md"),
    ),
    (
        SpecialistKind::Code,
        include_str!("../../../specialists/code.md"),
    ),
    (
        SpecialistKind::BrowserVerifier,
        include_str!("../../../specialists/browser_verifier.md"),
    ),
    (
        SpecialistKind::Review,
        include_str!("../../../specialists/review.md"),
    ),
    (
        SpecialistKind::CommsDraft,
        include_str!("../../../specialists/comms_draft.md"),
    ),
    (
        SpecialistKind::Generic,
        include_str!("../../../specialists/generic.md"),
    ),
];

#[allow(dead_code)] // production wiring lands in Tasks 5-8; covered by tests now
impl SpecialistRegistry {
    pub fn load(user_dir: Option<&Path>) -> Self {
        let mut by_kind = std::collections::HashMap::new();

        for (kind, content) in BUNDLED {
            match parse_specialist(*kind, content) {
                Ok(def) => {
                    by_kind.insert(*kind, Arc::new(def));
                }
                Err(e) => {
                    // include_str! caught file existence at compile time; a parse
                    // error here means the bundled .md is malformed — tests catch
                    // this before release.
                    panic!(
                        "bundled specialist {} failed to parse: {}",
                        kind.as_str(),
                        e
                    );
                }
            }
        }

        if let Some(dir) = user_dir {
            if dir.is_dir() {
                Self::load_user_overrides(dir, &mut by_kind);
            }
        }

        SpecialistRegistry { by_kind }
    }

    fn load_user_overrides(
        dir: &Path,
        by_kind: &mut std::collections::HashMap<SpecialistKind, Arc<SpecialistDef>>,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(err) => {
                warn!(dir = %dir.display(), error = %err, "could not read specialists override dir");
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let kind = match SpecialistKind::from_str(stem) {
                Some(k) => k,
                None => {
                    warn!(file = %path.display(), "specialist override has unknown kind in filename; ignored");
                    continue;
                }
            };
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(err) => {
                    warn!(file = %path.display(), error = %err, "could not read specialist override; keeping bundled");
                    continue;
                }
            };
            match parse_specialist(kind, &content) {
                Ok(mut def) => {
                    def.source = SpecialistSource::UserOverride(path.clone());
                    by_kind.insert(kind, Arc::new(def));
                }
                Err(err) => {
                    warn!(file = %path.display(), error = %err, "specialist override failed to parse; keeping bundled");
                }
            }
        }
    }

    pub fn get(&self, kind: SpecialistKind) -> Arc<SpecialistDef> {
        // Infallible by construction — every SpecialistKind variant has a bundled file.
        self.by_kind
            .get(&kind)
            .cloned()
            .unwrap_or_else(|| panic!("specialist registry missing kind: {}", kind.as_str()))
    }

    pub fn kinds(&self) -> impl Iterator<Item = SpecialistKind> + '_ {
        self.by_kind.keys().copied()
    }

    pub fn render(&self, kind: SpecialistKind, ctx: &super::SpecialistRenderContext) -> String {
        let def = self.get(kind);
        super::render::render_template(&def.system_prompt_template, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_specialist_kind_has_a_bundled_definition() {
        let registry = SpecialistRegistry::load(None);
        for kind in SpecialistKind::all() {
            let def = registry.get(*kind);
            assert_eq!(def.kind, *kind);
            assert!(
                !def.description.trim().is_empty(),
                "{:?} has empty description",
                kind
            );
            assert!(
                !def.system_prompt_template.trim().is_empty(),
                "{:?} has empty body",
                kind
            );
        }
    }
}
