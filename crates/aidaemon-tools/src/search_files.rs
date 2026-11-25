//! `search_files`: a budgeted, breadth-first search by filename and content,
//! through an [`ExecutionBackend`](crate::backend::ExecutionBackend).

use std::sync::Arc;

use aidaemon_core::{AccessManifest, ToolAnnotations};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};

use crate::backend::{BackendFileType, BackendPath};
use crate::fs::{DEFAULT_IGNORE_DIRS, FilePolicy, SharedBackend, ToolText, workspace_parts};
use crate::outcome::{OutcomeTool, core_tool};
use crate::semantics::{
    ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole,
    ToolTargetHintKind, ToolVerificationMode,
};
use crate::workspace::Workspace;

pub struct SearchFiles {
    backend: SharedBackend,
    policy: Arc<dyn FilePolicy>,
    text: ToolText,
}

impl SearchFiles {
    /// A tool confined to `workspace`.
    pub fn new(workspace: Workspace) -> Self {
        let (backend, policy) = workspace_parts(workspace);
        Self::on_backend(backend, policy)
    }

    /// A tool working through `backend` under an application's `policy`.
    pub fn on_backend(backend: SharedBackend, policy: Arc<dyn FilePolicy>) -> Self {
        Self {
            backend,
            policy,
            text: ToolText {
                short: "Search for files by name pattern or content regex".into(),
                schema: "Search for files by name pattern (glob) and/or content (regex). Skips \
                         .git, node_modules, target and similar directories."
                    .into(),
            },
        }
    }

    /// Replace the model-facing descriptions.
    pub fn with_text(mut self, text: ToolText) -> Self {
        self.text = text;
        self
    }
}

const MAX_RESULTS: usize = 200;
const DEFAULT_MAX_RESULTS: usize = 50;
const MAX_FILES_SCANNED: usize = 10_000;
const MAX_DEPTH: usize = 20;
/// Total directory entries one walk may visit before returning a partial
/// result. Keeps whole-home searches fast instead of watchdog-fatal.
const MAX_ENTRIES_VISITED: usize = 150_000;
const MAX_CONTENT_SEARCH_FILE_SIZE: u64 = 1024 * 1024; // 1 MiB per file
/// Matching lines shown per file unless `max_per_file` asks for more.
const DEFAULT_MATCHES_PER_FILE: usize = 5;
const MAX_MATCHES_PER_FILE: usize = 50;
/// The most `context_lines` a call may ask for around each match.
const MAX_CONTEXT_LINES: usize = 5;

#[async_trait]
impl OutcomeTool for SearchFiles {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        &self.text.short
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_files",
            "description": self.text.schema,
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search file contents for"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Filename glob pattern (e.g., '*.rs', '*.ts', 'Cargo.*')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in, or one file to search (default: current directory)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum files to return (default: 50, max: 200)"
                    },
                    "max_per_file": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Matching lines shown per file (default: 5, max: 50); the rest are counted"
                    },
                    "context_lines": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Lines of context shown before and after each match (default: 0, max: 5)"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match the pattern regardless of letter case (default: false)"
                    }
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run(arguments, facts)
            .await
            .map(ToolCallOutcome::from_output)
    }
}

core_tool!(
    SearchFiles,
    annotations: ToolAnnotations::READ_ONLY,
    access: |this, arguments| AccessManifest {
        reads: vec![this.policy.declare_path(
            arguments.get("path").and_then(Value::as_str).unwrap_or(".")
        )],
        ..Default::default()
    }
);

impl SearchFiles {
    /// Run one call with the dispatcher's `facts`.
    pub(crate) async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let content_pattern = args["pattern"].as_str();
        let glob_pattern = args["glob"].as_str();
        let path_value = args.get("path").and_then(|v| v.as_str());
        let path_str = path_value.unwrap_or(".");
        let used_default_path = path_value.is_none();
        let max_results = args["max_results"]
            .as_u64()
            .map(|n| (n as usize).min(MAX_RESULTS))
            .unwrap_or(DEFAULT_MAX_RESULTS);
        let shape = MatchShape {
            per_file: args["max_per_file"]
                .as_u64()
                .map(|n| (n as usize).clamp(1, MAX_MATCHES_PER_FILE))
                .unwrap_or(DEFAULT_MATCHES_PER_FILE),
            context: args["context_lines"]
                .as_u64()
                .map(|n| (n as usize).min(MAX_CONTEXT_LINES))
                .unwrap_or(0),
        };
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);

        if content_pattern.is_none() && glob_pattern.is_none() {
            anyhow::bail!(
                "At least one of 'pattern' (content regex) or 'glob' (filename pattern) is required"
            );
        }

        let backend = self.backend.clone();
        let mut search_dir = backend.resolve_path(path_str).await?;
        if backend.metadata(&search_dir).await.is_err() {
            // Paths copy-pasted from prose or earlier tool output often carry
            // trailing punctuation ("…/projects)" from a parenthesized
            // mention). Only fall back when the literal path is missing and
            // the trimmed one is a real directory.
            let trimmed = path_str
                .trim_end()
                .trim_end_matches([')', ':', ',', ';', '.'])
                .trim_end();
            if trimmed != path_str
                && !trimmed.is_empty()
                && let Ok(candidate) = backend.resolve_path(trimmed).await
                && backend
                    .metadata(&candidate)
                    .await
                    .is_ok_and(|metadata| metadata.is_dir())
            {
                search_dir = candidate;
            }
        }
        let search_metadata = backend
            .metadata(&search_dir)
            .await
            .map_err(|_| crate::TargetNotFound(format!("Directory not found: {search_dir}")))?;
        // One file is searched on its own (live 2026-11-14 to 11-20: a file
        // `path` failed "Not a directory" nine times and was redone with grep).
        let single_file = !search_metadata.is_dir();
        let collaboration_root = if let Some(root) = facts.workspace_collaboration_root.as_deref() {
            anyhow::ensure!(
                backend.kind() == crate::backend::BackendKind::Local,
                "Workspace collaboration requires a local execution backend"
            );
            let resolved_root = backend.resolve_path(root).await?;
            let canonical_root = backend.canonicalize(&resolved_root).await?;
            anyhow::ensure!(
                canonical_root.as_str() == root,
                "Delegated project path changed after authorization"
            );
            let canonical_search = backend.canonicalize(&search_dir).await?;
            anyhow::ensure!(
                std::path::Path::new(canonical_search.as_str())
                    .starts_with(std::path::Path::new(canonical_root.as_str())),
                "Search path is outside the delegated project"
            );
            search_dir = canonical_search;
            Some(canonical_root)
        } else {
            None
        };

        let content_regex = if let Some(pat) = content_pattern {
            Some(
                RegexBuilder::new(pat)
                    .case_insensitive(case_insensitive)
                    .build()
                    .map_err(|e| anyhow::anyhow!("Invalid regex '{}': {}", pat, e))?,
            )
        } else {
            None
        };

        let glob_matcher = if let Some(g) = glob_pattern {
            Some(compile_glob(g)?)
        } else {
            None
        };

        let mut results = Vec::new();
        let mut stats = SearchStats::default();

        if single_file {
            if self
                .policy
                .is_sensitive(std::path::Path::new(search_dir.as_str()))
            {
                anyhow::bail!("Refusing to search a protected file: {}", search_dir);
            }
            search_one_file(
                &backend,
                &search_dir,
                search_metadata.len,
                &content_regex,
                &glob_matcher,
                shape,
                &mut results,
                &mut stats,
            )
            .await;
        } else {
            walk_dir(
                backend,
                self.policy.as_ref(),
                &search_dir,
                &content_regex,
                &glob_matcher,
                max_results,
                MAX_ENTRIES_VISITED,
                collaboration_root.as_ref(),
                shape,
                &mut results,
                &mut stats,
            )
            .await;
        }

        let default_path_note = used_default_path.then(|| {
            format!(
                "Note: no 'path' was provided, so search_files defaulted to current directory: {}",
                search_dir
            )
        });

        if results.is_empty() {
            // The scanned dir must end its line cleanly: models copy-paste it
            // into later calls, and a glued ")" produces invalid paths.
            let mut output = format!(
                "No matches found. {} files scanned in {}",
                stats.files_scanned, search_dir
            );
            if stats.truncated {
                output.push('\n');
                output.push_str(&truncation_note(&stats));
            }
            if let Some(note) = &default_path_note {
                output.push('\n');
                output.push_str(note);
                // The model gave no 'path' and found nothing in cwd — point it
                // at common user locations so it retries instead of giving up.
                output.push('\n');
                output.push_str(
                    "The file may live elsewhere. Retry with an explicit 'path' — common user locations: ~/Downloads, ~/Desktop, ~/Documents, or ~ to search the whole home directory.",
                );
                output.push('\n');
                output.push_str(&whole_machine_search_hint(self.policy.shell_tool_name()));
            }
            if stats.oversized_files_skipped > 0 {
                output.push('\n');
                output.push_str(&oversized_note(stats.oversized_files_skipped));
            }
            return Ok(output);
        }

        let mut output = String::new();
        if let Some(note) = &default_path_note {
            output.push_str(note);
            output.push_str("\n\n");
        }
        // Content matches are counted as lines, files as files: "1 match"
        // over ten matching lines read as a contradiction (live 2026-11-21).
        let matching_lines: usize = results
            .iter()
            .map(|result| result.lines.iter().filter(|line| line.2).count() + result.more)
            .sum();
        let files = format!(
            "{} file{}",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        );
        let found = if matching_lines > 0 {
            format!(
                "Found {matching_lines} matching line{} in {files}",
                if matching_lines == 1 { "" } else { "s" }
            )
        } else {
            format!("Found {files}")
        };
        output.push_str(&format!(
            "{found}. {} files scanned in {}\n\n",
            stats.files_scanned, search_dir
        ));
        if stats.oversized_files_skipped > 0 {
            output.push_str(&oversized_note(stats.oversized_files_skipped));
            output.push_str("\n\n");
        }

        for result in &results {
            output.push_str(&result.format());
            output.push('\n');
        }

        if results.len() >= max_results {
            output.push_str(&format!(
                "\n(Results capped at {}. Use a more specific pattern or glob to narrow results.)",
                max_results
            ));
        }

        Ok(output)
    }
}

impl crate::semantics::ToolSemantics for SearchFiles {
    fn dispatch_facts(&self) -> crate::semantics::ToolDispatchFacts {
        crate::semantics::ToolDispatchFacts {
            offering: crate::semantics::Offering::Core,
            same_tool_streak: crate::semantics::SameToolStreak::Versatile,
            recap: crate::semantics::RecapKind::Search,
            activity: crate::semantics::ToolActivity::Fixed(
                aidaemon_core::card::ActivityKind::Search,
            ),
            project_dir_argument: crate::semantics::ProjectDirArgument::SearchRoot,
            evidence: crate::semantics::EvidenceSource::FileSearch,
            semantic_scope: Some(crate::semantics::ToolSemanticScope::LocalWorkspace),
            status_label: Some("searching files"),
            ..Default::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn output_trust(
        &self,
        arguments: &str,
        roots: &crate::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        // A search with no `path` runs in the working directory.
        let args = serde_json::from_str::<Value>(arguments).ok();
        let path = match args.as_ref().map(|args| args.get("path")) {
            Some(None) | Some(Some(Value::Null)) => Some("."),
            Some(Some(value)) => value.as_str(),
            None => None,
        };
        match path {
            Some(path) if roots.contains(path) => aidaemon_core::SourceTrust::Trusted,
            _ => aidaemon_core::SourceTrust::Untrusted,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                args.get("path")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();

        ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Path, path)
    }
}

fn oversized_note(count: usize) -> String {
    format!(
        "Skipped {} oversized file{} larger than {} bytes during content search.",
        count,
        if count == 1 { "" } else { "s" },
        MAX_CONTENT_SEARCH_FILE_SIZE
    )
}

struct SearchResult {
    path: BackendPath,
    /// (line number, text, whether the line matched): matches and, with
    /// `context_lines`, the lines around them.
    lines: Vec<(usize, String, bool)>,
    /// Matching lines past `max_per_file`, counted, not shown.
    more: usize,
}

/// How much of each file's matches a result shows.
#[derive(Clone, Copy)]
struct MatchShape {
    per_file: usize,
    context: usize,
}

/// A file's shown lines before its path is known.
struct FoundLines {
    lines: Vec<(usize, String, bool)>,
    more: usize,
}

impl FoundLines {
    fn at(self, path: BackendPath) -> SearchResult {
        SearchResult {
            path,
            lines: self.lines,
            more: self.more,
        }
    }
}

/// The lines of `bytes` that match `re`, the first `shape.per_file` shown
/// with `shape.context` lines around them (overlapping context merged), the
/// rest counted. `None` when nothing matches.
fn search_content(bytes: &[u8], re: &Regex, shape: MatchShape) -> Option<FoundLines> {
    let content = String::from_utf8_lossy(bytes);
    let all: Vec<&str> = content.lines().collect();
    let matched: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, line)| re.is_match(line))
        .map(|(index, _)| index)
        .collect();
    if matched.is_empty() {
        return None;
    }
    let shown = &matched[..matched.len().min(shape.per_file)];
    let mut keep = std::collections::BTreeSet::new();
    for &index in shown {
        let from = index.saturating_sub(shape.context);
        let to = (index + shape.context).min(all.len().saturating_sub(1));
        keep.extend(from..=to);
    }
    let shown_set: std::collections::BTreeSet<usize> = shown.iter().copied().collect();
    Some(FoundLines {
        lines: keep
            .into_iter()
            .map(|index| {
                (
                    index + 1,
                    all[index].to_string(),
                    shown_set.contains(&index),
                )
            })
            .collect(),
        more: matched.len() - shown.len(),
    })
}

/// A search whose `path` is one file: that file alone, by the same rules a
/// walk applies to each file it finds.
#[allow(clippy::too_many_arguments)]
async fn search_one_file(
    backend: &SharedBackend,
    path: &BackendPath,
    len: u64,
    content_regex: &Option<Regex>,
    glob_matcher: &Option<GlobMatcher>,
    shape: MatchShape,
    results: &mut Vec<SearchResult>,
    stats: &mut SearchStats,
) {
    let file_name = path.file_name().unwrap_or_default().to_string();
    if glob_matcher
        .as_ref()
        .is_some_and(|matcher| !matcher.is_match(&file_name))
    {
        return;
    }
    stats.files_scanned += 1;
    let Some(content_re) = content_regex else {
        results.push(SearchResult {
            path: path.clone(),
            lines: Vec::new(),
            more: 0,
        });
        return;
    };
    if len > MAX_CONTENT_SEARCH_FILE_SIZE {
        stats.oversized_files_skipped += 1;
        return;
    }
    if let Ok(bytes) = backend.read(path).await
        && let Some(found) = search_content(&bytes, content_re, shape)
    {
        results.push(found.at(path.clone()));
    }
}

#[derive(Default)]
struct SearchStats {
    files_scanned: usize,
    oversized_files_skipped: usize,
    /// Directory entries visited during the walk (files + dirs, matched or not).
    entries_visited: usize,
    /// True when the walk stopped early (entry or wall-clock budget) — the
    /// "no matches" result is then partial, not definitive.
    truncated: bool,
}

/// Fast whole-machine lookup hint. Crawling the entire disk file-by-file is
/// slow and permission-fenced; the OS file index answers in milliseconds.
fn whole_machine_search_hint(shell: &str) -> String {
    #[cfg(target_os = "macos")]
    {
        format!(
            "For a whole-machine filename search, use the {shell} tool with Spotlight instead of crawling: mdfind -name \"<filename>\" (instant, indexed)."
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        format!(
            "For a whole-machine filename search, use the {shell} tool with the file index if available: locate \"<filename>\" (or plocate), falling back to find / -name."
        )
    }
}

/// Honest coverage note for a budget-truncated walk. Without it, "no matches"
/// reads as definitive and the model stops looking.
fn truncation_note(stats: &SearchStats) -> String {
    format!(
        "Note: the search stopped early after visiting {} directory entries — coverage was partial, NOT exhaustive. The file may still exist. Retry with a narrower 'path' (e.g. ~/Downloads, ~/Documents) for a complete scan of that directory.",
        stats.entries_visited
    )
}

impl SearchResult {
    fn format(&self) -> String {
        let path_str = self.path.display();
        if self.lines.is_empty() {
            path_str.to_string()
        } else {
            // grep's convention: ':' after a matching line's number, '-'
            // after a context line's, "--" between separate blocks.
            let mut s = format!("{}:", path_str);
            let mut previous: Option<usize> = None;
            for (line_num, line, matched) in &self.lines {
                // Blocks are separated only when context lines make them
                // blocks; plain matches need no separator.
                let has_context = self.lines.iter().any(|line| !line.2);
                if has_context && previous.is_some_and(|previous| *line_num > previous + 1) {
                    s.push_str("\n  --");
                }
                let truncated = crate::text::truncate_str(line, 203);
                let mark = if *matched { ':' } else { '-' };
                s.push_str(&format!("\n  {line_num:>4}{mark} {truncated}"));
                previous = Some(*line_num);
            }
            if self.more > 0 {
                s.push_str(&format!(
                    "\n  (+{} more match{} in this file; raise max_per_file to see them)",
                    self.more,
                    if self.more == 1 { "" } else { "es" }
                ));
            }
            s
        }
    }
}

fn compile_glob(glob: &str) -> anyhow::Result<GlobMatcher> {
    Glob::new(glob)
        .map(|glob| glob.compile_matcher())
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", glob, e))
}

/// Wall-clock budget for one search walk. Large trees (a whole home
/// directory) must return a partial-but-fast answer, never hang until the
/// tool watchdog kills the call.
const MAX_WALK_MILLIS: u128 = 15_000;

/// Run one filesystem call inside the walk's time budget. `None`: the budget
/// ran out first. A directory can hang (a stalled mount, an OS permission
/// prompt), and the check between entries never runs while it does.
async fn within_budget<T>(
    started: std::time::Instant,
    call: impl std::future::Future<Output = T>,
) -> Option<T> {
    let budget = std::time::Duration::from_millis(MAX_WALK_MILLIS as u64);
    tokio::time::timeout(budget.saturating_sub(started.elapsed()), call)
        .await
        .ok()
}

/// Breadth-first, budgeted directory walk.
///
/// BFS matters: a file in a shallow user directory (~/Downloads/x.pdf) is
/// found within the first few thousand entries even when a sibling tree
/// (~/Library) holds hundreds of thousands — depth-first got lost in the
/// deep tree and timed out. `max_entries` bounds total entries visited;
/// exceeding it (or the time budget) sets `stats.truncated`.
#[allow(clippy::too_many_arguments)]
async fn walk_dir(
    backend: SharedBackend,
    policy: &dyn FilePolicy,
    root: &BackendPath,
    content_regex: &Option<Regex>,
    glob_matcher: &Option<GlobMatcher>,
    max_results: usize,
    max_entries: usize,
    collaboration_root: Option<&BackendPath>,
    shape: MatchShape,
    results: &mut Vec<SearchResult>,
    stats: &mut SearchStats,
) {
    let started = std::time::Instant::now();
    let mut queue: std::collections::VecDeque<(BackendPath, usize)> =
        std::collections::VecDeque::new();
    queue.push_back((root.clone(), 0));

    while let Some((dir, depth)) = queue.pop_front() {
        if results.len() >= max_results || stats.files_scanned >= MAX_FILES_SCANNED {
            return;
        }

        let Some(listed) = within_budget(started, backend.read_dir(&dir)).await else {
            stats.truncated = true;
            return;
        };
        let entries = match listed {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries {
            if results.len() >= max_results || stats.files_scanned >= MAX_FILES_SCANNED {
                return;
            }
            stats.entries_visited += 1;
            if stats.entries_visited >= max_entries
                || started.elapsed().as_millis() > MAX_WALK_MILLIS
            {
                stats.truncated = true;
                return;
            }

            let mut path = entry.path;
            if let Some(root) = collaboration_root {
                if policy.is_sensitive(std::path::Path::new(path.as_str())) {
                    continue;
                }
                let Ok(canonical) = backend.canonicalize(&path).await else {
                    continue;
                };
                if !std::path::Path::new(canonical.as_str())
                    .starts_with(std::path::Path::new(root.as_str()))
                    || policy.is_sensitive(std::path::Path::new(canonical.as_str()))
                {
                    continue;
                }
                path = canonical;
            }
            let file_name = path.file_name().unwrap_or_default().to_string();

            if entry.metadata.file_type == BackendFileType::Directory {
                if depth < MAX_DEPTH
                    && !DEFAULT_IGNORE_DIRS.contains(&file_name.as_str())
                    && !file_name.starts_with('.')
                {
                    queue.push_back((path, depth + 1));
                }
                continue;
            }

            if entry.metadata.file_type != BackendFileType::File {
                continue;
            }

            // Check glob match on filename
            if let Some(matcher) = glob_matcher
                && !matcher.is_match(&file_name)
            {
                continue;
            }

            stats.files_scanned += 1;

            // If content pattern specified, search contents
            if let Some(content_re) = content_regex {
                if entry.metadata.len > MAX_CONTENT_SEARCH_FILE_SIZE {
                    stats.oversized_files_skipped += 1;
                    continue;
                }
                let Some(read) = within_budget(started, backend.read(&path)).await else {
                    stats.truncated = true;
                    return;
                };
                if let Ok(bytes) = read
                    && let Some(found) = search_content(&bytes, content_re, shape)
                {
                    results.push(found.at(path));
                }
            } else {
                // Glob-only: just list the file
                results.push(SearchResult {
                    path,
                    lines: Vec::new(),
                    more: 0,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::{WorkspaceBackend, WorkspaceFilePolicy};

    #[tokio::test(start_paused = true)]
    async fn a_call_that_never_answers_ends_with_the_budget() {
        let started = std::time::Instant::now();
        let hung = within_budget(started, std::future::pending::<()>()).await;
        assert!(hung.is_none());
        let answered = within_budget(started, std::future::ready(7)).await;
        assert_eq!(answered, Some(7));
    }

    fn parts(path: &std::path::Path) -> (SharedBackend, WorkspaceFilePolicy) {
        let workspace = Workspace::new(path).unwrap();
        (
            Arc::new(WorkspaceBackend::new(workspace.clone())),
            WorkspaceFilePolicy::new(workspace),
        )
    }

    #[tokio::test]
    async fn breadth_first_walk_finds_shallow_file_under_tight_budget() {
        // A file two levels down (~/Downloads/x.pdf shape) must be found
        // before the walk exhausts its budget inside a deep sibling tree —
        // the failure mode that made home-directory searches time out.
        let root = tempfile::tempdir().unwrap();
        let deep = root.path().join("aaa_huge").join("lvl1");
        std::fs::create_dir_all(&deep).unwrap();
        for i in 0..300 {
            std::fs::write(deep.join(format!("noise_{i}.txt")), "x").unwrap();
        }
        let shallow = root.path().join("bbb_docs");
        std::fs::create_dir(&shallow).unwrap();
        std::fs::write(shallow.join("target.pdf"), "pdf").unwrap();

        let glob = compile_glob("*target.pdf").unwrap();
        let mut results = Vec::new();
        let mut stats = SearchStats::default();
        let (backend, policy) = parts(root.path());
        let root_path = backend
            .resolve_path(&root.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &policy,
            &root_path,
            &None,
            &Some(glob),
            10,
            50, // entry budget far smaller than the deep tree
            None,
            MatchShape {
                per_file: DEFAULT_MATCHES_PER_FILE,
                context: 0,
            },
            &mut results,
            &mut stats,
        )
        .await;

        assert_eq!(results.len(), 1, "shallow file must be found under budget");
        assert!(results[0].path.as_str().ends_with("target.pdf"));
    }

    #[tokio::test]
    async fn search_files_finds_unrelated_project_config_toml() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.toml"), "title = \"a\"").unwrap();
        let glob = compile_glob("*config.toml").unwrap();
        let mut results = Vec::new();
        let mut stats = SearchStats::default();
        let (backend, policy) = parts(root.path());
        let root_path = backend
            .resolve_path(&root.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &policy,
            &root_path,
            &None,
            &Some(glob),
            10,
            50,
            None,
            MatchShape {
                per_file: DEFAULT_MATCHES_PER_FILE,
                context: 0,
            },
            &mut results,
            &mut stats,
        )
        .await;
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn walk_truncates_at_entry_budget_and_reports_it() {
        let root = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(root.path().join(format!("file_{i}.txt")), "x").unwrap();
        }

        let glob = compile_glob("*.zzz").unwrap();
        let mut results = Vec::new();
        let mut stats = SearchStats::default();
        let (backend, policy) = parts(root.path());
        let root_path = backend
            .resolve_path(&root.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &policy,
            &root_path,
            &None,
            &Some(glob),
            10,
            10,
            None,
            MatchShape {
                per_file: DEFAULT_MATCHES_PER_FILE,
                context: 0,
            },
            &mut results,
            &mut stats,
        )
        .await;

        assert!(stats.truncated, "budget exhaustion must set truncated");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn truncated_search_output_carries_honest_note() {
        // The output must tell the model coverage was partial — otherwise
        // "no matches" reads as a definitive answer and it gives up.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..30 {
            std::fs::write(dir.path().join(format!("file_{i}.txt")), "x").unwrap();
        }
        // max_results=1 path is fine; we exercise the public call with a
        // glob that cannot match, against a tree we know exceeds the test
        // budget via the internal walker — so use walk_dir + render check
        // indirectly: the call() budget is large, so instead assert the
        // note constant is wired by checking a tiny-budget walk + format.
        let glob = compile_glob("*.zzz").unwrap();
        let mut results = Vec::new();
        let mut stats = SearchStats::default();
        let (backend, policy) = parts(dir.path());
        let root_path = backend
            .resolve_path(&dir.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &policy,
            &root_path,
            &None,
            &Some(glob),
            10,
            5,
            None,
            MatchShape {
                per_file: DEFAULT_MATCHES_PER_FILE,
                context: 0,
            },
            &mut results,
            &mut stats,
        )
        .await;
        assert!(stats.truncated);
        let note = truncation_note(&stats);
        assert!(note.contains("partial"), "note: {note}");
        assert!(
            note.contains("path"),
            "note must steer toward narrower path"
        );
    }

    #[test]
    fn test_glob_to_regex() {
        let re = compile_glob("*.rs").unwrap();
        assert!(re.is_match("main.rs"));
        assert!(!re.is_match("main.py"));

        let re = compile_glob("Cargo.*").unwrap();
        assert!(re.is_match("Cargo.toml"));
        assert!(re.is_match("Cargo.lock"));
    }

    #[test]
    fn glob_matches_literal_regex_metacharacters_and_commas() {
        let re = compile_glob("a+b.txt").unwrap();
        assert!(re.is_match("a+b.txt"));
        assert!(!re.is_match("aaab.txt"));

        let re = compile_glob("report,final.txt").unwrap();
        assert!(re.is_match("report,final.txt"));
        assert!(!re.is_match("report-draft.txt"));
        assert!(!re.is_match("final.txt"));
    }

    /// The tool over a temp folder holding a synthetic source file.
    async fn search(args: Value) -> String {
        let root = tempfile::tempdir().unwrap();
        let mut text = String::new();
        for i in 1..=30 {
            if i % 3 == 0 {
                text.push_str(&format!("let Value_{i} = compute({i});\n"));
            } else {
                text.push_str(&format!("// plain line {i}\n"));
            }
        }
        std::fs::write(root.path().join("lib.rs"), &text).unwrap();
        std::fs::write(root.path().join("other.rs"), "nothing here\n").unwrap();
        let tool = SearchFiles::new(Workspace::new(root.path()).unwrap());
        let mut args = args;
        if let Some(name) = args.get("path").and_then(Value::as_str).map(str::to_string) {
            args["path"] = json!(root.path().join(name).to_string_lossy());
        } else {
            args["path"] = json!(root.path().to_string_lossy());
        }
        tool.run(&args.to_string(), &ToolCallContext::default())
            .await
            .unwrap()
    }

    /// A file path searches that file (it used to fail "Not a directory").
    #[tokio::test]
    async fn a_file_path_searches_that_file() {
        let out = search(json!({"pattern": "compute", "path": "lib.rs"})).await;
        assert!(out.contains("lib.rs:"), "{out}");
        assert!(out.contains("   3: let Value_3"), "{out}");
        assert!(!out.contains("other.rs"), "{out}");
    }

    /// Matches past the per-file limit are counted, and `max_per_file`
    /// shows more.
    #[tokio::test]
    async fn matches_past_the_limit_are_counted() {
        let out = search(json!({"pattern": "compute"})).await;
        assert!(out.contains("  15: let Value_15"), "{out}");
        assert!(!out.contains("Value_18"), "{out}");
        assert!(out.contains("(+5 more matches in this file"), "{out}");
        let out = search(json!({"pattern": "compute", "max_per_file": 50})).await;
        assert!(
            out.contains("Value_30") && !out.contains("more match"),
            "{out}"
        );
    }

    /// Context lines come before and after each match, marked like grep.
    #[tokio::test]
    async fn context_lines_surround_each_match() {
        let out = search(json!({"pattern": "Value_3 ", "context_lines": 1})).await;
        assert!(out.contains("   2- // plain line 2"), "{out}");
        assert!(out.contains("   3: let Value_3 "), "{out}");
        assert!(out.contains("   4- // plain line 4"), "{out}");
    }

    /// The pattern can ignore letter case.
    #[tokio::test]
    async fn the_pattern_can_ignore_case() {
        let out = search(json!({"pattern": "value_6"})).await;
        assert!(out.starts_with("No matches"), "{out}");
        let out = search(json!({"pattern": "value_6", "case_insensitive": true})).await;
        assert!(out.contains("   6: let Value_6"), "{out}");
    }
}
