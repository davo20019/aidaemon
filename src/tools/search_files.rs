use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use regex::Regex;
use serde_json::{json, Value};

use crate::execution::{
    active_execution_backend, BackendFileType, BackendPath, SharedExecutionBackend,
};
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolRole, ToolTargetHintKind, ToolVerificationMode,
};

use super::fs_utils;

pub struct SearchFilesTool;

const MAX_RESULTS: usize = 200;
const DEFAULT_MAX_RESULTS: usize = 50;
const MAX_FILES_SCANNED: usize = 10_000;
const MAX_DEPTH: usize = 20;
/// Total directory entries one walk may visit before returning a partial
/// result. Keeps whole-home searches fast instead of watchdog-fatal.
const MAX_ENTRIES_VISITED: usize = 150_000;
const MAX_CONTENT_SEARCH_FILE_SIZE: u64 = 1024 * 1024; // 1 MiB per file

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search for files by name pattern or content regex"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_files",
            "description": "Search for files by name pattern (glob) and/or content (regex). Use this instead of terminal find/grep. Automatically skips .git, node_modules, target, etc.",
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
                        "description": "Directory to search in (default: current directory)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 50, max: 200)"
                    }
                },
                "additionalProperties": false
            }
        })
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
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

        if content_pattern.is_none() && glob_pattern.is_none() {
            anyhow::bail!("At least one of 'pattern' (content regex) or 'glob' (filename pattern) is required");
        }

        let backend = active_execution_backend();
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
            if trimmed != path_str && !trimmed.is_empty() {
                if let Ok(candidate) = backend.resolve_path(trimmed).await {
                    if backend
                        .metadata(&candidate)
                        .await
                        .is_ok_and(|metadata| metadata.is_dir())
                    {
                        search_dir = candidate;
                    }
                }
            }
        }
        let search_metadata = backend
            .metadata(&search_dir)
            .await
            .map_err(|_| anyhow::anyhow!("Directory not found: {}", search_dir))?;
        if !search_metadata.is_dir() {
            anyhow::bail!("Not a directory: {}", search_dir);
        }

        let content_regex = if let Some(pat) = content_pattern {
            Some(Regex::new(pat).map_err(|e| anyhow::anyhow!("Invalid regex '{}': {}", pat, e))?)
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

        walk_dir(
            backend,
            &search_dir,
            &content_regex,
            &glob_matcher,
            max_results,
            MAX_ENTRIES_VISITED,
            &mut results,
            &mut stats,
        )
        .await;

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
                output.push_str(whole_machine_search_hint());
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
        output.push_str(&format!(
            "Found {} match{}. {} files scanned in {}\n\n",
            results.len(),
            if results.len() == 1 { "" } else { "es" },
            stats.files_scanned,
            search_dir
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
    matches: Vec<(usize, String)>, // (line_number, line_content)
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
fn whole_machine_search_hint() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "For a whole-machine filename search, use the terminal tool with Spotlight instead of crawling: mdfind -name \"<filename>\" (instant, indexed)."
    }
    #[cfg(not(target_os = "macos"))]
    {
        "For a whole-machine filename search, use the terminal tool with the file index if available: locate \"<filename>\" (or plocate), falling back to find / -name."
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
        if self.matches.is_empty() {
            path_str.to_string()
        } else {
            let mut s = format!("{}:", path_str);
            for (line_num, line) in &self.matches {
                let truncated = crate::utils::truncate_str(line, 203);
                s.push_str(&format!("\n  {:>4}: {}", line_num, truncated));
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

/// Breadth-first, budgeted directory walk.
///
/// BFS matters: a file in a shallow user directory (~/Downloads/x.pdf) is
/// found within the first few thousand entries even when a sibling tree
/// (~/Library) holds hundreds of thousands — depth-first got lost in the
/// deep tree and timed out. `max_entries` bounds total entries visited;
/// exceeding it (or the time budget) sets `stats.truncated`.
#[allow(clippy::too_many_arguments)]
async fn walk_dir(
    backend: SharedExecutionBackend,
    root: &BackendPath,
    content_regex: &Option<Regex>,
    glob_matcher: &Option<GlobMatcher>,
    max_results: usize,
    max_entries: usize,
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

        let entries = match backend.read_dir(&dir).await {
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

            let path = entry.path;
            let file_name = path.file_name().unwrap_or_default().to_string();

            if entry.metadata.file_type == BackendFileType::Directory {
                if depth < MAX_DEPTH
                    && !fs_utils::should_skip_dir(&file_name)
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
            if let Some(ref matcher) = glob_matcher {
                if !matcher.is_match(&file_name) {
                    continue;
                }
            }

            stats.files_scanned += 1;

            // If content pattern specified, search contents
            if let Some(ref content_re) = content_regex {
                if entry.metadata.len > MAX_CONTENT_SEARCH_FILE_SIZE {
                    stats.oversized_files_skipped += 1;
                    continue;
                }
                if let Ok(bytes) = backend.read(&path).await {
                    let content = String::from_utf8_lossy(&bytes);
                    let mut matches = Vec::new();
                    for (i, line) in content.lines().enumerate() {
                        if content_re.is_match(line) {
                            matches.push((i + 1, line.to_string()));
                            if matches.len() >= 5 {
                                break; // Cap matches per file
                            }
                        }
                    }
                    if !matches.is_empty() {
                        results.push(SearchResult { path, matches });
                    }
                }
            } else {
                // Glob-only: just list the file
                results.push(SearchResult {
                    path,
                    matches: vec![],
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let backend = active_execution_backend();
        let root_path = backend
            .resolve_path(&root.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &root_path,
            &None,
            &Some(glob),
            10,
            50, // entry budget far smaller than the deep tree
            &mut results,
            &mut stats,
        )
        .await;

        assert_eq!(results.len(), 1, "shallow file must be found under budget");
        assert!(results[0].path.as_str().ends_with("target.pdf"));
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
        let backend = active_execution_backend();
        let root_path = backend
            .resolve_path(&root.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &root_path,
            &None,
            &Some(glob),
            10,
            10,
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
        let backend = active_execution_backend();
        let root_path = backend
            .resolve_path(&dir.path().to_string_lossy())
            .await
            .unwrap();
        walk_dir(
            backend,
            &root_path,
            &None,
            &Some(glob),
            10,
            5,
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

    #[tokio::test]
    async fn default_path_no_match_suggests_user_directories() {
        // No 'path' arg → tool defaults to cwd. When nothing matches a
        // filename-style glob, the output must point the model at common
        // user locations instead of dead-ending.
        let args = serde_json::json!({
            "glob": "*zz-definitely-not-present-xyz (1).pdf"
        })
        .to_string();
        let result = SearchFilesTool.call(&args).await.unwrap();

        assert!(result.contains("No matches found"));
        assert!(
            result.contains("~/Downloads"),
            "no-match default-path output must suggest user dirs, got: {result}"
        );
        assert!(result.contains("'path'"));
    }

    #[tokio::test]
    async fn explicit_path_no_match_has_no_user_dir_hint() {
        let dir = tempfile::tempdir().unwrap();
        let args = serde_json::json!({
            "glob": "*zz-definitely-not-present-xyz.pdf",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();
        let result = SearchFilesTool.call(&args).await.unwrap();

        assert!(result.contains("No matches found"));
        assert!(!result.contains("~/Downloads"));
    }

    #[tokio::test]
    async fn scanned_dir_is_never_glued_to_closing_punctuation() {
        // Models copy-paste the scanned dir from tool output into their next
        // call. "(N files scanned in /path)" produced paths like
        // "/Users/x/projects)" — the dir must end its line cleanly.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hit.rs"), "fn main() {}").unwrap();
        let dir_str = dir.path().to_str().unwrap();

        for glob in ["*zz-no-match-xyz.pdf", "*.rs"] {
            let args = serde_json::json!({ "glob": glob, "path": dir_str }).to_string();
            let result = SearchFilesTool.call(&args).await.unwrap();
            for bad_suffix in [")", "):"] {
                assert!(
                    !result.contains(&format!("{}{}", dir_str, bad_suffix)),
                    "scanned dir must not be followed by {:?}, got: {}",
                    bad_suffix,
                    result
                );
            }
            assert!(
                result.contains("files scanned in "),
                "scanned-dir marker must survive for project_dir extraction, got: {}",
                result
            );
        }
    }

    #[tokio::test]
    async fn trailing_punctuation_on_missing_path_is_recovered() {
        // A path copy-pasted from prose/old output may carry trailing ')' or
        // ':'. When the literal path doesn't exist but the trimmed one does,
        // search there instead of failing.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hit.rs"), "fn main() {}").unwrap();

        for suffix in [")", "):", ","] {
            let args = serde_json::json!({
                "glob": "*.rs",
                "path": format!("{}{}", dir.path().to_str().unwrap(), suffix)
            })
            .to_string();
            let result = SearchFilesTool.call(&args).await.unwrap_or_else(|e| {
                panic!(
                    "path with trailing {:?} must recover, got error: {}",
                    suffix, e
                )
            });
            assert!(result.contains("hit.rs"), "got: {}", result);
        }
    }

    #[tokio::test]
    async fn trailing_paren_path_that_exists_is_not_trimmed() {
        // Directories with a literal trailing paren (e.g. "Folder (1)") must
        // still be searched as-is.
        let dir = tempfile::tempdir().unwrap();
        let paren_dir = dir.path().join("Copy (1)");
        std::fs::create_dir(&paren_dir).unwrap();
        std::fs::write(paren_dir.join("hit.rs"), "fn main() {}").unwrap();

        let args = serde_json::json!({
            "glob": "*.rs",
            "path": paren_dir.to_str().unwrap()
        })
        .to_string();
        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("hit.rs"), "got: {}", result);
    }

    #[test]
    fn test_schema_has_required_fields() {
        let tool = SearchFilesTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "search_files");
        assert!(!schema["description"].as_str().unwrap().is_empty());
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

    #[tokio::test]
    async fn test_search_by_glob() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("test.py"), "def main(): pass").unwrap();

        let args = json!({
            "glob": "*.rs",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("test.rs"));
        assert!(!result.contains("test.py"));
    }

    #[tokio::test]
    async fn test_search_by_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world\nfoo bar\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "goodbye world\n").unwrap();

        let args = json!({
            "pattern": "hello",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("a.txt"));
        assert!(result.contains("hello world"));
        assert!(!result.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_content_search_skips_oversized_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut large = vec![b'a'; 1_048_577];
        large.extend_from_slice(b"\nneedle_in_large_file\n");
        std::fs::write(dir.path().join("large.log"), large).unwrap();

        let args = json!({
            "pattern": "needle_in_large_file",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(!result.contains("needle_in_large_file"));
        assert!(result.contains("Skipped 1 oversized file"));
    }

    #[tokio::test]
    async fn test_search_no_results() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();

        let args = json!({
            "pattern": "nonexistent_pattern_xyz",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("No matches"));
        assert!(result.contains(dir.path().to_str().unwrap()));
    }

    #[tokio::test]
    async fn test_search_requires_pattern_or_glob() {
        let args = json!({"path": "/tmp"}).to_string();
        let result = SearchFilesTool.call(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_skips_ignored_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let node_modules = dir.path().join("node_modules");
        std::fs::create_dir(&node_modules).unwrap();
        std::fs::write(node_modules.join("hidden.js"), "should not find").unwrap();
        std::fs::write(dir.path().join("visible.js"), "should find").unwrap();

        let args = json!({
            "glob": "*.js",
            "path": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("visible.js"));
        assert!(!result.contains("hidden.js"));
    }

    #[tokio::test]
    async fn test_search_warns_when_path_omitted() {
        let args = json!({
            "glob": "*",
            "max_results": 1
        })
        .to_string();

        let result = SearchFilesTool.call(&args).await.unwrap();
        assert!(result.contains("defaulted to current directory"));
    }
}
