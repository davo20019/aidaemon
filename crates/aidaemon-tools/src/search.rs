//! `glob` and `grep`. Both walk the workspace with the `ignore` crate, so
//! `.gitignore`, `.ignore` and hidden files are skipped like `ripgrep` does.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use serde::Deserialize;
use serde_json::{Value, json};

use aidaemon_core::{
    AccessManifest, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};

use crate::util::{fail, looks_binary, parse_args, truncate_chars};
use crate::workspace::Workspace;

/// Default cap on paths returned by `glob` and `grep` (files mode).
pub const DEFAULT_MAX_RESULTS: usize = 200;
/// Default cap on matching lines returned by `grep` (content mode).
pub const DEFAULT_MAX_LINES: usize = 200;
/// Matching lines are cut to this many characters.
const MAX_MATCH_LINE_CHARS: usize = 300;
/// Files larger than this are skipped by `grep`.
const MAX_GREP_FILE_BYTES: u64 = 5 * 1024 * 1024;

fn walker(base: &Path) -> ignore::Walk {
    ignore::WalkBuilder::new(base)
        .hidden(true)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .require_git(false)
        .parents(true)
        .follow_links(false)
        .build()
}

fn resolve_base(ws: &Workspace, path: Option<&str>) -> Result<PathBuf, String> {
    let base = match path {
        Some(p) if !p.trim().is_empty() => ws.resolve(p).map_err(|e| e.to_string())?,
        _ => ws.root().to_path_buf(),
    };
    if !base.exists() {
        return Err(format!("`{}` does not exist", ws.display(&base)));
    }
    Ok(base)
}

fn declare_base(ws: &Workspace, arguments: &Value) -> Vec<String> {
    let path = arguments
        .get("path")
        .and_then(Value::as_str)
        .filter(|p| !p.trim().is_empty());
    vec![match path {
        Some(p) => ws.declare(p),
        None => ws.root().display().to_string(),
    }]
}

// --------------------------------------------------------------------- glob

/// Find files by glob pattern, respecting `.gitignore`.
#[derive(Debug, Clone)]
pub struct Glob {
    workspace: Workspace,
    max_results: usize,
}

impl Glob {
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            max_results: DEFAULT_MAX_RESULTS,
        }
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max.max(1);
        self
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobArgs {
    pattern: String,
    path: Option<String>,
}

fn build_glob(pattern: &str, literal_separator: bool) -> Result<GlobMatcher, String> {
    GlobBuilder::new(pattern)
        .literal_separator(literal_separator)
        .build()
        .map(|g| g.compile_matcher())
        .map_err(|e| format!("invalid glob pattern `{pattern}`: {e}"))
}

#[async_trait]
impl Tool for Glob {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob".into(),
            description: format!(
                "Find files whose path matches a glob pattern, relative to `path` (default: the \
                 workspace root). `*` does not cross directories; use `**` for any depth, for \
                 example `**/*.rs` or `src/**/mod.rs`. Files ignored by .gitignore and hidden \
                 files are skipped. Returns sorted paths relative to the workspace root, at most \
                 {} of them.",
                self.max_results
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, for example `**/*.toml`."},
                    "path": {"type": "string", "description": "Directory to search from, relative to the workspace root. Default: the root."}
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            reads: declare_base(&self.workspace, arguments),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: GlobArgs = match parse_args("glob", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let base = match resolve_base(&self.workspace, args.path.as_deref()) {
            Ok(b) => b,
            Err(e) => return Ok(fail(e)),
        };
        let pattern = args.pattern.trim_start_matches("./").to_string();
        let matcher = match build_glob(&pattern, true) {
            Ok(m) => m,
            Err(e) => return Ok(fail(e)),
        };
        let ws = self.workspace.clone();
        let max = self.max_results;
        let result = tokio::task::spawn_blocking(move || {
            let mut found: Vec<String> = Vec::new();
            for entry in walker(&base).flatten() {
                if !entry.file_type().is_some_and(|t| t.is_file()) {
                    continue;
                }
                let rel = entry.path().strip_prefix(&base).unwrap_or(entry.path());
                if matcher.is_match(rel) {
                    found.push(ws.display(entry.path()));
                }
            }
            found.sort();
            found
        })
        .await
        .map_err(|e| format!("glob task failed: {e}"))?;

        let total = result.len();
        if total == 0 {
            return Ok(ToolOutput::text(format!(
                "No files match `{pattern}`. Patterns are relative to the search directory; \
                 try `**/{pattern}` to search all depths."
            ))
            .with_details(json!({"matches": 0})));
        }
        let mut out = result
            .iter()
            .take(max)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if total > max {
            out.push_str(&format!(
                "\n[Showing {max} of {total} matches. Narrow the pattern or the path to see the rest.]"
            ));
        }
        Ok(ToolOutput::text(out).with_details(json!({
            "matches": total,
            "truncated": total > max,
        })))
    }
}

// --------------------------------------------------------------------- grep

/// Search file contents with a regular expression.
#[derive(Debug, Clone)]
pub struct Grep {
    workspace: Workspace,
    max_results: usize,
    max_lines: usize,
}

impl Grep {
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            max_results: DEFAULT_MAX_RESULTS,
            max_lines: DEFAULT_MAX_LINES,
        }
    }

    pub fn with_limits(mut self, max_files: usize, max_lines: usize) -> Self {
        self.max_results = max_files.max(1);
        self.max_lines = max_lines.max(1);
        self
    }
}

#[derive(Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GrepMode {
    #[default]
    Files,
    Content,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrepArgs {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    output_mode: GrepMode,
}

#[async_trait]
impl Tool for Grep {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: format!(
                "Search file contents with a regular expression (Rust regex syntax, like \
                 ripgrep). `output_mode` `files` (default) lists matching files; `content` \
                 shows matching lines as `path:line:text`. Filter files with `glob` (matched \
                 against the path, so `*.rs` matches at any depth). Skips .gitignored, hidden \
                 and binary files. Returns at most {} files or {} lines.",
                self.max_results, self.max_lines
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "File or directory to search, relative to the workspace root. Default: the root."},
                    "glob": {"type": "string", "description": "Only search files matching this glob, for example `*.py` or `src/**/*.ts`."},
                    "case_insensitive": {"type": "boolean", "description": "Ignore case. Default false."},
                    "output_mode": {"type": "string", "enum": ["files", "content"], "description": "`files` (default) or `content`."}
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            reads: declare_base(&self.workspace, arguments),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: GrepArgs = match parse_args("grep", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let regex = match regex::RegexBuilder::new(&args.pattern)
            .case_insensitive(args.case_insensitive)
            .size_limit(10 * 1024 * 1024)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                return Ok(fail(format!(
                    "invalid regex `{}`: {e}. Escape special characters such as `(`, `[` or `{{` with a backslash.",
                    args.pattern
                )));
            }
        };
        let filter = match args.glob.as_deref().filter(|g| !g.trim().is_empty()) {
            Some(g) => match build_glob(g.trim_start_matches("./"), false) {
                Ok(m) => Some(m),
                Err(e) => return Ok(fail(e)),
            },
            None => None,
        };
        let base = match resolve_base(&self.workspace, args.path.as_deref()) {
            Ok(b) => b,
            Err(e) => return Ok(fail(e)),
        };
        let ws = self.workspace.clone();
        let mode = args.output_mode;
        let (max_files, max_lines) = (self.max_results, self.max_lines);

        let (lines, files, total_lines) = tokio::task::spawn_blocking(move || {
            let mut paths: Vec<PathBuf> = walker(&base)
                .flatten()
                .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
                .map(|e| e.into_path())
                .filter(|p| {
                    let rel = p.strip_prefix(&base).unwrap_or(p);
                    let rel = if rel.as_os_str().is_empty() {
                        Path::new(p.file_name().unwrap_or_default())
                    } else {
                        rel
                    };
                    filter.as_ref().is_none_or(|m| m.is_match(rel))
                })
                .collect();
            paths.sort();

            let mut lines: Vec<String> = Vec::new();
            let mut files: Vec<String> = Vec::new();
            let mut total_lines = 0usize;
            for path in paths {
                if std::fs::metadata(&path).map_or(true, |m| m.len() > MAX_GREP_FILE_BYTES) {
                    continue;
                }
                let Ok(bytes) = std::fs::read(&path) else {
                    continue;
                };
                if looks_binary(&bytes) {
                    continue;
                }
                let text = String::from_utf8_lossy(&bytes);
                let shown = ws.display(&path);
                let mut matched = false;
                for (i, line) in text.lines().enumerate() {
                    if regex.is_match(line) {
                        matched = true;
                        if mode == GrepMode::Files {
                            break;
                        }
                        total_lines += 1;
                        if lines.len() < max_lines {
                            let (line, _) = truncate_chars(line, MAX_MATCH_LINE_CHARS);
                            lines.push(format!("{shown}:{}:{line}", i + 1));
                        }
                    }
                }
                if matched {
                    files.push(shown);
                }
            }
            (lines, files, total_lines)
        })
        .await
        .map_err(|e| format!("grep task failed: {e}"))?;

        if files.is_empty() {
            return Ok(
                ToolOutput::text(format!("No matches for `{}`.", args.pattern))
                    .with_details(json!({"files": 0, "lines": 0})),
            );
        }
        let out = match mode {
            GrepMode::Files => {
                let mut out = files
                    .iter()
                    .take(max_files)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
                if files.len() > max_files {
                    out.push_str(&format!(
                        "\n[Showing {max_files} of {} matching files. Narrow `path` or `glob`.]",
                        files.len()
                    ));
                }
                out
            }
            GrepMode::Content => {
                let mut out = lines.join("\n");
                if total_lines > lines.len() {
                    out.push_str(&format!(
                        "\n[Showing {} of {total_lines} matching lines in {} files. Narrow the pattern, `path` or `glob`.]",
                        lines.len(),
                        files.len()
                    ));
                }
                out
            }
        };
        Ok(ToolOutput::text(out).with_details(json!({
            "files": files.len(),
            "lines": total_lines,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{ctx, ws};

    fn tree(ws: &Workspace) {
        let r = ws.root();
        std::fs::create_dir_all(r.join("src/nested")).unwrap();
        std::fs::create_dir_all(r.join("target/debug")).unwrap();
        std::fs::write(r.join(".gitignore"), "target/\n*.log\n").unwrap();
        std::fs::write(r.join("src/lib.rs"), "pub fn alpha() {}\n// TODO: beta\n").unwrap();
        std::fs::write(r.join("src/nested/mod.rs"), "fn Alpha() {}\n").unwrap();
        std::fs::write(r.join("README.md"), "alpha docs\n").unwrap();
        std::fs::write(r.join("target/debug/gen.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(r.join("run.log"), "alpha log\n").unwrap();
        std::fs::write(r.join("img.bin"), b"alpha\0binary").unwrap();
    }

    #[tokio::test]
    async fn glob_respects_gitignore_and_sorts() {
        let (_d, ws) = ws();
        tree(&ws);
        let tool = Glob::new(ws.clone());
        let out = tool
            .call(json!({"pattern": "**/*.rs"}), &ctx())
            .await
            .unwrap();
        assert_eq!(out.content, "src/lib.rs\nsrc/nested/mod.rs");
        let out = tool.call(json!({"pattern": "*.rs"}), &ctx()).await.unwrap();
        assert!(out.content.starts_with("No files match"), "{}", out.content);
        let out = tool
            .call(json!({"pattern": "*.rs", "path": "src"}), &ctx())
            .await
            .unwrap();
        assert_eq!(out.content, "src/lib.rs");
        let out = tool.call(json!({"pattern": "*"}), &ctx()).await.unwrap();
        assert!(!out.content.contains("run.log") && out.content.contains("README.md"));
        let out = tool
            .call(json!({"pattern": "*", "path": "../"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn glob_caps_results() {
        let (_d, ws) = ws();
        for i in 0..12 {
            std::fs::write(ws.root().join(format!("f{i:02}.txt")), "x").unwrap();
        }
        let tool = Glob::new(ws).with_max_results(5);
        let out = tool
            .call(json!({"pattern": "*.txt"}), &ctx())
            .await
            .unwrap();
        assert_eq!(out.content.lines().count(), 6);
        assert!(out.content.contains("Showing 5 of 12 matches"));
        assert_eq!(out.details["truncated"], true);
    }

    #[tokio::test]
    async fn grep_files_and_content_modes() {
        let (_d, ws) = ws();
        tree(&ws);
        let tool = Grep::new(ws.clone());
        let out = tool
            .call(json!({"pattern": "alpha"}), &ctx())
            .await
            .unwrap();
        assert_eq!(out.content, "README.md\nsrc/lib.rs");

        let out = tool
            .call(
                json!({"pattern": "alpha", "case_insensitive": true, "glob": "*.rs"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "src/lib.rs\nsrc/nested/mod.rs");

        let out = tool
            .call(
                json!({"pattern": "TODO|alpha", "output_mode": "content", "path": "src"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_eq!(
            out.content,
            "src/lib.rs:1:pub fn alpha() {}\nsrc/lib.rs:2:// TODO: beta"
        );

        let out = tool
            .call(
                json!({"pattern": "alpha", "path": "src/lib.rs", "output_mode": "content"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "src/lib.rs:1:pub fn alpha() {}");

        let out = tool.call(json!({"pattern": "zzz"}), &ctx()).await.unwrap();
        assert!(!out.is_error && out.content.starts_with("No matches"));

        let out = tool.call(json!({"pattern": "fn("}), &ctx()).await.unwrap();
        assert!(out.is_error && out.content.contains("invalid regex"));

        let out = tool
            .call(json!({"pattern": "x", "output_mode": "lines"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn grep_caps_content_lines() {
        let (_d, ws) = ws();
        let body: String = (0..50).map(|i| format!("hit {i}\n")).collect();
        std::fs::write(ws.root().join("many.txt"), body).unwrap();
        let tool = Grep::new(ws).with_limits(10, 5);
        let out = tool
            .call(json!({"pattern": "hit", "output_mode": "content"}), &ctx())
            .await
            .unwrap();
        assert_eq!(out.content.lines().count(), 6);
        assert!(out.content.contains("Showing 5 of 50 matching lines"));
    }
}
