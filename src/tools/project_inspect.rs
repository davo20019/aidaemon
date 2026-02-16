use std::collections::HashSet;
use std::path::Path;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct ProjectInspectTool;
const MAX_BATCH_PATHS: usize = 12;

#[async_trait]
impl Tool for ProjectInspectTool {
    fn name(&self) -> &str {
        "project_inspect"
    }

    fn description(&self) -> &str {
        "Inspect a project directory to understand its type, structure, and metadata"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "project_inspect",
            "description": "Inspect one or more project directories to understand type, structure, dependencies, and git status. Prefer one batched call (`paths`) instead of many single calls when comparing multiple folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Single project directory path (default: current directory)"
                    },
                    "paths": {
                        "type": "array",
                        "description": "Multiple project directories to inspect in one call (max 12)",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": MAX_BATCH_PATHS
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let paths = parse_project_paths(&args)?;

        let mut reports = Vec::new();
        let mut failures = Vec::new();
        for path in &paths {
            match inspect_project(path).await {
                Ok(report) => reports.push(report),
                Err(err) => failures.push((path.clone(), err.to_string())),
            }
        }

        if reports.is_empty() {
            let details = failures
                .iter()
                .map(|(path, err)| format!("{}: {}", path, err))
                .collect::<Vec<_>>()
                .join("; ");
            anyhow::bail!(
                "project_inspect could not inspect any requested path. {}",
                details
            );
        }

        if reports.len() == 1 && failures.is_empty() {
            return Ok(reports.remove(0));
        }

        let mut output = String::new();
        output.push_str(&reports.join("\n\n---\n\n"));
        if !failures.is_empty() {
            output.push_str("\n\n## Skipped paths\n");
            for (path, err) in failures {
                output.push_str(&format!("- `{}`: {}\n", path, err));
            }
        }
        Ok(output)
    }
}

fn parse_project_paths(args: &Value) -> anyhow::Result<Vec<String>> {
    let mut requested = Vec::new();

    if let Some(path) = args.get("path").and_then(Value::as_str) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            requested.push(trimmed.to_string());
        }
    }

    if let Some(paths_value) = args.get("paths") {
        let Some(path_arr) = paths_value.as_array() else {
            anyhow::bail!("`paths` must be an array of directory paths");
        };
        for entry in path_arr {
            let Some(path) = entry.as_str() else {
                anyhow::bail!("Each item in `paths` must be a string");
            };
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                requested.push(trimmed.to_string());
            }
        }
    }

    if requested.is_empty() {
        requested.push(".".to_string());
    }

    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for path in requested {
        if seen.insert(path.clone()) {
            deduped.push(path);
        }
    }

    if deduped.len() > MAX_BATCH_PATHS {
        anyhow::bail!(
            "Too many paths for project_inspect: {} (max {})",
            deduped.len(),
            MAX_BATCH_PATHS
        );
    }

    Ok(deduped)
}

async fn inspect_project(path_str: &str) -> anyhow::Result<String> {
    let project_dir = fs_utils::validate_path(path_str)?;

    if !project_dir.exists() || !project_dir.is_dir() {
        anyhow::bail!("Not a valid directory: {}", path_str);
    }

    let mut output = String::new();
    output.push_str(&format!("# Project: {}\n\n", project_dir.display()));

    // Detect project type
    let project_type = detect_project_type(&project_dir).await;
    output.push_str(&format!("## Type: {}\n\n", project_type));

    // Read project metadata
    let metadata = read_project_metadata(&project_dir).await;
    if !metadata.is_empty() {
        output.push_str("## Metadata\n");
        output.push_str(&metadata);
        output.push('\n');
    }

    // Git info
    let git_info = get_git_summary(&project_dir).await;
    if !git_info.is_empty() {
        output.push_str("## Git\n");
        output.push_str(&git_info);
        output.push('\n');
    }

    // Directory structure (top 2 levels)
    let structure = get_directory_structure(&project_dir, 0, 2).await;
    output.push_str("## Structure\n```\n");
    output.push_str(&structure);
    output.push_str("```\n");

    Ok(output)
}

async fn detect_project_type(dir: &Path) -> String {
    let checks = [
        ("Cargo.toml", "Rust (Cargo)"),
        ("package.json", "JavaScript/TypeScript (npm)"),
        ("pyproject.toml", "Python (pyproject)"),
        ("setup.py", "Python (setuptools)"),
        ("requirements.txt", "Python"),
        ("go.mod", "Go"),
        ("pom.xml", "Java (Maven)"),
        ("build.gradle", "Java/Kotlin (Gradle)"),
        ("build.gradle.kts", "Kotlin (Gradle)"),
        ("Gemfile", "Ruby (Bundler)"),
        ("composer.json", "PHP (Composer)"),
        ("CMakeLists.txt", "C/C++ (CMake)"),
        ("Makefile", "Make-based"),
        ("deno.json", "Deno"),
        ("bun.lockb", "Bun"),
        (".csproj", "C# (.NET)"),
        ("mix.exs", "Elixir (Mix)"),
        ("pubspec.yaml", "Dart/Flutter"),
        ("Package.swift", "Swift (SPM)"),
    ];

    let mut types = Vec::new();
    for (file, name) in checks {
        if dir.join(file).exists() {
            types.push(name.to_string());
        }
    }

    if types.is_empty() {
        "Unknown".to_string()
    } else {
        types.join(", ")
    }
}

async fn read_project_metadata(dir: &Path) -> String {
    let mut metadata = String::new();

    // Cargo.toml
    if let Ok(content) = tokio::fs::read_to_string(dir.join("Cargo.toml")).await {
        if let Some(name) = extract_toml_field(&content, "name") {
            metadata.push_str(&format!("- Name: {}\n", name));
        }
        if let Some(version) = extract_toml_field(&content, "version") {
            metadata.push_str(&format!("- Version: {}\n", version));
        }
        if let Some(edition) = extract_toml_field(&content, "edition") {
            metadata.push_str(&format!("- Rust edition: {}\n", edition));
        }
    }

    // package.json
    if let Ok(content) = tokio::fs::read_to_string(dir.join("package.json")).await {
        if let Ok(pkg) = serde_json::from_str::<Value>(&content) {
            if let Some(name) = pkg["name"].as_str() {
                metadata.push_str(&format!("- Name: {}\n", name));
            }
            if let Some(version) = pkg["version"].as_str() {
                metadata.push_str(&format!("- Version: {}\n", version));
            }
            if let Some(deps) = pkg["dependencies"].as_object() {
                metadata.push_str(&format!("- Dependencies: {}\n", deps.len()));
            }
            if let Some(dev_deps) = pkg["devDependencies"].as_object() {
                metadata.push_str(&format!("- Dev dependencies: {}\n", dev_deps.len()));
            }
        }
    }

    // pyproject.toml
    if let Ok(content) = tokio::fs::read_to_string(dir.join("pyproject.toml")).await {
        if let Some(name) = extract_toml_field(&content, "name") {
            metadata.push_str(&format!("- Name: {}\n", name));
        }
        if let Some(version) = extract_toml_field(&content, "version") {
            metadata.push_str(&format!("- Version: {}\n", version));
        }
    }

    // go.mod
    if let Ok(content) = tokio::fs::read_to_string(dir.join("go.mod")).await {
        if let Some(module) = content.lines().next() {
            if let Some(stripped) = module.strip_prefix("module ") {
                metadata.push_str(&format!("- Module: {}\n", stripped));
            }
        }
    }

    metadata
}

fn extract_toml_field(content: &str, field: &str) -> Option<String> {
    let pattern = format!("{} = \"", field);
    content.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.starts_with(&pattern) {
            let start = pattern.len();
            let value = &trimmed[start..];
            value.find('"').map(|end| value[..end].to_string())
        } else {
            None
        }
    })
}

async fn get_git_summary(dir: &Path) -> String {
    if !dir.join(".git").exists() {
        return String::new();
    }

    let mut info = String::new();

    // Current branch
    if let Ok(out) = fs_utils::run_cmd("git rev-parse --abbrev-ref HEAD", Some(dir), 5).await {
        if out.exit_code == 0 {
            info.push_str(&format!("- Branch: {}\n", out.stdout.trim()));
        }
    }

    // Status summary
    if let Ok(out) = fs_utils::run_cmd("git status --porcelain", Some(dir), 5).await {
        if out.exit_code == 0 {
            let lines: Vec<&str> = out.stdout.lines().collect();
            if lines.is_empty() {
                info.push_str("- Status: clean\n");
            } else {
                info.push_str(&format!("- Status: {} changed files\n", lines.len()));
            }
        }
    }

    // Last commit
    if let Ok(out) = fs_utils::run_cmd("git log -1 --format='%h %s (%cr)'", Some(dir), 5).await {
        if out.exit_code == 0 && !out.stdout.trim().is_empty() {
            info.push_str(&format!("- Last commit: {}\n", out.stdout.trim()));
        }
    }

    // Remote
    if let Ok(out) = fs_utils::run_cmd("git remote -v", Some(dir), 5).await {
        if out.exit_code == 0 {
            if let Some(first_line) = out.stdout.lines().next() {
                info.push_str(&format!("- Remote: {}\n", first_line));
            }
        }
    }

    info
}

fn get_directory_structure(
    dir: &Path,
    depth: usize,
    max_depth: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + '_>> {
    Box::pin(async move {
        if depth >= max_depth {
            return String::new();
        }

        let mut entries = match tokio::fs::read_dir(dir).await {
            Ok(e) => e,
            Err(_) => return String::new(),
        };

        let mut items: Vec<(String, bool)> = Vec::new(); // (name, is_dir)

        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') && depth == 0 && name != ".github" {
                continue;
            }
            if fs_utils::should_skip_dir(&name) {
                continue;
            }
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            items.push((name, is_dir));
        }

        items.sort_by(|a, b| {
            // Directories first, then alphabetical
            b.1.cmp(&a.1).then(a.0.cmp(&b.0))
        });

        let mut output = String::new();
        let indent = "  ".repeat(depth);

        for (name, is_dir) in &items {
            if *is_dir {
                output.push_str(&format!("{}{}/\n", indent, name));
                let subdir = dir.join(name);
                let sub = get_directory_structure(&subdir, depth + 1, max_depth).await;
                output.push_str(&sub);
            } else {
                output.push_str(&format!("{}{}\n", indent, name));
            }
        }

        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = ProjectInspectTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "project_inspect");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["paths"].is_object());
        assert_eq!(schema["parameters"]["properties"]["paths"]["maxItems"], 12);
    }

    #[test]
    fn test_parse_project_paths_defaults_and_dedupes() {
        let args = json!({
            "path": "",
            "paths": ["./a", "./a", "   ", "./b"]
        });
        let parsed = parse_project_paths(&args).unwrap();
        assert_eq!(parsed, vec!["./a".to_string(), "./b".to_string()]);

        let defaulted = parse_project_paths(&json!({})).unwrap();
        assert_eq!(defaulted, vec![".".to_string()]);
    }

    #[test]
    fn test_parse_project_paths_rejects_invalid_shapes() {
        let invalid_paths = parse_project_paths(&json!({"paths": "not-an-array"}));
        assert!(invalid_paths.is_err());

        let invalid_item = parse_project_paths(&json!({"paths": ["ok", 123]}));
        assert!(invalid_item.is_err());
    }

    #[tokio::test]
    async fn test_inspect_rust_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"test-proj\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").err(); // src doesn't exist yet
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = ProjectInspectTool.call(&args).await.unwrap();

        assert!(result.contains("Rust"));
        assert!(result.contains("test-proj"));
        assert!(result.contains("0.1.0"));
        assert!(result.contains("src/"));
    }

    #[tokio::test]
    async fn test_inspect_node_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("package.json"),
            r#"{"name": "my-app", "version": "1.0.0", "dependencies": {"react": "^18.0.0"}}"#,
        )
        .unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = ProjectInspectTool.call(&args).await.unwrap();

        assert!(result.contains("JavaScript"));
        assert!(result.contains("my-app"));
        assert!(result.contains("1.0.0"));
    }

    #[tokio::test]
    async fn test_inspect_nonexistent() {
        let args = json!({"path": "/tmp/nonexistent_project_12345"}).to_string();
        let result = ProjectInspectTool.call(&args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_inspect_multiple_projects_in_one_call() {
        let rust_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            rust_dir.path().join("Cargo.toml"),
            "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        let node_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            node_dir.path().join("package.json"),
            r#"{"name":"beta","version":"2.0.0"}"#,
        )
        .unwrap();

        let args = json!({
            "paths": [
                rust_dir.path().to_str().unwrap(),
                node_dir.path().to_str().unwrap()
            ]
        })
        .to_string();
        let result = ProjectInspectTool.call(&args).await.unwrap();

        assert_eq!(result.matches("# Project: ").count(), 2);
        assert!(result.contains("alpha"));
        assert!(result.contains("beta"));
    }

    #[tokio::test]
    async fn test_inspect_multiple_projects_reports_partial_failures() {
        let valid_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            valid_dir.path().join("package.json"),
            r#"{"name":"gamma","version":"1.0.0"}"#,
        )
        .unwrap();

        let args = json!({
            "paths": [
                valid_dir.path().to_str().unwrap(),
                "/tmp/nonexistent_project_12345"
            ]
        })
        .to_string();
        let result = ProjectInspectTool.call(&args).await.unwrap();

        assert!(result.contains("# Project: "));
        assert!(result.contains("## Skipped paths"));
        assert!(result.contains("nonexistent_project_12345"));
    }

    #[test]
    fn test_extract_toml_field() {
        let toml = "name = \"myproject\"\nversion = \"0.1.0\"\n";
        assert_eq!(
            extract_toml_field(toml, "name"),
            Some("myproject".to_string())
        );
        assert_eq!(
            extract_toml_field(toml, "version"),
            Some("0.1.0".to_string())
        );
        assert_eq!(extract_toml_field(toml, "edition"), None);
    }

    proptest! {
        #[test]
        fn parse_project_paths_invariants_for_string_inputs(
            maybe_path in ".*",
            include_path in any::<bool>(),
            raw_paths in prop::collection::vec(".*", 0..20),
        ) {
            let args = if include_path {
                json!({"path": maybe_path.clone(), "paths": raw_paths.clone()})
            } else {
                json!({"paths": raw_paths.clone()})
            };
            let result = parse_project_paths(&args);

            let mut unique_non_empty = HashSet::new();
            if include_path {
                let trimmed = maybe_path.trim();
                if !trimmed.is_empty() {
                    unique_non_empty.insert(trimmed.to_string());
                }
            }
            for raw in raw_paths {
                let trimmed = raw.trim();
                if !trimmed.is_empty() {
                    unique_non_empty.insert(trimmed.to_string());
                }
            }

            if unique_non_empty.len() > MAX_BATCH_PATHS {
                prop_assert!(result.is_err());
            } else {
                let parsed = result.unwrap();
                prop_assert!(!parsed.is_empty());
                prop_assert!(parsed.len() <= MAX_BATCH_PATHS);

                let mut seen = HashSet::new();
                for entry in &parsed {
                    prop_assert!(!entry.trim().is_empty());
                    prop_assert!(seen.insert(entry.clone()));
                }
            }
        }
    }
}
