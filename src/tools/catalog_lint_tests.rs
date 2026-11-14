//! The schema lint, run over the tool catalog the model actually sees
//! (`startup::tools::lint_catalog`), not over source text: every tool's
//! rendered function object, its golden, and its line in the size budget.
//!
//! A tool cannot join the catalog without a golden (`<name>.golden`) and a
//! line in `_budget.golden`. Regenerate both on purpose with
//! `UPDATE_TOOL_GOLDENS=1 cargo test --lib catalog_lint`, and read the diff:
//! the budget file is the tool-token number every roster change is measured
//! against.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::Value;

use crate::startup::tools::lint_catalog;
use crate::traits::Tool;

/// One tool's rendered function object may not exceed this. Big enough for
/// the largest tool today; a tool past it is split or trimmed, not waved
/// through.
const MAX_TOOL_CHARS: usize = 12_000;
/// The whole catalog, every feature on. A ceiling, not a target: the
/// roster work brings the per-call number down (`_budget.golden`).
const MAX_TOTAL_CHARS: usize = 115_000;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tools/schema_goldens")
}

fn updating() -> bool {
    std::env::var_os("UPDATE_TOOL_GOLDENS").is_some()
}

fn check_file(name: &str, rendered: &str) {
    let path = goldens_dir().join(name);
    if updating() {
        std::fs::create_dir_all(goldens_dir()).unwrap();
        std::fs::write(&path, rendered).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}; generate it with UPDATE_TOOL_GOLDENS=1",
            path.display()
        )
    });
    assert_eq!(
        rendered, expected,
        "{name} changed; if on purpose, rerun with UPDATE_TOOL_GOLDENS=1"
    );
}

/// What the model is sent for one tool.
fn function_object(tool: &dyn Tool) -> Value {
    serde_json::json!({"type": "function", "function": tool.schema()})
}

fn rendered_chars(tool: &dyn Tool) -> usize {
    function_object(tool).to_string().len()
}

/// Every tool, in catalog order.
async fn catalog_tools() -> (lint_catalog::Catalog, Vec<std::sync::Arc<dyn Tool>>) {
    let catalog = lint_catalog::build_full().await.unwrap();
    let tools = catalog.tools.clone();
    (catalog, tools)
}

#[tokio::test]
async fn catalog_tools_render_full_function_objects() {
    let (_catalog, tools) = catalog_tools().await;
    let mut names = BTreeSet::new();
    let mut problems = Vec::new();
    for tool in &tools {
        let name = tool.name();
        if !names.insert(name.to_string()) {
            problems.push(format!("{name}: registered twice"));
        }
        let schema = tool.schema();
        if schema["name"].as_str() != Some(name) {
            problems.push(format!("{name}: schema name is {:?}", schema["name"]));
        }
        match schema["description"].as_str() {
            Some(text) if text.trim().is_empty() => {
                problems.push(format!("{name}: empty description"))
            }
            Some(_) => {}
            None => problems.push(format!("{name}: no description")),
        }
        let parameters = &schema["parameters"];
        if parameters["type"] != "object" {
            problems.push(format!("{name}: parameters are not an object schema"));
        }
        if parameters["additionalProperties"] != Value::Bool(false) {
            problems.push(format!(
                "{name}: parameters.additionalProperties is not false"
            ));
        }
        let extra: Vec<&String> = schema
            .as_object()
            .map(|object| {
                object
                    .keys()
                    .filter(|key| !matches!(key.as_str(), "name" | "description" | "parameters"))
                    .collect()
            })
            .unwrap_or_default();
        if !extra.is_empty() {
            problems.push(format!("{name}: unexpected top-level keys {extra:?}"));
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// `tool.a.b` for every property without a description, nested objects
/// and array items included.
fn undescribed(prefix: &str, schema: &Value, out: &mut BTreeSet<String>) {
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property) in properties {
            let path = format!("{prefix}.{key}");
            let described = property
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.trim().is_empty());
            if !described {
                out.insert(path.clone());
            }
            undescribed(&path, property, out);
        }
    }
    if let Some(items) = schema.get("items") {
        undescribed(&format!("{prefix}[]"), items, out);
    }
}

/// Every parameter should say what it is. The ones that do not yet are
/// listed in `_undescribed.golden`; the list may only shrink: a new
/// undescribed parameter fails here.
#[tokio::test]
async fn catalog_parameters_are_described_or_listed() {
    let (_catalog, tools) = catalog_tools().await;
    let mut missing = BTreeSet::new();
    for tool in &tools {
        undescribed(tool.name(), &tool.schema()["parameters"], &mut missing);
    }
    let rendered: String = missing.iter().map(|path| format!("{path}\n")).collect();
    if updating() {
        check_file("_undescribed.golden", &rendered);
        return;
    }
    let path = goldens_dir().join("_undescribed.golden");
    let allowed: BTreeSet<String> = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect();
    let new: Vec<&String> = missing.difference(&allowed).collect();
    assert!(
        new.is_empty(),
        "parameters without a description (describe them): {new:#?}"
    );
    let fixed: Vec<&String> = allowed.difference(&missing).collect();
    assert!(
        fixed.is_empty(),
        "now described, remove from _undescribed.golden (UPDATE_TOOL_GOLDENS=1): {fixed:#?}"
    );
}

/// Each tool's name, description and parameter schema, pinned.
#[tokio::test]
async fn catalog_tool_goldens() {
    let (_catalog, tools) = catalog_tools().await;
    for tool in &tools {
        let value = serde_json::json!({
            "name": tool.name(),
            "description": tool.description(),
            "schema": tool.schema(),
        });
        let rendered = format!("{}\n", serde_json::to_string_pretty(&value).unwrap());
        check_file(&format!("{}.golden", tool.name()), &rendered);
    }
}

/// The size of every tool as sent (chars of the function object and the
/// chars/4 token estimate the daemon logs), and the total. A tool missing
/// a line, or a size that moved, fails until the budget is regenerated.
#[tokio::test]
async fn catalog_budget() {
    let (_catalog, tools) = catalog_tools().await;
    let sizes: BTreeMap<&str, usize> = tools
        .iter()
        .map(|tool| (tool.name(), rendered_chars(tool.as_ref())))
        .collect();
    let total: usize = sizes.values().sum();
    let mut rendered = String::from("# tool chars est_tokens\n");
    for (name, chars) in &sizes {
        rendered.push_str(&format!("{name} {chars} {}\n", chars.div_ceil(4)));
    }
    rendered.push_str(&format!(
        "TOTAL {total} {} ({} tools)\n",
        total.div_ceil(4),
        sizes.len()
    ));
    let over: Vec<_> = sizes
        .iter()
        .filter(|(_, chars)| **chars > MAX_TOOL_CHARS)
        .collect();
    assert!(
        over.is_empty(),
        "tools over {MAX_TOOL_CHARS} chars: {over:?}"
    );
    assert!(
        total <= MAX_TOTAL_CHARS,
        "catalog is {total} chars, over {MAX_TOTAL_CHARS}"
    );
    check_file("_budget.golden", &rendered);
}

/// Every golden on disk belongs to a tool in the catalog (or to a tool
/// pinned by its own test), so a removed tool takes its golden with it.
#[tokio::test]
async fn no_orphan_goldens() {
    if updating() {
        return;
    }
    let (_catalog, tools) = catalog_tools().await;
    let names: BTreeSet<String> = tools.iter().map(|tool| tool.name().to_string()).collect();
    let orphans: Vec<String> = std::fs::read_dir(goldens_dir())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .filter_map(|file| file.strip_suffix(".golden").map(str::to_string))
        .filter(|name| !name.starts_with('_') && !names.contains(name))
        .collect();
    assert!(orphans.is_empty(), "goldens with no tool: {orphans:?}");
}
