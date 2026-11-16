//! MCP servers a project's repository declares in `.mcp.json` (the format
//! Claude Code and other agents read):
//!
//! ```json
//! { "mcpServers": { "docs": { "command": "npx", "args": ["-y", "docs-mcp"], "env": {"MODE": "ro"} } } }
//! ```
//!
//! Nothing here starts a process. A server runs only after the owner approves
//! its exact command for that project (`manage_mcp` `project_start`), under a
//! registry name that carries the project ([`registered_name`]), is not saved
//! across restarts, and is offered only on that project's turns
//! ([`offered_in_project`]). Environment values are passed as written:
//! `${VAR}` references are never expanded, so a repository cannot pull the
//! host's secrets into a process it chose.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Deserialize;

use super::McpServerConfig;

/// The file a project declares its MCP servers in, at its root.
pub const PROJECT_MCP_FILE: &str = ".mcp.json";
/// Registry names of project servers start with this.
pub const PROJECT_SERVER_PREFIX: &str = "prj_";
/// The largest `.mcp.json` read.
const MAX_FILE_BYTES: u64 = 256 * 1024;
/// At most this many servers are read from one project.
const MAX_SERVERS: usize = 32;

/// One server a project declares.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectServer {
    /// The name in `.mcp.json`.
    pub name: String,
    /// The name it runs under in the registry ([`registered_name`]).
    pub registered_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Why it cannot be started, when it cannot (a remote transport, an
    /// unexpanded `${VAR}`).
    pub unusable: Option<String>,
}

impl ProjectServer {
    /// How the registry starts it.
    pub fn config(&self) -> McpServerConfig {
        McpServerConfig {
            command: self.command.clone(),
            args: self.args.clone(),
            env: self
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<_, _>>(),
            read_only_tools: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
struct McpFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, RawServer>,
}

#[derive(Deserialize)]
struct RawServer {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default, rename = "type")]
    transport: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

/// The project id's part of a registry name: its first 8 letters or digits
/// after any `project-` prefix, lowercased.
fn project_tag(project_id: &str) -> String {
    project_id
        .strip_prefix("project-")
        .unwrap_or(project_id)
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .take(8)
        .collect()
}

/// The registry name of `name` from `project_id`: `prj_<tag>_<name>`, at
/// most 32 characters of lowercase letters, digits and `_`.
pub fn registered_name(project_id: &str, name: &str) -> String {
    let prefix = format!("{PROJECT_SERVER_PREFIX}{}_", project_tag(project_id));
    let mut clean = String::new();
    for c in name.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            clean.push(c);
        } else if !clean.ends_with('_') {
            clean.push('_');
        }
    }
    let clean = clean.trim_matches('_');
    let room = 32usize.saturating_sub(prefix.len());
    let clean: String = clean.chars().take(room).collect();
    format!(
        "{prefix}{}",
        if clean.is_empty() { "server" } else { &clean }
    )
}

/// Whether the tool or server `name` belongs to some project.
pub fn is_project_scoped(name: &str) -> bool {
    name.starts_with(PROJECT_SERVER_PREFIX)
}

/// Whether the tool or server `name` may be offered on a turn in
/// `project_id` (`None`: a turn in no project): every server that is not a
/// project's, and a project's only on that project's turns.
pub fn offered_in_project(name: &str, project_id: Option<&str>) -> bool {
    if !is_project_scoped(name) {
        return true;
    }
    match project_id {
        Some(id) => name.starts_with(&format!("{PROJECT_SERVER_PREFIX}{}_", project_tag(id))),
        None => false,
    }
}

/// The servers `root/.mcp.json` declares, by name. No file: none. A file
/// whose real path leaves `root`, that is too large or that does not parse
/// is an error naming why.
pub fn read_project_servers(root: &Path, project_id: &str) -> Result<Vec<ProjectServer>, String> {
    let path = root.join(PROJECT_MCP_FILE);
    let meta = match std::fs::symlink_metadata(&path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("could not read {PROJECT_MCP_FILE}: {error}")),
    };
    if meta.file_type().is_symlink() {
        let real_root = std::fs::canonicalize(root).map_err(|e| e.to_string())?;
        let real = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
        if !real.starts_with(&real_root) {
            return Err(format!("{PROJECT_MCP_FILE} links outside the project"));
        }
    }
    let size = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
    if size > MAX_FILE_BYTES {
        return Err(format!(
            "{PROJECT_MCP_FILE} is {size} bytes; at most {MAX_FILE_BYTES} are read"
        ));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let file: McpFile = serde_json::from_str(&text)
        .map_err(|error| format!("{PROJECT_MCP_FILE} is not valid: {error}"))?;
    Ok(file
        .mcp_servers
        .into_iter()
        .take(MAX_SERVERS)
        .map(|(name, raw)| {
            let remote = raw.url.is_some()
                || raw
                    .transport
                    .as_deref()
                    .is_some_and(|t| !t.eq_ignore_ascii_case("stdio"));
            let unexpanded = raw.args.iter().any(|a| a.contains("${"))
                || raw.env.values().any(|v| v.contains("${"))
                || raw.command.as_deref().is_some_and(|c| c.contains("${"));
            let unusable = if remote {
                Some("a remote (http/sse) server; only local commands are started".to_string())
            } else if raw.command.as_deref().is_none_or(|c| c.trim().is_empty()) {
                Some("no command".to_string())
            } else if unexpanded {
                Some(
                    "it uses ${VAR} references, which are never expanded from this computer's \
                     environment; add the server with manage_mcp add and set_env instead"
                        .to_string(),
                )
            } else {
                None
            };
            ProjectServer {
                registered_name: registered_name(project_id, &name),
                name,
                command: raw.command.unwrap_or_default().trim().to_string(),
                args: raw.args,
                env: raw.env,
                unusable,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "project-0a1b2c3d-0000-4000-8000-000000000001";

    #[test]
    fn names_carry_the_project_and_fit_the_registry() {
        let name = registered_name(ID, "Docs Search!");
        assert_eq!(name, "prj_0a1b2c3d_docs_search");
        let long = registered_name(ID, &"x".repeat(80));
        assert_eq!(long.len(), 32);
        assert!(
            long.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        );
        assert_eq!(registered_name(ID, "!!!"), "prj_0a1b2c3d_server");
    }

    #[test]
    fn project_servers_are_offered_only_in_their_project() {
        let tool = format!("{}__search", registered_name(ID, "docs"));
        assert!(offered_in_project(&tool, Some(ID)));
        assert!(!offered_in_project(&tool, Some("project-ffffffff-0000")));
        assert!(!offered_in_project(&tool, None));
        assert!(offered_in_project("github__search", None));
        assert!(offered_in_project("github__search", Some(ID)));
    }

    #[test]
    fn the_file_is_read_without_expanding_anything() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_project_servers(dir.path(), ID).unwrap().is_empty());
        std::fs::write(
            dir.path().join(PROJECT_MCP_FILE),
            r#"{"mcpServers": {
                "docs": {"command": "npx", "args": ["-y", "docs-mcp"], "env": {"MODE": "ro"}},
                "web": {"type": "http", "url": "https://mcp.example.invalid"},
                "keyed": {"command": "uvx", "args": ["tool"], "env": {"TOKEN": "${API_TOKEN}"}}
            }}"#,
        )
        .unwrap();
        let servers = read_project_servers(dir.path(), ID).unwrap();
        let docs = servers.iter().find(|s| s.name == "docs").unwrap();
        assert_eq!(docs.unusable, None);
        assert_eq!(docs.config().env["MODE"], "ro");
        assert_eq!(docs.registered_name, "prj_0a1b2c3d_docs");
        let web = servers.iter().find(|s| s.name == "web").unwrap();
        assert!(web.unusable.as_deref().unwrap().contains("remote"));
        let keyed = servers.iter().find(|s| s.name == "keyed").unwrap();
        assert!(
            keyed
                .unusable
                .as_deref()
                .unwrap()
                .contains("never expanded")
        );
        std::fs::write(dir.path().join(PROJECT_MCP_FILE), "{not json").unwrap();
        assert!(read_project_servers(dir.path(), ID).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_file_linked_from_outside_the_project_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("x.json"), r#"{"mcpServers":{}}"#).unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("x.json"),
            dir.path().join(PROJECT_MCP_FILE),
        )
        .unwrap();
        assert!(
            read_project_servers(dir.path(), ID)
                .unwrap_err()
                .contains("outside")
        );
    }
}
