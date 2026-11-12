//! Typed key paths into `config.toml` and the edits made through them.
//!
//! A path is dotted keys with list positions: `provider.fallbacks[2].models`
//! or `provider.fallbacks.2.models` both name the `models` table of the third
//! `[[provider.fallbacks]]` entry. Edits go through `toml_edit`, so the rest
//! of the file keeps its layout and comments, and a path that does not lead
//! anywhere in the file is refused instead of creating a stray table.

use std::fmt;

use toml_edit::{DocumentMut, InlineTable, Item, Table, TableLike, Value};

/// One step of a config key path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PathSegment {
    Key(String),
    Index(usize),
}

/// A parsed config key path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ConfigPath {
    segments: Vec<PathSegment>,
}

impl fmt::Display for ConfigPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                PathSegment::Key(key) if i == 0 => write!(f, "{key}")?,
                PathSegment::Key(key) => write!(f, ".{key}")?,
                PathSegment::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

impl ConfigPath {
    pub(super) fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("the key path is empty".to_string());
        }
        let mut segments = Vec::new();
        for part in raw.split('.') {
            if part.is_empty() {
                return Err(format!("`{raw}` has an empty part between dots"));
            }
            if part.bytes().all(|b| b.is_ascii_digit()) {
                segments.push(PathSegment::Index(parse_index(raw, part)?));
                continue;
            }
            let (name, mut rest) = match part.find('[') {
                Some(at) => (&part[..at], &part[at..]),
                None => (part, ""),
            };
            if !name.is_empty() {
                if !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(format!(
                        "`{name}` in `{raw}` is not a config key (keys use letters, digits, `_` and `-`)"
                    ));
                }
                segments.push(PathSegment::Key(name.to_string()));
            }
            while !rest.is_empty() {
                let close = rest
                    .find(']')
                    .ok_or_else(|| format!("`{raw}` has a `[` without a closing `]`"))?;
                segments.push(PathSegment::Index(parse_index(raw, &rest[1..close])?));
                rest = &rest[close + 1..];
                if !rest.is_empty() && !rest.starts_with('[') {
                    return Err(format!("`{raw}`: put a dot between `]` and the next key"));
                }
            }
        }
        if !matches!(segments.first(), Some(PathSegment::Key(_))) {
            return Err(format!("`{raw}` must start with a key name"));
        }
        Ok(Self { segments })
    }

    pub(super) fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// The last key name in the path, if the path ends in a key.
    pub(super) fn last_key(&self) -> Option<&str> {
        match self.segments.last() {
            Some(PathSegment::Key(key)) => Some(key),
            _ => None,
        }
    }

    pub(super) fn ends_in_index(&self) -> bool {
        matches!(self.segments.last(), Some(PathSegment::Index(_)))
    }

    /// The value at this path in a parsed config, if there is one.
    pub(super) fn lookup<'a>(&self, root: &'a toml::Value) -> Option<&'a toml::Value> {
        let mut current = root;
        for segment in &self.segments {
            current = match segment {
                PathSegment::Key(key) => current.get(key.as_str())?,
                PathSegment::Index(index) => current.as_array()?.get(*index)?,
            };
        }
        Some(current)
    }

    /// Set the value at this path. Every table and list entry on the way
    /// must exist (a missing plain table is created, as TOML allows); the
    /// last key may be new.
    pub(super) fn set(&self, doc: &mut DocumentMut, value: Value) -> Result<(), String> {
        let (last, parents) = self.split_last();
        let parent = self.walk(Node::Item(doc.as_item_mut()), parents, true)?;
        match last {
            PathSegment::Key(key) => {
                let shown = self.prefix(parents.len());
                let table = parent.into_table_like().ok_or_else(|| {
                    format!("`{shown}` is not a table, so it has no `{key}` setting")
                })?;
                match table.get_mut(key) {
                    Some(Item::Value(old)) => {
                        let decor = old.decor().clone();
                        *old = value;
                        *old.decor_mut() = decor;
                    }
                    _ => {
                        table.insert(key, Item::Value(value));
                    }
                }
                Ok(())
            }
            PathSegment::Index(index) => {
                let shown = self.prefix(parents.len());
                match parent {
                    Node::Item(Item::ArrayOfTables(entries)) => {
                        let len = entries.len();
                        let slot = entries
                            .get_mut(*index)
                            .ok_or_else(|| out_of_range(&shown, *index, len))?;
                        let Value::InlineTable(table) = value else {
                            return Err(format!(
                                "each `{shown}` entry is a table; give the new entry as an inline table"
                            ));
                        };
                        *slot = table.into_table();
                        Ok(())
                    }
                    Node::Item(Item::Value(Value::Array(items)))
                    | Node::Value(Value::Array(items)) => {
                        let len = items.len();
                        let slot = items
                            .get_mut(*index)
                            .ok_or_else(|| out_of_range(&shown, *index, len))?;
                        let decor = slot.decor().clone();
                        *slot = value;
                        *slot.decor_mut() = decor;
                        Ok(())
                    }
                    _ => Err(format!("`{shown}` is not a list")),
                }
            }
        }
    }

    /// Remove the setting or list entry at this path; it must exist.
    pub(super) fn unset(&self, doc: &mut DocumentMut) -> Result<(), String> {
        let (last, parents) = self.split_last();
        let parent = self.walk(Node::Item(doc.as_item_mut()), parents, false)?;
        let shown = self.prefix(parents.len());
        match last {
            PathSegment::Key(key) => {
                let table = parent
                    .into_table_like()
                    .ok_or_else(|| format!("`{shown}` is not a table"))?;
                table
                    .remove(key)
                    .map(|_| ())
                    .ok_or_else(|| format!("config.toml has no setting `{self}`"))
            }
            PathSegment::Index(index) => match parent {
                Node::Item(Item::ArrayOfTables(entries)) => {
                    if *index >= entries.len() {
                        return Err(out_of_range(&shown, *index, entries.len()));
                    }
                    entries.remove(*index);
                    Ok(())
                }
                Node::Item(Item::Value(Value::Array(items))) | Node::Value(Value::Array(items)) => {
                    if *index >= items.len() {
                        return Err(out_of_range(&shown, *index, items.len()));
                    }
                    items.remove(*index);
                    Ok(())
                }
                _ => Err(format!("`{shown}` is not a list")),
            },
        }
    }

    fn split_last(&self) -> (&PathSegment, &[PathSegment]) {
        self.segments
            .split_last()
            .expect("a parsed path has at least one segment")
    }

    /// The path made of the first `len` segments, for messages.
    fn prefix(&self, len: usize) -> ConfigPath {
        ConfigPath {
            segments: self.segments[..len].to_vec(),
        }
    }

    fn walk<'a>(
        &self,
        mut node: Node<'a>,
        segments: &[PathSegment],
        create_tables: bool,
    ) -> Result<Node<'a>, String> {
        for (depth, segment) in segments.iter().enumerate() {
            let shown = self.prefix(depth);
            node = match segment {
                PathSegment::Key(key) => {
                    let here = self.prefix(depth + 1);
                    let entries = node.list_of_tables_len();
                    let inline = node.is_inline();
                    let Some(table) = node.into_table_like() else {
                        return Err(match entries {
                            Some(len) => format!(
                                "`{shown}` is a list of {len} entries; name one, for example `{shown}[0].{key}`"
                            ),
                            None => format!("`{shown}` is not a table, so it has no `{key}`"),
                        });
                    };
                    if !table.contains_key(key) {
                        if !create_tables {
                            return Err(format!("config.toml has no `{here}`"));
                        }
                        let created = if inline {
                            Item::Value(Value::InlineTable(InlineTable::new()))
                        } else {
                            let mut table = Table::new();
                            table.set_implicit(true);
                            Item::Table(table)
                        };
                        table.insert(key, created);
                    }
                    Node::Item(
                        table
                            .get_mut(key)
                            .expect("the key was just checked or inserted"),
                    )
                }
                PathSegment::Index(index) => match node {
                    Node::Item(Item::ArrayOfTables(entries)) => {
                        let len = entries.len();
                        Node::Table(
                            entries
                                .get_mut(*index)
                                .ok_or_else(|| out_of_range(&shown, *index, len))?,
                        )
                    }
                    Node::Item(Item::Value(Value::Array(items)))
                    | Node::Value(Value::Array(items)) => {
                        let len = items.len();
                        Node::Value(
                            items
                                .get_mut(*index)
                                .ok_or_else(|| out_of_range(&shown, *index, len))?,
                        )
                    }
                    _ => return Err(format!("`{shown}` is not a list, so it has no [{index}]")),
                },
            };
        }
        Ok(node)
    }
}

fn parse_index(raw: &str, text: &str) -> Result<usize, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("`{text}` in `{raw}` is not a list position"))
}

fn out_of_range(list: &ConfigPath, index: usize, len: usize) -> String {
    match len {
        0 => format!("`{list}` is empty, so it has no [{index}]"),
        1 => format!("`{list}` has 1 entry ([0]), so it has no [{index}]"),
        _ => format!(
            "`{list}` has {len} entries ([0] to [{}]), so it has no [{index}]",
            len - 1
        ),
    }
}

/// A place in the document a path walks through.
enum Node<'a> {
    Item(&'a mut Item),
    Table(&'a mut Table),
    Value(&'a mut Value),
}

impl<'a> Node<'a> {
    fn into_table_like(self) -> Option<&'a mut dyn TableLike> {
        match self {
            Node::Item(item) => item.as_table_like_mut(),
            Node::Table(table) => Some(table),
            Node::Value(value) => value
                .as_inline_table_mut()
                .map(|table| table as &mut dyn TableLike),
        }
    }

    fn is_inline(&self) -> bool {
        matches!(self, Node::Item(Item::Value(_)) | Node::Value(_))
    }

    fn list_of_tables_len(&self) -> Option<usize> {
        match self {
            Node::Item(Item::ArrayOfTables(entries)) => Some(entries.len()),
            _ => None,
        }
    }
}

/// Parse a value the way it is written after `key = ` in TOML. Text that is
/// not a TOML value (`gemini-4-hydrogen` without quotes) is taken as a
/// string, since that is what it can only mean.
pub(super) fn parse_value(raw: &str) -> Value {
    let raw = raw.trim();
    raw.parse::<Value>()
        .map(|mut value| {
            value.decor_mut().clear();
            value
        })
        .unwrap_or_else(|_| Value::from(raw))
}

/// The same value as a `toml::Value`, to compare with what the file holds.
pub(super) fn plain_value(value: &Value) -> Option<toml::Value> {
    let table: toml::Table = format!("v = {value}").parse().ok()?;
    table.get("v").cloned()
}

/// A value of the other shape than `value`, to learn whether the config
/// reads the key at all: the daemon's config refuses it where the key is
/// read, and ignores it where it is not.
pub(super) fn probe_value(value: &Value) -> Value {
    if value.is_inline_table() {
        Value::from("aidaemon-config-probe")
    } else {
        let mut table = InlineTable::new();
        table.insert("aidaemon_config_probe", Value::from(true));
        Value::InlineTable(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"# primary provider
[provider]
kind = "openai_compatible"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"

[provider.fallbacks.models]
default = "first-model"

[[provider.fallbacks]]
kind = "google_genai"

[provider.fallbacks.models]
default = "old-model" # keep this note
fallback = ["a", "b"]
"#;

    #[test]
    fn index_forms_parse_to_the_same_path() {
        let bracket = ConfigPath::parse("provider.fallbacks[1].models.default").unwrap();
        let dotted = ConfigPath::parse("provider.fallbacks.1.models.default").unwrap();
        assert_eq!(bracket, dotted);
        assert_eq!(bracket.to_string(), "provider.fallbacks[1].models.default");
        assert!(ConfigPath::parse("provider..models").is_err());
        assert!(ConfigPath::parse("provider.fallbacks[x]").is_err());
        assert!(ConfigPath::parse("provider.fallbacks[1").is_err());
        assert!(ConfigPath::parse("[0].models").is_err());
        assert!(ConfigPath::parse("provider.\"fallbacks[2]\"").is_err());
    }

    #[test]
    fn a_list_position_edits_that_entry_of_an_array_of_tables() {
        let mut doc: DocumentMut = DOC.parse().unwrap();
        ConfigPath::parse("provider.fallbacks[1].models.default")
            .unwrap()
            .set(&mut doc, parse_value("new-model"))
            .unwrap();
        let text = doc.to_string();
        assert!(!text.contains("fallbacks[1]"), "{text}");
        assert!(
            text.contains("default = \"new-model\" # keep this note"),
            "{text}"
        );
        assert!(text.contains("default = \"first-model\""), "{text}");
        assert!(text.starts_with("# primary provider"), "{text}");
        let parsed: toml::Value = text.parse().unwrap();
        assert_eq!(
            parsed["provider"]["fallbacks"][1]["models"]["default"].as_str(),
            Some("new-model")
        );
    }

    #[test]
    fn paths_that_lead_nowhere_are_refused() {
        let mut doc: DocumentMut = DOC.parse().unwrap();
        let err = ConfigPath::parse("provider.fallbacks[2].models.default")
            .unwrap()
            .set(&mut doc, parse_value("x"))
            .unwrap_err();
        assert!(err.contains("has 2 entries ([0] to [1])"), "{err}");
        let err = ConfigPath::parse("provider.fallbacks.models.default")
            .unwrap()
            .set(&mut doc, parse_value("x"))
            .unwrap_err();
        assert!(err.contains("is a list of 2 entries"), "{err}");
        let err = ConfigPath::parse("provider.kind.inner")
            .unwrap()
            .set(&mut doc, parse_value("x"))
            .unwrap_err();
        assert!(err.contains("is not a table"), "{err}");
        assert_eq!(doc.to_string(), DOC);
    }

    #[test]
    fn unset_removes_an_existing_setting_only() {
        let mut doc: DocumentMut = DOC.parse().unwrap();
        ConfigPath::parse("provider.fallbacks[1].models.fallback")
            .unwrap()
            .unset(&mut doc)
            .unwrap();
        assert!(!doc.to_string().contains("fallback = ["));
        let err = ConfigPath::parse("provider.models.missing")
            .unwrap()
            .unset(&mut doc)
            .unwrap_err();
        assert!(err.contains("no setting"), "{err}");
    }

    #[test]
    fn bare_text_is_a_string_and_toml_keeps_its_type() {
        assert_eq!(
            parse_value("gemini-4-hydrogen").as_str(),
            Some("gemini-4-hydrogen")
        );
        assert_eq!(parse_value("\"quoted\"").as_str(), Some("quoted"));
        assert_eq!(parse_value("3").as_integer(), Some(3));
        assert!(parse_value("[\"a\", \"b\"]").is_array());
    }
}
