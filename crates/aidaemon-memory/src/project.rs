//! Project memory: what the agent learns about one project, kept as plain
//! Markdown files in a directory of that project's own, apart from the
//! owner's personal memory.
//!
//! ```text
//! <root>/<project-id>/memory/
//!     MEMORY.md                 index, rebuilt after every write
//!     fact-<key>.md
//!     procedure-<key>.md
//!     pitfall-<key>.md
//!     reference-<key>.md
//! ```
//!
//! Each note is one file with a small front matter (`name`, `description`,
//! `type`, `category`, `source`, `updated`) and a body. Notes are keyed by
//! the project's id, so renaming the project or moving its folder keeps
//! them. A note the owner adds or edits by hand is read like any other; a
//! file without valid front matter is listed by its file name and never
//! deleted. The same key in the same project replaces that note and nothing
//! else: there is no fuzzy matching and nothing crosses projects.
//!
//! The files are plain text on disk: callers must never route personal or
//! identity memory here. Values of registered secrets are redacted before a
//! note is written.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aidaemon_core::ProjectRef;

/// The index file in each project's memory directory.
pub const INDEX_FILE: &str = "MEMORY.md";
/// A note's body is at most this many bytes.
pub const MAX_NOTE_BYTES: usize = 8 * 1024;
/// The index shown to the model is at most this many characters.
pub const MAX_INDEX_CHARS: usize = 4 * 1024;
/// A file larger than this is not read as a note.
const MAX_READ_BYTES: u64 = 256 * 1024;
/// At most this many notes are read for an index.
const MAX_NOTES: usize = 500;
/// A key is cut to this many characters in a file name.
const MAX_KEY_CHARS: usize = 80;
/// A description is cut to this many characters.
const MAX_DESCRIPTION_CHARS: usize = 160;

/// Where one memory goes on a turn that works in a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// The owner's personal memory, as on any other turn.
    Personal,
    /// The project's notes.
    Project,
}

impl MemoryScope {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "personal" | "owner" | "global" => Some(MemoryScope::Personal),
            "project" => Some(MemoryScope::Project),
            _ => None,
        }
    }
}

/// Categories about the owner and the people around them. They are never
/// project notes, whatever scope is asked for: notes are plain text on disk,
/// and who someone is does not belong to a project.
pub fn category_is_personal_only(category: &str) -> bool {
    matches!(
        category.trim().to_ascii_lowercase().as_str(),
        "user"
            | "personal"
            | "profile"
            | "identity"
            | "people"
            | "person"
            | "relationship"
            | "relationships"
            | "family"
            | "assistant"
            | "bot"
            | "health"
            | "contact"
    )
}

/// Where a memory of `category` goes. Without a project, always personal.
/// In a project, personal-only categories stay personal; otherwise the
/// `requested` scope wins, and without one a `preference` is personal and
/// anything else belongs to the project.
pub fn route(category: &str, requested: Option<MemoryScope>, in_project: bool) -> MemoryScope {
    if !in_project || category_is_personal_only(category) {
        return MemoryScope::Personal;
    }
    match requested {
        Some(scope) => scope,
        None if category.trim().eq_ignore_ascii_case("preference") => MemoryScope::Personal,
        None => MemoryScope::Project,
    }
}

/// The note kind a memory category names (`procedure`, `pitfall`,
/// `reference`), else a fact.
pub fn kind_for_category(category: &str) -> NoteKind {
    match NoteKind::parse(category) {
        Some(kind) => kind,
        None => match category.trim().to_ascii_lowercase().as_str() {
            "error_solution" | "error" | "gotcha" => NoteKind::Pitfall,
            "how_to" | "howto" | "workflow" => NoteKind::Procedure,
            _ => NoteKind::Fact,
        },
    }
}

/// The `source` of a [`crate::records::Fact`] that stands for a project
/// note in an extractor's ALREADY STORED list ([`notes_as_shown_facts`]).
pub const NOTE_FACT_SOURCE: &str = "project_note";

/// At most this many notes are shown to an extractor.
const MAX_NOTES_SHOWN: usize = 12;

/// The project's notes as stored facts an extractor is shown, so a
/// restatement of a note is marked instead of stored again and a correction
/// names the note it corrects. Each has a negative id, never a store row's.
pub fn notes_as_shown_facts(entries: &[NoteEntry]) -> Vec<crate::records::Fact> {
    let long_ago = chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap_or_default();
    entries
        .iter()
        .filter(|entry| entry.kind.is_some())
        .take(MAX_NOTES_SHOWN)
        .enumerate()
        .map(|(i, entry)| crate::records::Fact {
            id: -(i as i64) - 1,
            category: entry.category.clone().unwrap_or_else(|| {
                entry
                    .kind
                    .map(NoteKind::as_str)
                    .unwrap_or("project")
                    .to_string()
            }),
            key: entry.name.clone(),
            value: entry.description.clone(),
            source: NOTE_FACT_SOURCE.to_string(),
            created_at: long_ago,
            updated_at: long_ago,
            superseded_at: None,
            recall_count: 0,
            last_recalled_at: None,
            channel_id: None,
            privacy: crate::records::FactPrivacy::Private,
            first_seen_at: None,
            source_excerpt: None,
        })
        .collect()
}

/// Whether `fact` stands for a project note ([`notes_as_shown_facts`]).
pub fn is_note_fact(fact: &crate::records::Fact) -> bool {
    fact.id < 0 && fact.source == NOTE_FACT_SOURCE
}

/// What a note holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NoteKind {
    /// A stable fact about the project (a command, a convention, a path).
    Fact,
    /// How to do something in this project.
    Procedure,
    /// A mistake to avoid here and what works instead.
    Pitfall,
    /// Where something is (a document, a dashboard, a ticket).
    Reference,
}

impl NoteKind {
    pub const ALL: [NoteKind; 4] = [
        NoteKind::Fact,
        NoteKind::Procedure,
        NoteKind::Pitfall,
        NoteKind::Reference,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            NoteKind::Fact => "fact",
            NoteKind::Procedure => "procedure",
            NoteKind::Pitfall => "pitfall",
            NoteKind::Reference => "reference",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str().eq_ignore_ascii_case(value.trim()))
    }
}

/// Who wrote a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteSource {
    /// The agent, on purpose (a memory tool).
    Agent,
    /// Extracted from a conversation in the background.
    Extracted,
    /// Learned from how a run went.
    Learned,
}

impl NoteSource {
    pub fn as_str(self) -> &'static str {
        match self {
            NoteSource::Agent => "agent",
            NoteSource::Extracted => "extracted",
            NoteSource::Learned => "learned",
        }
    }
}

/// One note to write.
#[derive(Debug, Clone)]
pub struct NoteWrite {
    pub kind: NoteKind,
    /// The note's key; with the kind it names the file.
    pub key: String,
    /// The memory category it was given (`technical`, `project`, ...).
    pub category: Option<String>,
    /// One line for the index; the body's first line when `None`.
    pub description: Option<String>,
    pub body: String,
    pub source: NoteSource,
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    Created,
    Replaced,
    /// The note already said exactly this.
    Unchanged,
}

/// A note as the index lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteEntry {
    /// `None` for a file without valid front matter.
    pub kind: Option<NoteKind>,
    pub name: String,
    pub file: String,
    pub description: String,
    pub category: Option<String>,
}

/// Why a note was not written.
#[derive(Debug)]
pub enum NoteError {
    /// The key has no letters or digits.
    EmptyKey,
    /// The body is empty.
    EmptyBody,
    /// The body is over [`MAX_NOTE_BYTES`].
    TooLarge {
        bytes: usize,
    },
    /// The project's id is not a safe directory name.
    InvalidProject,
    /// The target is outside the project's memory directory (a symlink).
    OutsideDirectory,
    Io(io::Error),
}

impl std::fmt::Display for NoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoteError::EmptyKey => f.write_str("the key needs at least one letter or digit"),
            NoteError::EmptyBody => f.write_str("the value is empty"),
            NoteError::TooLarge { bytes } => write!(
                f,
                "the value is {bytes} bytes; a project note holds at most {MAX_NOTE_BYTES}"
            ),
            NoteError::InvalidProject => f.write_str("the project id is not valid"),
            NoteError::OutsideDirectory => {
                f.write_str("the note's path leaves the project's memory directory")
            }
            NoteError::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for NoteError {}

impl From<io::Error> for NoteError {
    fn from(error: io::Error) -> Self {
        NoteError::Io(error)
    }
}

/// Every project's memory, under one root directory.
pub struct ProjectNotes {
    root: PathBuf,
}

/// One lock per project memory directory, shared by every [`ProjectNotes`]
/// in the process, so writes to one project are serialized however many
/// handles the host made.
static LOCKS: std::sync::LazyLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
    std::sync::LazyLock::new(Mutex::default);

impl std::fmt::Debug for ProjectNotes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectNotes")
            .field("root", &self.root)
            .finish()
    }
}

impl ProjectNotes {
    /// Project memory under `root` (created on the first write).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory holding every project's memory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `project`'s memory directory, when its id is valid.
    pub fn dir(&self, project: &ProjectRef) -> Option<PathBuf> {
        ProjectRef::is_valid_id(&project.id).then(|| self.root.join(&project.id).join("memory"))
    }

    fn lock(&self, project_id: &str) -> Arc<Mutex<()>> {
        let key = self.root.join(project_id);
        let mut locks = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
        locks
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// The file name a note of `kind` and `key` is kept in, when the key
    /// has a letter or digit.
    pub fn file_name(kind: NoteKind, key: &str) -> Option<String> {
        let key = note_key(key)?;
        Some(format!("{}-{key}.md", kind.as_str()))
    }

    /// Writes (or replaces) one note.
    pub fn write(&self, project: &ProjectRef, note: &NoteWrite) -> Result<WriteOutcome, NoteError> {
        let dir = self.dir(project).ok_or(NoteError::InvalidProject)?;
        let key = note_key(&note.key).ok_or(NoteError::EmptyKey)?;
        let body = aidaemon_core::redaction::redact_registered(note.body.trim()).into_owned();
        if body.is_empty() {
            return Err(NoteError::EmptyBody);
        }
        if body.len() > MAX_NOTE_BYTES {
            return Err(NoteError::TooLarge { bytes: body.len() });
        }
        let lock = self.lock(&project.id);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        fs::create_dir_all(&dir)?;
        let dir = self.checked_dir(&dir)?;
        let file = format!("{}-{key}.md", note.kind.as_str());
        let path = dir.join(&file);
        refuse_symlink(&path)?;
        let description = one_line(
            note.description
                .as_deref()
                .filter(|d| !d.trim().is_empty())
                .unwrap_or_else(|| body.lines().find(|l| !l.trim().is_empty()).unwrap_or("")),
        );
        let description = aidaemon_core::redaction::redact_registered(&description).into_owned();
        let previous = fs::read_to_string(&path).ok();
        if let Some(previous) = &previous
            && let Some(parsed) = parse_note(previous)
            && parsed.body.trim() == body
            && parsed.description == description
            && parsed.kind == Some(note.kind)
            && parsed.category.as_deref() == note.category.as_deref().map(str::trim)
        {
            return Ok(WriteOutcome::Unchanged);
        }
        let mut text = String::new();
        text.push_str("---\n");
        text.push_str(&format!("name: {key}\n"));
        text.push_str(&format!("description: {description}\n"));
        text.push_str(&format!("type: {}\n", note.kind.as_str()));
        if let Some(category) = note.category.as_deref().map(one_line)
            && !category.is_empty()
        {
            text.push_str(&format!("category: {category}\n"));
        }
        text.push_str(&format!("source: {}\n", note.source.as_str()));
        text.push_str(&format!(
            "updated: {}\n",
            chrono::Utc::now().format("%Y-%m-%d")
        ));
        text.push_str("---\n");
        text.push_str(&body);
        text.push('\n');
        write_atomically(&path, &text)?;
        self.rebuild_index_locked(&dir)?;
        Ok(if previous.is_some() {
            WriteOutcome::Replaced
        } else {
            WriteOutcome::Created
        })
    }

    /// Deletes the note of `key`, of `kind` or of any kind. Returns whether
    /// a note was deleted.
    pub fn delete(
        &self,
        project: &ProjectRef,
        kind: Option<NoteKind>,
        key: &str,
    ) -> Result<bool, NoteError> {
        let dir = self.dir(project).ok_or(NoteError::InvalidProject)?;
        let Some(key) = note_key(key) else {
            return Ok(false);
        };
        if !dir.is_dir() {
            return Ok(false);
        }
        let lock = self.lock(&project.id);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let dir = self.checked_dir(&dir)?;
        let kinds: Vec<NoteKind> = match kind {
            Some(kind) => vec![kind],
            None => NoteKind::ALL.to_vec(),
        };
        let mut deleted = false;
        for kind in kinds {
            let path = dir.join(format!("{}-{key}.md", kind.as_str()));
            refuse_symlink(&path)?;
            match fs::remove_file(&path) {
                Ok(()) => deleted = true,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        if deleted {
            self.rebuild_index_locked(&dir)?;
        }
        Ok(deleted)
    }

    /// The note of `kind` and `key`, as written on disk.
    pub fn read(&self, project: &ProjectRef, kind: NoteKind, key: &str) -> Option<String> {
        let dir = self.dir(project)?;
        let path = dir.join(Self::file_name(kind, key)?);
        if fs::symlink_metadata(&path).ok()?.file_type().is_symlink() {
            return None;
        }
        fs::read_to_string(path).ok()
    }

    /// Every note of `project`, sorted by kind then name; empty when it has
    /// none.
    pub fn entries(&self, project: &ProjectRef) -> Vec<NoteEntry> {
        let Some(dir) = self.dir(project) else {
            return Vec::new();
        };
        read_entries(&dir)
    }

    /// The section shown to the model on `project`'s turns, or `None` when
    /// the project has no notes.
    pub fn section(&self, project: &ProjectRef) -> Option<String> {
        let dir = self.dir(project)?;
        let entries = read_entries(&dir);
        if entries.is_empty() {
            return None;
        }
        Some(render_section(project, &dir, &entries))
    }

    /// The index rebuilt from the notes on disk, after a write the caller
    /// made outside [`ProjectNotes::write`].
    pub fn rebuild_index(&self, project: &ProjectRef) -> Result<(), NoteError> {
        let dir = self.dir(project).ok_or(NoteError::InvalidProject)?;
        if !dir.is_dir() {
            return Ok(());
        }
        let lock = self.lock(&project.id);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let dir = self.checked_dir(&dir)?;
        self.rebuild_index_locked(&dir)
    }

    fn rebuild_index_locked(&self, dir: &Path) -> Result<(), NoteError> {
        let entries = read_entries(dir);
        let path = dir.join(INDEX_FILE);
        refuse_symlink(&path)?;
        if entries.is_empty() {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            return Ok(());
        }
        let mut text = String::from(
            "# Project memory\n\nGenerated from the notes in this directory; edit the notes, not this file.\n\n",
        );
        for entry in &entries {
            text.push_str(&index_line(entry));
            text.push('\n');
        }
        write_atomically(&path, &text)?;
        Ok(())
    }

    /// `dir`, resolved, when it is inside the resolved root.
    fn checked_dir(&self, dir: &Path) -> Result<PathBuf, NoteError> {
        let root = dunce_canonical(&self.root)?;
        let resolved = dunce_canonical(dir)?;
        if resolved.starts_with(&root) && resolved != root {
            Ok(resolved)
        } else {
            Err(NoteError::OutsideDirectory)
        }
    }
}

/// `path` resolved, without Windows' verbatim prefix.
fn dunce_canonical(path: &Path) -> io::Result<PathBuf> {
    let resolved = fs::canonicalize(path)?;
    #[cfg(windows)]
    {
        let text = resolved.to_string_lossy();
        if let Some(stripped) = text.strip_prefix(r"\\?\")
            && !stripped.starts_with("UNC\\")
        {
            return Ok(PathBuf::from(stripped));
        }
    }
    Ok(resolved)
}

fn refuse_symlink(path: &Path) -> Result<(), NoteError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(NoteError::OutsideDirectory),
        _ => Ok(()),
    }
}

/// Writes `text` to a temporary file beside `path`, then renames it over
/// `path`, so a reader never sees half a note.
fn write_atomically(path: &Path, text: &str) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("note path has no directory"))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| io::Error::other("note path has no file name"))?;
    let temp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut file = fs::File::create(&temp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
    }
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(())
}

/// A key as it appears in a file name: lowercase ASCII letters and digits,
/// other runs as one `_`; `None` without a letter or digit.
pub fn note_key(key: &str) -> Option<String> {
    let mut out = String::new();
    let mut last_sep = true;
    for ch in key.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
        if out.len() >= MAX_KEY_CHARS {
            break;
        }
    }
    let out = out.trim_matches('_').to_string();
    (!out.is_empty()).then_some(out)
}

/// `text` on one line, at most [`MAX_DESCRIPTION_CHARS`].
fn one_line(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= MAX_DESCRIPTION_CHARS {
        return line;
    }
    let cut: String = line.chars().take(MAX_DESCRIPTION_CHARS - 1).collect();
    format!("{}…", cut.trim_end())
}

struct ParsedNote {
    kind: Option<NoteKind>,
    name: Option<String>,
    description: String,
    category: Option<String>,
    body: String,
}

/// A note's front matter and body; `None` without front matter.
fn parse_note(text: &str) -> Option<ParsedNote> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let end = rest.find("\n---")?;
    let (header, after) = rest.split_at(end);
    let body = after
        .strip_prefix("\n---")
        .unwrap_or("")
        .trim_start_matches(['\r', '\n'])
        .to_string();
    let mut parsed = ParsedNote {
        kind: None,
        name: None,
        description: String::new(),
        category: None,
        body,
    };
    for line in header.lines() {
        let Some((field, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches('"').to_string();
        match field.trim() {
            "type" => parsed.kind = NoteKind::parse(&value),
            "name" if !value.is_empty() => parsed.name = Some(value),
            "description" => parsed.description = value,
            "category" if !value.is_empty() => parsed.category = Some(value),
            _ => {}
        }
    }
    Some(parsed)
}

/// Every note file in `dir`, sorted by kind (unknown last) then name.
fn read_entries(dir: &Path) -> Vec<NoteEntry> {
    let Ok(read) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for item in read.flatten() {
        if entries.len() >= MAX_NOTES {
            break;
        }
        let file = item.file_name().to_string_lossy().into_owned();
        if file == INDEX_FILE || file.starts_with('.') || !file.ends_with(".md") {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(item.path()) else {
            continue;
        };
        if !meta.is_file() || meta.len() > MAX_READ_BYTES {
            continue;
        }
        let Ok(text) = fs::read_to_string(item.path()) else {
            continue;
        };
        let stem = file.trim_end_matches(".md").to_string();
        let entry = match parse_note(&text) {
            Some(parsed) => NoteEntry {
                kind: parsed.kind,
                name: parsed
                    .name
                    .map(|n| one_line(&n))
                    .unwrap_or_else(|| stem.clone()),
                file,
                description: if parsed.description.is_empty() {
                    one_line(parsed.body.lines().next().unwrap_or(""))
                } else {
                    one_line(&parsed.description)
                },
                category: parsed.category,
            },
            None => NoteEntry {
                kind: None,
                name: stem,
                file,
                description: one_line(text.lines().find(|l| !l.trim().is_empty()).unwrap_or("")),
                category: None,
            },
        };
        entries.push(entry);
    }
    entries.sort_by(|a, b| {
        let rank = |entry: &NoteEntry| entry.kind.map(|k| k as usize).unwrap_or(usize::MAX);
        rank(a).cmp(&rank(b)).then_with(|| a.name.cmp(&b.name))
    });
    entries
}

fn index_line(entry: &NoteEntry) -> String {
    let kind = entry.kind.map(NoteKind::as_str).unwrap_or("note");
    if entry.description.is_empty() {
        format!("- {kind}: [{}]({})", entry.name, entry.file)
    } else {
        format!(
            "- {kind}: [{}]({}) — {}",
            entry.name, entry.file, entry.description
        )
    }
}

fn render_section(project: &ProjectRef, dir: &Path, entries: &[NoteEntry]) -> String {
    let mut lines = Vec::with_capacity(entries.len());
    for entry in entries {
        lines.push(index_line(entry));
    }
    let list = lines.join("\n");
    let total = list.chars().count();
    let list = if total > MAX_INDEX_CHARS {
        let mut cut = String::new();
        for line in &lines {
            if cut.chars().count() + line.chars().count() + 1 > MAX_INDEX_CHARS {
                break;
            }
            cut.push_str(line);
            cut.push('\n');
        }
        let shown = cut.chars().count();
        format!(
            "{}\n{}",
            cut.trim_end(),
            aidaemon_tools::text::truncation_notice_with_hint(
                shown,
                total,
                "List the directory above to see every note.",
            )
        )
    } else {
        list
    };
    format!(
        "## Project Memory ({name})\n\
         Notes kept for this project from earlier work, apart from the owner's personal memory. \
         Read a note's file before relying on its detail; the code is the truth when they differ. \
         Save what later work in this project should know with remember_fact (scope \"project\").\n\
         Directory: {dir}\n{list}",
        name = one_line(&project.name),
        dir = dir.display(),
    )
}

/// One project's notes as a [`Recall`](aidaemon_policies::Recall) backend:
/// its [`ProjectNotes::section`] as the note, for a memory system that
/// renders no block of its own to put the section in (wrap it in
/// [`MemoryInjection`](aidaemon_policies::MemoryInjection)).
pub struct ProjectNotesRecall {
    notes: Arc<ProjectNotes>,
    project: ProjectRef,
}

impl ProjectNotesRecall {
    pub fn new(notes: Arc<ProjectNotes>, project: ProjectRef) -> Self {
        Self { notes, project }
    }

    async fn section(&self) -> Option<String> {
        let (notes, project) = (self.notes.clone(), self.project.clone());
        tokio::task::spawn_blocking(move || notes.section(&project))
            .await
            .ok()
            .flatten()
    }
}

/// Which project a conversation is placed in. Recall uses it to keep one
/// project's conversation history (its episodes) to that project's turns,
/// and out of other projects' and personal recall.
#[async_trait::async_trait]
pub trait SessionProjects: Send + Sync {
    /// The project `session_id` is placed in; `None` when it is in none.
    async fn project_of(&self, session_id: &str) -> Option<String>;
}

#[async_trait::async_trait]
impl aidaemon_policies::Recall for ProjectNotesRecall {
    async fn recall(&self, _session_id: &str, _query: &str, _limit: usize) -> Vec<String> {
        self.section().await.into_iter().collect()
    }

    async fn render(&self, _session_id: &str, _query: &str, _limit: usize) -> Option<String> {
        self.section().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(id: &str) -> ProjectRef {
        ProjectRef::new(id, "Acme Site", None).unwrap()
    }

    fn note(key: &str, body: &str) -> NoteWrite {
        NoteWrite {
            kind: NoteKind::Fact,
            key: key.into(),
            category: Some("technical".into()),
            description: None,
            body: body.into(),
            source: NoteSource::Agent,
        }
    }

    #[test]
    fn routing_keeps_the_owner_personal() {
        use MemoryScope::*;
        // Without a project nothing changes.
        assert_eq!(route("technical", Some(Project), false), Personal);
        // Who the owner is never becomes a note.
        assert_eq!(route("user", Some(Project), true), Personal);
        assert_eq!(route("Family", None, true), Personal);
        assert_eq!(route("preference", None, true), Personal);
        assert_eq!(route("preference", Some(Project), true), Project);
        assert_eq!(route("technical", None, true), Project);
        assert_eq!(route("project", Some(Personal), true), Personal);
        assert_eq!(kind_for_category("pitfall"), NoteKind::Pitfall);
        assert_eq!(kind_for_category("error_solution"), NoteKind::Pitfall);
        assert_eq!(kind_for_category("technical"), NoteKind::Fact);
    }

    #[test]
    fn a_note_round_trips_and_is_indexed() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        assert_eq!(notes.section(&p), None);
        let outcome = notes
            .write(&p, &note("Test Command", "Tests run with `make test`."))
            .unwrap();
        assert_eq!(outcome, WriteOutcome::Created);
        let dir = notes.dir(&p).unwrap();
        let text = fs::read_to_string(dir.join("fact-test_command.md")).unwrap();
        assert!(text.contains("name: test_command"));
        assert!(text.contains("category: technical"));
        assert!(text.ends_with("Tests run with `make test`.\n"));
        let index = fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert!(
            index.contains("[test_command](fact-test_command.md) — Tests run with `make test`.")
        );
        let section = notes.section(&p).unwrap();
        assert!(section.starts_with("## Project Memory (Acme Site)"));
        assert!(section.contains("fact-test_command.md"));
    }

    #[test]
    fn the_same_key_replaces_only_its_note() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        notes.write(&p, &note("test_command", "make test")).unwrap();
        notes.write(&p, &note("test-runner", "make check")).unwrap();
        assert_eq!(
            notes.write(&p, &note("test_command", "make test")).unwrap(),
            WriteOutcome::Unchanged
        );
        assert_eq!(
            notes
                .write(&p, &note("TEST command", "cargo test"))
                .unwrap(),
            WriteOutcome::Replaced
        );
        let entries = notes.entries(&p);
        assert_eq!(entries.len(), 2);
        assert!(
            notes
                .read(&p, NoteKind::Fact, "test_runner")
                .unwrap()
                .contains("make check")
        );
        assert!(
            notes
                .read(&p, NoteKind::Fact, "test_command")
                .unwrap()
                .contains("cargo test")
        );
    }

    #[test]
    fn projects_are_apart() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        notes
            .write(&project("project-a"), &note("db", "Postgres 16"))
            .unwrap();
        assert!(notes.section(&project("project-b")).is_none());
        assert_eq!(notes.entries(&project("project-a")).len(), 1);
    }

    #[test]
    fn delete_removes_the_note_and_the_empty_index() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        notes.write(&p, &note("db", "Postgres 16")).unwrap();
        assert!(!notes.delete(&p, None, "other").unwrap());
        assert!(notes.delete(&p, None, "DB").unwrap());
        assert!(notes.entries(&p).is_empty());
        assert!(!notes.dir(&p).unwrap().join(INDEX_FILE).exists());
        assert!(!notes.delete(&project("project-none"), None, "db").unwrap());
    }

    #[test]
    fn hand_edited_and_malformed_notes_are_listed_and_kept() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        notes.write(&p, &note("db", "Postgres 16")).unwrap();
        let dir = notes.dir(&p).unwrap();
        fs::write(
            dir.join("procedure-release.md"),
            "---\nname: release\ndescription: How a release is cut\ntype: procedure\n---\nTag, then push.\n",
        )
        .unwrap();
        fs::write(
            dir.join("scratch.md"),
            "Remember the staging host is slow.\n",
        )
        .unwrap();
        fs::write(dir.join("notes.txt"), "not a note").unwrap();
        notes.rebuild_index(&p).unwrap();
        let entries = notes.entries(&p);
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["db", "release", "scratch"]);
        assert_eq!(entries[2].kind, None);
        assert_eq!(entries[2].description, "Remember the staging host is slow.");
        notes.write(&p, &note("db", "Postgres 17")).unwrap();
        assert!(dir.join("scratch.md").exists());
        let index = fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert!(index.contains("note: [scratch](scratch.md)"));
    }

    #[test]
    fn limits_and_bad_input_are_refused() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        assert!(matches!(
            notes.write(&p, &note("!!!", "x")),
            Err(NoteError::EmptyKey)
        ));
        assert!(matches!(
            notes.write(&p, &note("k", "  ")),
            Err(NoteError::EmptyBody)
        ));
        assert!(matches!(
            notes.write(&p, &note("k", &"x".repeat(MAX_NOTE_BYTES + 1))),
            Err(NoteError::TooLarge { .. })
        ));
        let bad = ProjectRef {
            id: "../escape".into(),
            name: "x".into(),
            root: None,
        };
        assert!(matches!(
            notes.write(&bad, &note("k", "v")),
            Err(NoteError::InvalidProject)
        ));
        assert!(!temp.path().join("escape").exists());
        let long = note_key(&"k".repeat(500)).unwrap();
        assert_eq!(long.len(), MAX_KEY_CHARS);
        assert_eq!(note_key("../../etc/passwd").unwrap(), "etc_passwd");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_note_or_directory_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        notes.write(&p, &note("db", "Postgres")).unwrap();
        let dir = notes.dir(&p).unwrap();
        let target = outside.path().join("victim.md");
        fs::write(&target, "keep").unwrap();
        std::os::unix::fs::symlink(&target, dir.join("fact-victim.md")).unwrap();
        assert!(matches!(
            notes.write(&p, &note("victim", "overwrite")),
            Err(NoteError::OutsideDirectory)
        ));
        assert_eq!(fs::read_to_string(&target).unwrap(), "keep");
        // A project directory that is a symlink out of the root.
        let q = project("project-2");
        fs::create_dir_all(temp.path().join("project-2")).unwrap();
        std::os::unix::fs::symlink(outside.path(), temp.path().join("project-2").join("memory"))
            .unwrap();
        assert!(matches!(
            notes.write(&q, &note("x", "y")),
            Err(NoteError::OutsideDirectory)
        ));
        assert!(!outside.path().join("fact-x.md").exists());
    }

    #[test]
    fn the_shown_index_is_bounded() {
        let temp = tempfile::tempdir().unwrap();
        let notes = ProjectNotes::new(temp.path());
        let p = project("project-1");
        for i in 0..120 {
            notes
                .write(&p, &note(&format!("fact_{i:03}"), &"detail ".repeat(20)))
                .unwrap();
        }
        let section = notes.section(&p).unwrap();
        let list_start = section.find("\n- ").unwrap();
        assert!(section[list_start..].chars().count() < MAX_INDEX_CHARS + 400);
        assert!(section.contains("fact_000"));
        assert!(!section.contains("fact_119"));
    }

    #[test]
    fn concurrent_writes_keep_every_note() {
        let temp = tempfile::tempdir().unwrap();
        let notes = Arc::new(ProjectNotes::new(temp.path()));
        let p = project("project-1");
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let notes = notes.clone();
                let p = p.clone();
                std::thread::spawn(move || {
                    notes.write(&p, &note(&format!("k{i}"), "v")).unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(notes.entries(&p).len(), 16);
        let index = fs::read_to_string(notes.dir(&p).unwrap().join(INDEX_FILE)).unwrap();
        assert_eq!(index.matches("\n- ").count(), 16);
        let leftovers = fs::read_dir(notes.dir(&p).unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
    }
}
