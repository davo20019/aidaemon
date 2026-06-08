use std::collections::{BTreeMap, VecDeque};

use serde_json::Value;

use crate::tools::fs_utils;
use crate::traits::{ReadFileResultMetadata, ReadFileSelectionMetadata};

const DEFAULT_MAX_ARTIFACTS: usize = 20;
const DEFAULT_MAX_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(in crate::agent) struct LineInterval {
    pub start: usize,
    pub end: usize,
}

impl LineInterval {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RequestedSelection {
    Full,
    Bounded { start: usize, end: usize },
    OpenEnded { start: usize },
    Tail { count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) struct ReadRequest {
    pub canonical_path: String,
    selection: RequestedSelection,
}

impl ReadRequest {
    pub fn full(path: impl Into<String>) -> Self {
        Self {
            canonical_path: path.into(),
            selection: RequestedSelection::Full,
        }
    }

    pub fn bounded(path: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            canonical_path: path.into(),
            selection: RequestedSelection::Bounded { start, end },
        }
    }

    pub fn open_ended(path: impl Into<String>, start: usize) -> Self {
        Self {
            canonical_path: path.into(),
            selection: RequestedSelection::OpenEnded { start },
        }
    }

    pub fn tail(path: impl Into<String>, count: usize) -> Self {
        Self {
            canonical_path: path.into(),
            selection: RequestedSelection::Tail { count },
        }
    }

    pub async fn from_arguments(arguments: &str) -> Option<Self> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        let canonical_path = canonical_path_from_value(&args).await?;
        if let Some(count) = args
            .get("tail_lines")
            .and_then(Value::as_u64)
            .or_else(|| args.get("last_lines").and_then(Value::as_u64))
            .or_else(|| args.get("last_n_lines").and_then(Value::as_u64))
        {
            return Some(Self::tail(canonical_path, count as usize));
        }
        let start = args
            .get("start_line")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let end = args
            .get("end_line")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        Some(match (start, end) {
            (Some(start), Some(end)) => Self::bounded(canonical_path, start, end),
            (Some(start), None) => Self::open_ended(canonical_path, start),
            (None, Some(end)) => Self::bounded(canonical_path, 1, end),
            (None, None) => Self::full(canonical_path),
        })
    }
}

pub(in crate::agent) async fn canonical_path_from_arguments(arguments: &str) -> Option<String> {
    let args: Value = serde_json::from_str(arguments).ok()?;
    canonical_path_from_value(&args).await
}

async fn canonical_path_from_value(args: &Value) -> Option<String> {
    let path = ["path", "file_path", "file", "filename"]
        .iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))?;
    let normalized = fs_utils::validate_path(path).ok()?;
    Some(
        tokio::fs::canonicalize(&normalized)
            .await
            .unwrap_or(normalized)
            .to_string_lossy()
            .into_owned(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum ReadDecision {
    Execute,
    Replay {
        metadata: ReadFileResultMetadata,
        covered_intervals: Vec<LineInterval>,
    },
    PartialOverlap {
        covered_intervals: Vec<LineInterval>,
        uncovered_intervals: Vec<LineInterval>,
    },
    Unknown,
}

#[derive(Debug, Clone)]
struct StoredArtifact {
    metadata: ReadFileResultMetadata,
    retained_bytes: usize,
}

#[derive(Debug)]
pub(in crate::agent) struct ReadFileObservationTracker {
    artifacts: VecDeque<StoredArtifact>,
    retained_bytes: usize,
    max_artifacts: usize,
    max_bytes: usize,
}

impl Default for ReadFileObservationTracker {
    fn default() -> Self {
        Self::with_limits(DEFAULT_MAX_ARTIFACTS, DEFAULT_MAX_BYTES)
    }
}

impl ReadFileObservationTracker {
    pub fn with_limits(max_artifacts: usize, max_bytes: usize) -> Self {
        Self {
            artifacts: VecDeque::new(),
            retained_bytes: 0,
            max_artifacts,
            max_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }

    pub fn clear(&mut self) {
        self.artifacts.clear();
        self.retained_bytes = 0;
    }

    pub fn invalidate_path(&mut self, canonical_path: &str) {
        self.artifacts
            .retain(|artifact| artifact.metadata.canonical_path != canonical_path);
        self.recalculate_retained_bytes();
    }

    pub fn insert(&mut self, metadata: ReadFileResultMetadata) {
        let retained_bytes = retained_bytes(&metadata);
        if retained_bytes > self.max_bytes {
            return;
        }

        if let Some(newest) = self
            .artifacts
            .iter()
            .rev()
            .find(|artifact| artifact.metadata.canonical_path == metadata.canonical_path)
        {
            if generation(&newest.metadata) != generation(&metadata) {
                self.invalidate_path(&metadata.canonical_path);
            }
        }

        self.retained_bytes = self.retained_bytes.saturating_add(retained_bytes);
        self.artifacts.push_back(StoredArtifact {
            metadata,
            retained_bytes,
        });
        while self.artifacts.len() > self.max_artifacts || self.retained_bytes > self.max_bytes {
            if let Some(evicted) = self.artifacts.pop_front() {
                self.retained_bytes = self.retained_bytes.saturating_sub(evicted.retained_bytes);
            } else {
                break;
            }
        }
    }

    pub fn decide(&self, request: &ReadRequest) -> ReadDecision {
        let matching: Vec<&ReadFileResultMetadata> = self
            .artifacts
            .iter()
            .filter(|artifact| artifact.metadata.canonical_path == request.canonical_path)
            .map(|artifact| &artifact.metadata)
            .collect();
        let Some(newest) = matching.last().copied() else {
            return ReadDecision::Execute;
        };
        let newest_generation = generation(newest);
        let matching: Vec<&ReadFileResultMetadata> = matching
            .into_iter()
            .filter(|metadata| generation(metadata) == newest_generation)
            .collect();

        if newest.total_lines == 0 {
            return if matches!(request.selection, RequestedSelection::Full) {
                ReadDecision::Replay {
                    metadata: newest.clone(),
                    covered_intervals: Vec::new(),
                }
            } else {
                ReadDecision::Execute
            };
        }

        let Some(target) = requested_interval(request, newest.total_lines) else {
            return ReadDecision::Unknown;
        };
        let mut lines = BTreeMap::<usize, String>::new();
        let mut conflict = false;
        for metadata in &matching {
            let Some(start) = metadata.returned_start_line else {
                continue;
            };
            for (offset, text) in metadata.selected_lines.iter().enumerate() {
                let line_number = start + offset;
                if let Some(existing) = lines.get(&line_number) {
                    if existing != text {
                        conflict = true;
                    }
                } else {
                    lines.insert(line_number, text.clone());
                }
            }
        }

        if conflict {
            return ReadDecision::Unknown;
        }

        let covered_intervals = intervals_from_lines(
            lines
                .keys()
                .copied()
                .filter(|line| *line >= target.start && *line <= target.end),
        );
        if covered_intervals.is_empty() {
            return ReadDecision::Execute;
        }
        let uncovered_intervals = subtract_intervals(target, &covered_intervals);
        if !uncovered_intervals.is_empty() {
            return ReadDecision::PartialOverlap {
                covered_intervals,
                uncovered_intervals,
            };
        }

        let selected_lines = (target.start..=target.end)
            .filter_map(|line| lines.get(&line).cloned())
            .collect::<Vec<_>>();
        if selected_lines.len() != target.end - target.start + 1 {
            return ReadDecision::Unknown;
        }
        let selection = match request.selection {
            RequestedSelection::Full => ReadFileSelectionMetadata::Full,
            RequestedSelection::Bounded { start, end } => ReadFileSelectionMetadata::BoundedRange {
                start_line: start,
                end_line: end,
            },
            RequestedSelection::OpenEnded { start } => {
                ReadFileSelectionMetadata::OpenEndedRange { start_line: start }
            }
            RequestedSelection::Tail { count } => ReadFileSelectionMetadata::Tail {
                requested_lines: count,
            },
        };
        ReadDecision::Replay {
            metadata: ReadFileResultMetadata {
                display_path: newest.display_path.clone(),
                canonical_path: newest.canonical_path.clone(),
                selection,
                returned_start_line: Some(target.start),
                returned_end_line: Some(target.end),
                total_lines: newest.total_lines,
                file_size: newest.file_size,
                modified: newest.modified.clone(),
                selected_lines,
            },
            covered_intervals,
        }
    }

    fn recalculate_retained_bytes(&mut self) {
        self.retained_bytes = self
            .artifacts
            .iter()
            .map(|artifact| artifact.retained_bytes)
            .sum();
    }
}

fn generation(metadata: &ReadFileResultMetadata) -> (u64, Option<&str>) {
    (metadata.file_size, metadata.modified.as_deref())
}

fn retained_bytes(metadata: &ReadFileResultMetadata) -> usize {
    metadata.display_path.len()
        + metadata.canonical_path.len()
        + metadata.modified.as_ref().map_or(0, String::len)
        + metadata
            .selected_lines
            .iter()
            .map(|line| line.len())
            .sum::<usize>()
}

fn requested_interval(request: &ReadRequest, total_lines: usize) -> Option<LineInterval> {
    let interval = match request.selection {
        RequestedSelection::Full => LineInterval::new(1, total_lines),
        RequestedSelection::Bounded { start, end } if start > 0 && end >= start => {
            LineInterval::new(start, end.min(total_lines))
        }
        RequestedSelection::OpenEnded { start } if start > 0 && start <= total_lines => {
            LineInterval::new(start, total_lines)
        }
        RequestedSelection::Tail { count } if count > 0 => LineInterval::new(
            total_lines.saturating_sub(count).saturating_add(1).max(1),
            total_lines,
        ),
        _ => return None,
    };
    (interval.start <= interval.end).then_some(interval)
}

fn intervals_from_lines(lines: impl Iterator<Item = usize>) -> Vec<LineInterval> {
    let mut intervals: Vec<LineInterval> = Vec::new();
    for line in lines {
        match intervals.last_mut() {
            Some(last) if last.end + 1 == line => last.end = line,
            _ => intervals.push(LineInterval::new(line, line)),
        }
    }
    intervals
}

fn subtract_intervals(target: LineInterval, covered: &[LineInterval]) -> Vec<LineInterval> {
    let mut uncovered = Vec::new();
    let mut cursor = target.start;
    for interval in covered {
        if cursor < interval.start {
            uncovered.push(LineInterval::new(cursor, interval.start - 1));
        }
        cursor = cursor.max(interval.end.saturating_add(1));
    }
    if cursor <= target.end {
        uncovered.push(LineInterval::new(cursor, target.end));
    }
    uncovered
}

#[cfg(test)]
mod tests {
    use crate::traits::{ReadFileResultMetadata, ReadFileSelectionMetadata};

    use super::{LineInterval, ReadDecision, ReadFileObservationTracker, ReadRequest};

    fn artifact(
        path: &str,
        selection: ReadFileSelectionMetadata,
        start: Option<usize>,
        end: Option<usize>,
        total_lines: usize,
        modified: &str,
        lines: &[&str],
    ) -> ReadFileResultMetadata {
        ReadFileResultMetadata {
            display_path: path.to_string(),
            canonical_path: path.to_string(),
            selection,
            returned_start_line: start,
            returned_end_line: end,
            total_lines,
            file_size: (total_lines * 16) as u64,
            modified: Some(modified.to_string()),
            selected_lines: lines.iter().map(|line| (*line).to_string()).collect(),
        }
    }

    #[test]
    fn full_artifact_replays_duplicate_full_request() {
        let mut tracker = ReadFileObservationTracker::default();
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::Full,
            Some(1),
            Some(3),
            3,
            "v1",
            &["one", "two", "three"],
        ));

        let decision = tracker.decide(&ReadRequest::full("/tmp/resume.md"));

        let ReadDecision::Replay { metadata, .. } = decision else {
            panic!("expected cached replay");
        };
        assert_eq!(metadata.selected_lines, vec!["one", "two", "three"]);
        assert!(matches!(
            metadata.selection,
            ReadFileSelectionMetadata::Full
        ));
    }

    #[test]
    fn covered_inner_range_replays_only_requested_lines() {
        let mut tracker = ReadFileObservationTracker::default();
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::Full,
            Some(1),
            Some(4),
            4,
            "v1",
            &["one", "two", "three", "four"],
        ));

        let decision = tracker.decide(&ReadRequest::bounded("/tmp/resume.md", 2, 3));

        let ReadDecision::Replay { metadata, .. } = decision else {
            panic!("expected cached replay");
        };
        assert_eq!(metadata.returned_start_line, Some(2));
        assert_eq!(metadata.returned_end_line, Some(3));
        assert_eq!(metadata.selected_lines, vec!["two", "three"]);
    }

    #[test]
    fn adjacent_artifacts_can_satisfy_spanning_request() {
        let mut tracker = ReadFileObservationTracker::default();
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::BoundedRange {
                start_line: 1,
                end_line: 2,
            },
            Some(1),
            Some(2),
            4,
            "v1",
            &["one", "two"],
        ));
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::BoundedRange {
                start_line: 3,
                end_line: 4,
            },
            Some(3),
            Some(4),
            4,
            "v1",
            &["three", "four"],
        ));

        let decision = tracker.decide(&ReadRequest::bounded("/tmp/resume.md", 1, 4));

        let ReadDecision::Replay { metadata, .. } = decision else {
            panic!("expected assembled replay");
        };
        assert_eq!(metadata.selected_lines, vec!["one", "two", "three", "four"]);
    }

    #[test]
    fn partial_overlap_reports_prefix_and_suffix_gaps() {
        let mut tracker = ReadFileObservationTracker::default();
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::BoundedRange {
                start_line: 3,
                end_line: 5,
            },
            Some(3),
            Some(5),
            8,
            "v1",
            &["three", "four", "five"],
        ));

        let decision = tracker.decide(&ReadRequest::bounded("/tmp/resume.md", 1, 8));

        assert_eq!(
            decision,
            ReadDecision::PartialOverlap {
                covered_intervals: vec![LineInterval::new(3, 5)],
                uncovered_intervals: vec![LineInterval::new(1, 2), LineInterval::new(6, 8)],
            }
        );
    }

    #[test]
    fn conflicting_cached_lines_force_physical_execution() {
        let mut tracker = ReadFileObservationTracker::default();
        for line in ["old", "new"] {
            tracker.insert(artifact(
                "/tmp/resume.md",
                ReadFileSelectionMetadata::BoundedRange {
                    start_line: 2,
                    end_line: 2,
                },
                Some(2),
                Some(2),
                3,
                "v1",
                &[line],
            ));
        }

        assert_eq!(
            tracker.decide(&ReadRequest::bounded("/tmp/resume.md", 2, 2)),
            ReadDecision::Unknown
        );
    }

    #[test]
    fn changed_generation_replaces_old_artifacts() {
        let mut tracker = ReadFileObservationTracker::default();
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::Full,
            Some(1),
            Some(1),
            1,
            "v1",
            &["old"],
        ));
        tracker.insert(artifact(
            "/tmp/resume.md",
            ReadFileSelectionMetadata::Full,
            Some(1),
            Some(1),
            1,
            "v2",
            &["new"],
        ));

        let ReadDecision::Replay { metadata, .. } =
            tracker.decide(&ReadRequest::full("/tmp/resume.md"))
        else {
            panic!("expected replay from newest generation");
        };
        assert_eq!(metadata.selected_lines, vec!["new"]);
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn invalidation_removes_only_target_path() {
        let mut tracker = ReadFileObservationTracker::default();
        for path in ["/tmp/a.md", "/tmp/b.md"] {
            tracker.insert(artifact(
                path,
                ReadFileSelectionMetadata::Full,
                Some(1),
                Some(1),
                1,
                "v1",
                &[path],
            ));
        }

        tracker.invalidate_path("/tmp/a.md");

        assert_eq!(
            tracker.decide(&ReadRequest::full("/tmp/a.md")),
            ReadDecision::Execute
        );
        assert!(matches!(
            tracker.decide(&ReadRequest::full("/tmp/b.md")),
            ReadDecision::Replay { .. }
        ));
    }

    #[test]
    fn oversized_artifact_is_not_retained() {
        let mut tracker = ReadFileObservationTracker::with_limits(20, 8);
        tracker.insert(artifact(
            "/tmp/large.md",
            ReadFileSelectionMetadata::Full,
            Some(1),
            Some(1),
            1,
            "v1",
            &["more than eight bytes"],
        ));

        assert_eq!(tracker.len(), 0);
        assert_eq!(
            tracker.decide(&ReadRequest::full("/tmp/large.md")),
            ReadDecision::Execute
        );
    }
}
