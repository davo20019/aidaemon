# File-Reading Strategy Design

## Problem

File-heavy tasks can waste context and produce inaccurate answers when the agent:

- reads the same file repeatedly,
- scans a file with overlapping line ranges,
- loses earlier file content to context truncation,
- starts with an arbitrary truncated range instead of locating the relevant text, or
- receives only a truncated cached replay after a repeated call.

The current loop guard hashes exact tool arguments and allows repeated `read_file` calls until the
sixth matching call. It does not recognize that different argument objects can describe the same
path or overlapping ranges. Successful results are cached by exact call hash and truncated at 8 KB,
which is insufficient for retaining a complete reusable file artifact.

## Goals

1. Block the second redundant `read_file` call within a task.
2. Recognize full coverage and overlap across different line-range arguments.
3. Replay complete cached read output when the requested content is already available.
4. Direct uncertain large-file lookup toward `search_files` followed by one exact range read.
5. Preserve legitimate reads of distinct, non-overlapping ranges.
6. Keep memory bounded and invalidate observations after file writes.

## Non-Goals

- Persisting file artifacts across tasks or daemon restarts.
- Automatically concatenating or rewriting file contents from overlapping observations.
- Replacing `search_files` or changing its matching behavior.
- Detecting external filesystem changes before every read with a separate metadata operation.
- Changing the general repetitive-call policy for tools other than `read_file`.

## Approach

Add task-scoped semantic tracking for successful `read_file` calls. The tracker is separate from the
existing exact-call result cache because it needs path and interval semantics rather than only a
hash.

### Read Observation

Each successful read records:

- normalized expanded path,
- selection kind: full, bounded range, open-ended range, or tail,
- returned inclusive line interval,
- total line count when reported by the tool,
- file size and modified timestamp reported in the result header,
- complete tool output,
- output character count, and
- insertion order for bounded eviction.

The tracker parses effective tool arguments and the structured header emitted by `ReadFileTool`.
Only successful, non-structural results become observations. If the result cannot be parsed
reliably, normal tool execution continues and the existing exact-call cache remains the fallback.

### Pre-Execution Decisions

Before executing `read_file`, normalize its path and requested selection and compare it with prior
observations for that path.

The decision order is:

1. **Exact or fully covered request:** replay the newest complete observation that covers the
   requested interval. This applies on the second request; the tool is not executed again.
2. **Partial overlap:** block the read and return guidance identifying the already-covered and
   uncovered intervals. Tell the model to request only the uncovered interval when sequential
   scanning is justified, or use `search_files` first when looking for specific content.
3. **Distinct range:** allow execution when the requested interval does not overlap prior coverage.
4. **Unknown semantics:** allow execution and fall back to existing guards.

A prior full-file observation covers all subsequent range reads for the same path. A bounded range
only covers requests inside its returned interval. Tail reads are compared using the concrete
returned interval parsed from the result, not the requested tail count.

Open-ended requests that start inside a cached interval are treated as partially overlapping unless
a full observation exists. This avoids assuming the cached interval reaches the current end of the
file.

### Freshness and Invalidation

The tracker is task-scoped, so cross-task staleness is impossible. Within a task:

- successful `write_file` and `edit_file` calls invalidate all observations for the normalized
  target path;
- failed writes do not invalidate observations;
- reads after invalidation execute normally and establish fresh observations;
- external changes made outside aidaemon tools are not proactively detected.

The modified timestamp remains recorded for diagnostics and future extension, but the design does
not add a filesystem metadata probe before every read. That would undermine the call-reduction goal.

### Cache Bounds

Complete artifacts are preferred over truncated replays. The semantic tracker uses both:

- a maximum observation count, and
- a maximum total character budget.

Eviction is oldest-first until both limits are satisfied. An individual result larger than the
total character budget is not stored in the semantic tracker; the call executes normally on future
requests rather than replaying incomplete data as complete. The existing exact-call cache may keep
its bounded diagnostic preview.

Initial implementation constants:

- 20 observations,
- 256 KiB total cached characters.

These values retain several typical source files or resumes without allowing task memory to grow
without bound.

## User-Facing Guidance

Update the `read_file` schema description and coding workflow prompt with these rules:

- Read a file in full once when it fits the tool limit.
- For large files and a known target, use `search_files`, then read one exact surrounding range.
- For sequential inspection, request only new non-overlapping ranges.
- Do not re-read a file or previously returned range; cached content will be replayed.

New tool-result notices must be task-neutral. Existing repeated-read notices assume the user is
editing code and demand `write_file`; that is incorrect for resume analysis and other read-only
tasks. Replayed content should instruct the model to answer or take the requested action, while
partial-overlap guidance should recommend a precise uncovered range or targeted search.

## Components

### `ReadFileObservationTracker`

A focused state component under `agent/loop/state/` owns observation insertion, eviction,
invalidation, and coverage decisions. It exposes semantic types rather than parsing tool calls in
the main loop.

### Tool Execution Integration

The tool execution phase:

1. asks the tracker for a pre-execution decision,
2. emits a synthetic tool message for replay or overlap guidance,
3. executes allowed calls normally,
4. records successful `read_file` results, and
5. invalidates paths after successful file edits or writes.

Synthetic replay and overlap responses count as handled tool calls but not successful external tool
executions. They must not inflate the generic repetitive-call counters into a hard failure.

### Notices and Telemetry

Add distinct notices for:

- cached read replay,
- fully covered range replay, and
- partial overlap guidance.

Emit a warning decision event containing the normalized path, decision kind, requested interval,
and covering/overlapping interval. Do not include file contents in telemetry metadata.

## Error Handling

- Path normalization failure: allow the read and use existing tool validation.
- Malformed arguments: allow existing argument-contract handling to report the error.
- Unparseable result header: do not create a semantic observation.
- Oversized result: skip semantic caching and preserve the existing bounded preview cache.
- Cache replay construction failure: execute the real read rather than blocking access.

## Testing

Unit tests will cover:

1. second identical full read replays immediately;
2. equivalent path aliases normalize to the same observation;
3. a full read covers later bounded ranges;
4. a bounded range covers an inner range;
5. partial overlap returns the exact uncovered interval;
6. adjacent and non-overlapping ranges execute;
7. tail reads use their returned concrete interval;
8. open-ended ranges are not incorrectly treated as fully covered;
9. successful edit/write invalidates observations;
10. failed edit/write preserves observations;
11. observation count and character budgets evict oldest entries;
12. oversized artifacts are not presented as complete cached results;
13. notices are neutral for analysis-only tasks;
14. generic repetitive guards still behave unchanged for other tools.

An integration test will script a model that reads a resume, repeats the same read, then requests an
overlapping range. It will verify that only the first physical read executes, the complete artifact
is replayed, and overlap guidance directs the model to targeted search or the uncovered range.

## Success Criteria

- A consecutive duplicate `read_file` call performs one physical file read, not two.
- Covered and overlapping ranges are recognized even when arguments differ.
- Cached replay never labels a truncated artifact as complete.
- The agent is explicitly guided to `search_files` plus one exact range for unknown locations.
- Existing legitimate multi-file and non-overlapping range workflows continue to work.
- Formatting, Clippy with all features, and the full test suite pass before commit.
