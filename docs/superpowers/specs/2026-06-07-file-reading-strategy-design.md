# File-Reading Strategy Design

**Status:** Proposed

## Problem

File-heavy tasks can waste context and produce inaccurate answers when the agent:

- reads the same file repeatedly,
- scans a file with overlapping line ranges,
- loses earlier file content to context truncation,
- starts with an arbitrary truncated range instead of locating the relevant text, or
- receives only a truncated cached replay after a repeated call.

The current loop guard hashes exact tool arguments and redirects when the same `read_file` hash
appears for the sixth time in the recent-call window. It does not recognize that different argument
objects can describe the same path or overlapping ranges. Successful results are cached by exact
call hash and truncated at 8 KB, which is insufficient for retaining a complete reusable file
artifact.

## Goals

1. Intercept the second redundant `read_file` call within a task without another physical read.
2. Recognize full coverage and overlap across different line-range arguments.
3. Replay complete cached read output when the requested content is already available.
4. Direct uncertain large-file lookup toward `search_files` followed by one exact range read.
5. Preserve legitimate reads of distinct, non-overlapping ranges.
6. Keep memory bounded and invalidate artifacts after filesystem mutations.

## Non-Goals

- Persisting file artifacts across tasks or daemon restarts.
- Replacing `search_files` or changing its matching behavior.
- Detecting external filesystem changes before every read with a separate metadata operation.
- Changing the general repetitive-call policy for tools other than `read_file`.
- Detecting whether an equivalent result is still present in the active model context before replay.

## Approach

Add task-scoped semantic tracking for successful `read_file` calls. The tracker is separate from the
existing exact-call result cache because it needs path, file-generation, and interval semantics
rather than only a hash.

### Typed Read Artifact

Each successful read records:

- canonical path key and original display path,
- selection kind: full, bounded range, open-ended range, or tail,
- returned inclusive line interval,
- total line count,
- file size and modified timestamp,
- complete selected lines without display line-number prefixes, captured before context-window
  compression,
- retained UTF-8 byte count for all owned strings, and
- insertion order for bounded eviction.

`ReadFileTool` must expose these fields as typed execution metadata. The tracker must not recover
semantic fields by parsing the human-readable `File: ...` header. User-visible output keeps its
current format, but header wording is not a tracker contract.

The execution path must make both forms of the result available:

- complete typed metadata, including selected lines, for semantic tracking, and
- the existing context-window-compressed output for the conversation message.

Only successful text-file reads become artifacts. Empty text files are valid full-file artifacts
with an empty returned interval and a total line count of zero. Binary-file results do not become
artifacts because they have no readable line interval.

### Path Identity

The tracker uses the same path argument aliases as `ReadFileTool`: `path`, `file_path`, `file`, and
`filename`. It expands `~`, resolves relative paths against the effective working directory, removes
lexical `.` and `..` components, and uses `tokio::fs::canonicalize()` when possible so symlink and
case aliases resolve to the same existing file on macOS. If canonicalization fails, the lexical
absolute path is used as a fallback and normal tool validation remains authoritative.

Successful path-specific mutations use the same normalization algorithm before invalidation.

### Pre-Execution Decisions

Before executing `read_file`, normalize its path and requested selection and compare it with the
union of prior artifacts for that path and file generation. A generation is identified by the
typed file-size and modified-timestamp pair. This is not a substitute for invalidation, but it
prevents artifacts from different observed generations from being combined.

The decision order is:

1. **Exact or fully covered request:** synthesize output for exactly the requested interval from
   the newest generation's cached lines. Coverage may come from one artifact or multiple contiguous
   artifacts. This applies on the second request; the tool is not executed again.
2. **Partial overlap:** intercept the read and return guidance listing every covered and uncovered
   interval. Tell the model to request each uncovered interval separately when sequential scanning
   is justified, or use `search_files` first when looking for specific content.
3. **Distinct range:** allow execution when the requested interval does not overlap prior coverage.
4. **Unknown semantics:** allow execution and fall back to existing guards.

A prior full-file artifact covers all subsequent range reads for the same generation. A bounded
range covers requests inside its returned interval. Adjacent artifacts may be combined only when
they share the same generation metadata and together cover the request without a gap. If duplicate
artifacts contain conflicting text for the same line and generation metadata, the tracker treats
the semantics as unknown and executes the real read.

Tail reads use their concrete returned interval, not only the requested tail count. An open-ended
request is fully covered when cached coverage runs continuously from its requested start through a
known `total_lines`, even if the source artifact was a bounded or tail read. Otherwise it is partial
or unknown; the tracker must not assume cached coverage reaches the current end of file.

Synthetic output must preserve the standard `ReadFileTool` header and line-number format while
including only the requested lines. Physical and synthetic reads use the same renderer. A range
replay from a larger artifact must not return the larger artifact's extra lines.

### Freshness and Invalidation

The tracker is task-scoped, so cross-task staleness is impossible. Within a task:

- successful `write_file` and `edit_file` calls invalidate all artifacts for the normalized target
  path;
- invalidation is keyed off pre-execution `call_semantics` computed from the effective arguments,
  not post-execution `result_metadata.semantics`. The tool execution phase already computes
  `call_semantics` before dispatch when compiling the step execution plan. Any execution whose
  pre-execution semantics mutate state — `mutation` or `observation_and_mutation` — and whose
  changed paths are not individually known invalidates the entire tracker immediately before tool
  dispatch, regardless of final status, because a failing command can still have partial side
  effects. This covers mutating `terminal`/`run_command` commands, `cli_agent`, and `spawn_agent`;
- executions classified as pure `observation` (e.g. `terminal cat`/`ls`/`grep`, read-only
  `http_request`) do not invalidate the tracker. Keying off the existing classifier rather than the
  tool name preserves cache utility across the read/terminal interleaving common in orchestrator
  mode, where blanket name-based invalidation would clear the tracker on nearly every turn;
- unknown pre-execution semantics are treated conservatively: if the tool capabilities permit
  mutation or external side effects, invalidate the entire tracker before dispatch. Unknown
  semantics for a tool that is explicitly read-only do not invalidate;
- failed `write_file` and `edit_file` calls preserve artifacts only when the operation is atomic and
  no write was committed. `write_file` is atomic today (temp-file + rename, `write_file.rs:146-149`);
  the implementer must confirm `edit_file` offers the same guarantee, and if it does not, a failed
  `edit_file` must invalidate the target path;
- a future structured changed-paths execution metadata field (which does not exist today — current
  `result_metadata` carries no changed-paths list) would let any tool, including MCP tools, apply
  path-specific invalidation when all changed paths are known and whole-tracker invalidation
  otherwise. Until that field exists, only `write_file`/`edit_file` (path from arguments) get
  path-specific invalidation; all other state-mutating executions fall to whole-tracker
  invalidation;
- reads after invalidation execute normally and establish fresh artifacts;
- when an allowed physical read returns size or modified-time metadata that differs from the newest
  generation for that path, older artifacts for the path are invalidated before insertion;
- external changes made outside aidaemon tools are not proactively detected before a covered replay.

This deliberately favors correctness over cache retention after opaque tools. A formatter or
sub-agent may touch only one file, but clearing a task-local 256 KiB tracker is cheaper than
replaying stale source. The design does not add a freshness `stat` before every covered replay.

### Shell Semantics Prerequisite

The existing shell-command classifier must be hardened before its output can control cache
invalidation. Prefix-only classification currently misclassifies commands such as
`echo value > file`, `cat > file`, and `ls && rm file` as pure observations.

Classification must inspect the complete shell expression before applying read-only command
prefixes:

- any output redirection, including `>`, `>>`, `>|`, descriptor redirection, or `&>`, makes the
  command state-mutating unless a shell-aware parser proves it targets a non-filesystem sink;
- compound commands separated by `&&`, `||`, or `;` combine the semantics of every segment;
- pipelines combine every stage, so `rg pattern file | head` remains an observation while
  `rg pattern file | tee output.txt` mutates state;
- quoted or escaped operator characters are not treated as shell operators;
- if tokenization or classification is ambiguous, fall back to `mutation`.

The combined effect is `observation` only when every segment is observational,
`observation_and_mutation` when observational and mutating segments are mixed, and `mutation` when
the expression is mutating without observational evidence. This classifier hardening is part of
this feature, not a follow-up, because stale-cache prevention depends on it.

`classify_shell_command` is shared. Its `ToolCallSemantics` output is consumed by `run_command`
(`run_command.rs`) and feeds `mutates_state()`/`observes_state()` checks in
`tool_prelude_phase.rs`, `execution_state.rs`, and the completion/verification gates, not only this
tracker. Hardening it reclassifies commands such as `echo value > file` from `observation` to
`mutation` for those subsystems as well. This is a net correctness gain — they currently
under-detect mutation — but it is behavior change outside the cache feature. Verification must
therefore include the full test suite and the completion/verification integration tests, not only
the classifier unit tests, to confirm no regression in those consumers.

### Cache Bounds

Complete artifacts are preferred over truncated replays. The semantic tracker uses both:

- a maximum artifact count, and
- a maximum total UTF-8 byte budget.

Eviction is oldest-first until both limits are satisfied. An individual result larger than the
total byte budget is not stored in the semantic tracker; the call executes normally on future
requests rather than replaying incomplete data as complete.

Initial implementation constants:

- 20 artifacts,
- 256 KiB total cached bytes.

`ReadFileTool` allows a full-file read up to 100 KiB before line-number formatting overhead, so this
budget typically retains two maximum-size full reads or more smaller source files and resumes.

## User-Facing Guidance

Update the `read_file` schema description and coding workflow prompt with these rules:

- Read a file in full once when it fits the tool limit.
- For large files and a known target, use `search_files`, then read one exact surrounding range.
- For sequential inspection, request only new non-overlapping ranges.
- Do not re-read a file or previously returned range; cached content will be replayed.

New tool-result notices must be task-neutral. Existing repeated-read notices assume the user is
editing code and demand `write_file`; that is incorrect for resume analysis and other read-only
tasks. Replayed content should instruct the model to answer or take the requested action, while
partial-overlap guidance should list precise uncovered ranges or recommend targeted search.

The existing read-saturation warning and critical directives must also become task-neutral. They
may tell the model to stop reading and use the information already available, but must not require
`write_file` or `edit_file` unless the task itself requires mutation.

## Components

### `ReadFileObservationTracker`

A focused state component under `agent/loop/state/` owns artifact insertion, eviction, invalidation,
and coverage decisions. It exposes semantic types rather than parsing tool calls in the main loop.
It is owned directly by per-task `TurnState`; it is not shared through `Arc` state.

The tracker models coverage as normalized inclusive interval sets and returns one of:

- `Execute`,
- `Replay { rendered_output, covered_intervals }`,
- `PartialOverlap { covered_intervals, uncovered_intervals }`, or
- `Unknown`.

The replay renderer is shared with `ReadFileTool` so physical and synthetic results use the same
header and line-number conventions.

### Existing Exact-Call Cache

The semantic tracker becomes the only content-replay cache for `read_file`. Successful `read_file`
results stop being inserted into the existing 8 KB exact-call cache. That cache remains unchanged
for `search_files` and other existing consumers. Oversized or semantically unknown `read_file`
results may still reach the generic repetitive guard, but that guard must not present a truncated
preview as complete file content.

### Tool Execution Integration

After hard policy and tool-budget validation, but before generic repetitive-call and consecutive
read guards, the tool execution phase:

1. asks the tracker for a pre-execution decision,
2. emits a synthetic tool message for replay or overlap guidance,
3. executes allowed calls normally,
4. retains complete typed metadata before conversation compression,
5. records successful `read_file` artifacts, and
6. performs path-specific invalidation after committed `write_file`/`edit_file` writes and
   whole-tracker invalidation immediately before dispatch when pre-execution `call_semantics` is
   `mutation` or `observation_and_mutation`; pure `observation` executions do not invalidate.
   Unknown semantics invalidate when tool capabilities permit mutation or external side effects.

Post-execution `result_metadata.semantics` may enrich telemetry and completion tracking, but it is
not the invalidation authority because transport errors and timeouts can return default metadata
after partial side effects.

Synthetic replay and overlap responses count as handled tool calls but not successful external tool
executions. They have the following accounting:

- count toward model tool attempts and hard tool budgets;
- count as logical iteration progress so they do not trigger no-progress fallback expansion;
- do not increment successful external execution counts;
- do not enter `recent_tool_calls`, `consecutive_same_tool`, or `recent_tool_names`;
- do not advance read-saturation counters; and
- do not emit a new direct-read evidence record.

The original valid physical-read evidence remains usable for edit/write evidence gates. Invalidation
removes semantic replay eligibility and the corresponding `FileRead` evidence. Path-specific
invalidation clears evidence for that path; whole-tracker invalidation clears all task-local
file-read evidence. Therefore a required post-mutation recheck performs a physical read and
establishes fresh evidence.

### Notices and Telemetry

Add distinct notices for:

- cached full-read replay,
- fully covered range replay, and
- partial overlap guidance.

Emit a dedicated semantic-read decision event containing the normalized path, decision kind,
requested interval, covered intervals, and uncovered intervals. Do not include file contents in
telemetry metadata.

## Error Handling

- Path normalization failure: allow the read and use existing tool validation.
- Malformed arguments: allow existing argument-contract handling to report the error.
- Missing or inconsistent typed result metadata: do not create a semantic artifact.
- Conflicting cached text for the same generation and line: execute the real read.
- Oversized result: skip semantic caching; no truncated artifact may be replayed as complete.
- Cache replay construction failure: execute the real read rather than blocking access.

## Testing

Unit tests will cover:

1. second identical full read replays immediately;
2. relative, `~`, symlink, and case-equivalent path aliases normalize to the same artifact where the
   platform resolves them to the same file;
3. a full read covers later bounded ranges;
4. a bounded range covers an inner range and replay returns only the inner range;
5. partial overlap returns all exact uncovered intervals, including uncovered prefixes and suffixes;
6. adjacent and non-overlapping ranges execute;
7. adjacent cached artifacts from one generation can satisfy a spanning request;
8. gapped or generation-mismatched artifacts cannot satisfy a spanning request;
9. conflicting cached line text falls back to physical execution;
10. tail reads use their returned concrete interval;
11. open-ended ranges are covered when continuous coverage reaches known EOF and remain partial
    otherwise;
12. empty files are cached as full coverage and binary files are not cached;
13. successful edit/write invalidates one path;
14. a state-mutating terminal/run-command/CLI-agent/spawn-agent execution clears the tracker even
    when it later reports failure, while a read-only (`observation`) terminal command does not clear
    it;
15. shell semantics classify output redirections such as `echo value > file`, `cat > file`, and
    `rg pattern file | tee output.txt` as state-mutating;
16. shell semantics classify every segment of `ls && rm file` and similar compound commands, with
    any mutating segment preventing an observation-only result;
17. quoted or escaped `>`, `|`, `&&`, `||`, and `;` characters do not create false mutation
    classifications;
18. ambiguous or unparseable shell expressions conservatively classify as mutation;
19. failed atomic edit/write operations preserve artifacts when no write was committed;
20. a later physical read with changed size or modified time replaces the older generation;
21. artifact count and byte budgets evict oldest entries;
22. oversized artifacts are not presented as complete cached results;
23. typed artifacts remain complete when context-window tool-result compression is enabled;
24. synthetic responses do not advance repetitive or read-saturation counters;
25. replay relies on existing evidence without recording a new physical-read evidence event, and
    artifact invalidation clears the matching file-read evidence;
26. notices and saturation directives are neutral for analysis-only tasks;
27. `read_file` no longer uses the exact-call content cache; and
28. generic repetitive guards remain unchanged for other tools.

An integration test will script a model that reads a resume, repeats the same read, then requests an
overlapping range. It will verify that only the first physical read executes, the complete artifact
is replayed, and overlap guidance directs the model to targeted search or every uncovered range.

A second integration test will enable context-window compression, read a file whose output exceeds
the configured tool-result limit, and then request a covered inner range. It will verify that the
conversation received compressed physical output, the tracker retained complete raw lines, and the
synthetic range replay is complete and exactly scoped.

A mutation integration test will read a file, modify it through `terminal` using an output
redirection or mixed command chain, and read it again. It will verify that pre-execution semantics
invalidate the tracker before dispatch, force a physical read, and prevent stale replay even if the
terminal command later reports failure. A paired read-only terminal command will verify that valid
artifacts are retained.

## Success Criteria

- A consecutive duplicate `read_file` call performs one physical file read, not two.
- Covered and overlapping ranges are recognized even when arguments differ.
- Covered range replay returns exactly the requested lines, including coverage assembled from
  contiguous artifacts of one generation.
- Cached replay never labels a truncated artifact as complete.
- Context-window compression does not truncate the task-local semantic artifact.
- Pre-execution semantics, including shell redirections and every segment of compound commands,
  prevent state-mutating executions from leaving replayable stale artifacts even when they later
  report failure; read-only executions do not discard valid artifacts.
- The agent is explicitly guided to `search_files` plus one exact range for unknown locations.
- Existing legitimate multi-file and non-overlapping range workflows continue to work.
- Formatting, Clippy with all features, and the full test suite pass before commit.
