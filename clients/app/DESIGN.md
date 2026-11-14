# AIdaemon app: design standards

A workbench for working with an agent: quiet, precise and dense, with what the agent is doing always visible and never shouting. Content (the conversation, the work) carries the screen. Chrome stays out of the way.

## Targets

- **Desktop and web: one family.** Pointer and keyboard first; compact controls (26/32 pt); persistent sidebar and inspector; hover on everything pressable; tooltips on icon-only controls; shortcuts for frequent commands.
- **Terminal.** Its own conventions, not the window's: a key line at the bottom names the keys for what is on screen; Ctrl+C has one meaning per state (stop the turn; otherwise press twice to quit, with a note after the first); Ctrl+Y copies the last reply; copies reach the system clipboard over OSC 52 (works over SSH and tmux with `set-clipboard on`); Alt+Enter adds a line (terminals rarely send Shift+Enter); the palette (Ctrl+K) holds every command, including answering each waiting approval. Drafts, pins and the open conversation persist in `~/.aidaemon/app-terminal`. The same panes on whole cells; nsh maps the tokens to terminal colours by meaning. No radii or shadows are relied on to separate regions: hairlines and space do.
- **Shortcuts while typing.** nsh keeps editing keys (Esc, Cmd/Ctrl+C, Y, Z...) for a focused input unless a binding uses `KeyOptions { when: When::Always }`. App-wide keys that must work from the composer (Esc stops, Copy reply, the terminal's Ctrl+C) use it; Esc also passes through.
- **Touch (phones, tablets, mobile browsers):** `ui::touch()` switches the size scale (`ui::fit`): controls and rows 44–52 pt, reading text 16 pt (mobile browsers zoom into smaller fields), icons 18 pt in 22 pt slots, 52 pt bars. A narrow window shows one thing at a time: the conversation, or the conversations or activity pane full-screen, reached from the header. Return adds a line; Send sends.
- **Accessibility:** every control has a role and a name (text buttons name themselves; icon buttons take a label, which is also their tooltip and their terminal text). Decorative glyphs and dots are hidden from screen readers. Pane, conversation and card titles are headings. A conversation row reads as one item ("title, place, working"). Banners use the alert role.

## Tokens (`src/ui/theme.rs`)

- **Surfaces:** `BACKGROUND` (the window), `SIDEBAR` (side panes, one step away), `SURFACE` (cards, popups, the composer), `INSET` (code and tool output inside a surface). Washes: `HOVER` is weaker than `SELECTED`.
- **Text:** `TEXT`, `MUTED` (secondary), `FAINT` (metadata, hints). Never smaller than 11 pt.
- **One accent:** `ACCENT` (teal) means the agent: running work, the focus ring, the caret, links. `PRIMARY` is the one main action's fill (ink on paper, paper on graphite).
- **Status** (`Tone`): success, warning, danger, info, neutral, accent. Each has a colour and a soft tint, and always comes with a word or glyph, never colour alone.
- **Radii:** surfaces 12, controls 7, small parts 4. Nested shapes are concentric.
- **Type:** 11 / 12.5 / 13.5 / 14.5 (reading) / 17 (titles). Two weights per screen: regular and medium or semibold. Monospace only for code, commands, tool names and keys.
- **Sizes:** sidebar 248, inspector 320, conversation column 760, rows 30, icon slot 16.

## Components (`src/ui/controls.rs`)

`meter`, `key_line`, `btn`, `btn_icon`, `icon_btn` (variants primary, secondary, ghost, outline, danger, success; sizes sm and md), `field`, `badge`, `status`, `kbd`, `row`, `surface`, `separator`, `section_label`, `empty`, `working`, `code`, `banner`, `hint` and `with_hint`. Features compose these and layout primitives only. A missing look becomes a variant here first.

## Motion (`src/ui/motion.rs`)

Tokens: `FAST` 120 ms (hover and colour), `BASE` 180 ms (press, toggle), `ENTER` 260 ms (arriving rows, panels); press scale 0.97 (icon buttons 0.94). Reduced motion is the platform's: transitions take their end state.

### Motion inventory

| Screen | Change | Tool |
|---|---|---|
| All | Hover on buttons and rows | `transition_colors`, `FAST`, ease-out |
| All | Press on buttons | `active` scale to `PRESS_SCALE`, `FAST` |
| Composer | Focus | Border to `ACCENT`, `transition_colors` |
| Composer | Grows with its text | Height bound to content size, clamped 22–180 |
| Conversation | New rows arrive | `follow_end` keeps the end in view while the owner is at it |
| Conversation | Reply streams | Markdown `streaming`: only the last block is remade |
| Conversation | Agent working | Pulse on the status dot (`animate_pulse`) |
| Inspector, sidebar | Panes fold at narrow widths | Hidden by style, state kept |
| Palette | Opens | Native dialog presentation |
| Conversation | Agent offers a view move while you type | Banner (accent tint) with Go (Cmd/Ctrl+G) and Dismiss; nothing moves until taken |
| Conversation | Agent moved the view | Neutral note banner, leaves after 6 s |
| Approval card | Brought up by a move | Border thickens to accent until answered |
| System | Approval arrives while the app is in the background | System notification (closed when answered anywhere); waiting count on the app icon; pressing it shows the card |

## Copy

Say what happened and what to do next, in the owner's words: "The daemon is not running", "Retrying in 3s", "Allow once", "Deny". No internal identifiers in headings. Session channels and tool names show as metadata, in monospace where they are identifiers.
| Conversation | Reading back while replies arrive | "Jump to latest" pill (bottom, centred) when the last row is out of view; follow-end only while at the end |
| Conversation | Stop pressed | Stop becomes "Stopping…" (warning status) until the turn ends |
| Sidebar | A reply lands elsewhere | Filled dot and full-strength title until opened; pinned (★) first; age at the right |
| Inbox | Work running | Elapsed time ticks each second; long tool output shows its last 8 lines with Show all / Copy |
| System | Open | Reads on open and every 30 s; "Needs a look" (warning tint, alert role) leads when anything fails |
| System | A reply finishes while the app is in the background | "Done: <conversation>" notification; pressing it opens the conversation |
| Conversation | A reply's turn used tools | One quiet disclosure line above the reply ("Used 4 tools: read_file ×2, … · 12s"); ▸ turns ▾ and lists every tool in order. The palette's "Tool activity" cycles hidden / one line / every tool (kept per device) |
| Conversation | Agent working, tools used | The working line adds "· N tools so far" |
| Composer | Typing while the agent works | Send becomes "Add to the current work" (Enter); Queue (Mod+Enter) waits for its own turn; the hint names both and Esc |
| Composer | Draft starts with `/` | The daemon's matching commands above the composer; Tab or a press completes the first |
| Composer | Up on an empty composer | The last message sent here comes back to edit |
| Conversation | Last turn failed or was stopped | Retry under it sends the same message again |
| Sidebar | A conversation waits on an approval, or works | "Needs you" first, then "Working", then the others |
| Sidebar | Searching | Titles filter at once; the daemon's matches in every conversation follow under "In messages" with a snippet, who said it and when |
| System | A goal ran in the last day | "While you were away": each run with Done, Running, Failed, Stopped or "No report" (warning: finished without saying what it did); failed and silent runs also go to "Needs a look" |
| Terminal | Quit | The open conversation prints to the normal scrollback as Markdown |

## Areas

- **Conversation:** day breaks ("Today", "Yesterday", weekday, date), times under messages, Copy on every reply, Copy conversation as Markdown and Pin in the header.
- **Inbox** (the right pane): waiting approvals of other conversations, new replies, running work, and the current turn's steps.
- **Memory:** what the daemon remembers about the owner, newest first, filterable, each with Forget.
- **System:** daemon version, uptime and models; what needs a look; token use per day; model call times; scheduled goals; background jobs and queues; health checks.

Times are shown in the owner's zone (`ui::time`): the browser's offset on the web, the system's elsewhere.
