# AIdaemon app: design standards

A workbench for working with an agent: quiet, precise and dense, with what the agent is doing always visible and never shouting. Content (the conversation, the work) carries the screen. Chrome stays out of the way.

## Targets

- **Desktop and web: one family.** Pointer and keyboard first; compact controls (26/32 pt); persistent sidebar and inspector; hover on everything pressable; tooltips on icon-only controls; shortcuts for frequent commands.
- **Terminal.** The same panes on whole cells; nsh maps the tokens to terminal colours by meaning. Every action has a key (Cmd/Ctrl+K palette, Cmd/Ctrl+N new, Cmd/Ctrl+B sidebar, Cmd/Ctrl+I activity, Esc stops). No radii or shadows are relied on to separate regions: hairlines and space do.
- **Mobile** (later): iOS and Android sizes and navigation (a stack per conversation, sheets for the inspector and palette).

## Tokens (`src/ui/theme.rs`)

- **Surfaces:** `BACKGROUND` (the window), `SIDEBAR` (side panes, one step away), `SURFACE` (cards, popups, the composer), `INSET` (code and tool output inside a surface). Washes: `HOVER` is weaker than `SELECTED`.
- **Text:** `TEXT`, `MUTED` (secondary), `FAINT` (metadata, hints). Never smaller than 11 pt.
- **One accent:** `ACCENT` (teal) means the agent: running work, the focus ring, the caret, links. `PRIMARY` is the one main action's fill (ink on paper, paper on graphite).
- **Status** (`Tone`): success, warning, danger, info, neutral, accent. Each has a colour and a soft tint, and always comes with a word or glyph, never colour alone.
- **Radii:** surfaces 12, controls 7, small parts 4. Nested shapes are concentric.
- **Type:** 11 / 12.5 / 13.5 / 14.5 (reading) / 17 (titles). Two weights per screen: regular and medium or semibold. Monospace only for code, commands, tool names and keys.
- **Sizes:** sidebar 248, inspector 320, conversation column 760, rows 30, icon slot 16.

## Components (`src/ui/controls.rs`)

`btn`, `btn_icon`, `icon_btn` (variants primary, secondary, ghost, outline, danger, success; sizes sm and md), `field`, `badge`, `status`, `kbd`, `row`, `surface`, `separator`, `section_label`, `empty`, `working`, `code`, `banner`, `hint` and `with_hint`. Features compose these and layout primitives only. A missing look becomes a variant here first.

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

## Copy

Say what happened and what to do next, in the owner's words: "The daemon is not running", "Retrying in 3s", "Allow once", "Deny". No internal identifiers in headings. Session channels and tool names show as metadata, in monospace where they are identifiers.
