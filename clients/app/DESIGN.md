# AIdaemon app: design standards

A workbench for working with an agent: quiet, precise and dense, with what the agent is doing always visible and never shouting. Content (the conversation, the work) carries the screen. Chrome stays out of the way.

## Targets

- **Desktop and web: one family.** Pointer and keyboard first; compact controls (26/32 pt); persistent sidebar; the details pane (Changes and Steps) beside a coding conversation, a key away beside any other; hover on everything pressable; tooltips on icon-only controls; shortcuts for frequent commands.
- **Terminal.** Its own conventions, not the window's: a key line at the bottom names the keys for what is on screen; Ctrl+C has one meaning per state (stop the turn; otherwise press twice to quit, with a note after the first); Ctrl+Y copies the last reply; copies reach the system clipboard over OSC 52 (works over SSH and tmux with `set-clipboard on`); Alt+Enter adds a line (terminals rarely send Shift+Enter); the palette (Ctrl+K) holds every command, including answering each waiting approval. Drafts and pins persist in `~/.aidaemon/app-terminal`. A coding tool first, not the window shrunk: the conversation fills the terminal and the list (Ctrl+B) and details (Ctrl+O, since a terminal sends Ctrl+I as Tab; Ctrl+D for Changes) open on a key; run in a project's folder (or below it), it starts on a new conversation in that project, as a command-line agent does. Whole cells; nsh maps the tokens to terminal colours by meaning. No radii or shadows are relied on to separate regions: hairlines and space do.
- **Shortcuts while typing.** nsh keeps editing keys (Esc, Cmd/Ctrl+C, Y, Z...) for a focused input unless a binding uses `KeyOptions { when: When::Always }`. App-wide keys that must work from the composer (Esc stops, Copy reply, the terminal's Ctrl+C) use it; Esc also passes through.
- **Touch (phones, tablets, mobile browsers):** `ui::touch()` switches the size scale (`ui::fit`): controls and rows 44–52 pt, reading text 16 pt (mobile browsers zoom into smaller fields), icons 18 pt in 22 pt slots, 52 pt bars. A narrow window shows the conversation; the conversations and activity panes are drawers over it, from the left and the right, reached from the header. Keyboard hints are not shown. Return adds a line; Send sends.
- **Insets:** the window's root pads by the safe area (notch, status bar, home indicator: a phone browser in full screen or an installed web app) and its bottom by the larger of the home indicator and the on-screen keyboard, so the composer and the end of the conversation stay above the keyboard. Drawers pad top and bottom by the safe area.
- **Accessibility:** every control has a role and a name (text buttons name themselves; icon buttons take a label, which is also their tooltip and their terminal text). Decorative glyphs and dots are hidden from screen readers. Pane, conversation and card titles are headings. A conversation row reads as one item ("title, place, working"). Banners use the alert role.

## Tokens (`src/ui/theme.rs`)

- **Surfaces:** `BACKGROUND` (the window), `SIDEBAR` (side panes, one step away), `SURFACE` (cards, popups, the composer), `INSET` (code and tool output inside a surface). Washes: `HOVER` is weaker than `SELECTED`.
- **Edges:** `BORDER` frames cards and buttons that name themselves; `EDGE` (3:1 or more on the page and on a surface) frames what is typed in or chosen from (fields, selects, the composer) and what floats (popups, menus, the palette): in dark a shadow does not show, the edge does. `theme.rs`'s test holds text at 4.5:1 and edges at 3:1 in both themes.
- **Text:** `TEXT`, `MUTED` (secondary), `FAINT` (metadata, hints; at least 4.5:1 on the background in both themes). Never smaller than 11 pt.
- **Icons:** `ui::icon(Glyph)`: Lucide line art (ISC) drawn in the text colour in a window and a browser; the plain symbol in a terminal.
- **One accent:** `ACCENT` (teal) means the agent: running work, the focus ring, the caret, links. `PRIMARY` is the one main action's fill (ink on paper, paper on graphite).
- **Status** (`Tone`): success, warning, danger, info, neutral, accent. Each has a colour and a soft tint, and always comes with a word or glyph, never colour alone.
- **Radii:** surfaces 12, controls 7, small parts 4. Nested shapes are concentric.
- **Type:** 11 / 12.5 / 13.5 / 15.5 (reading) / 20 (titles); on touch 13 / 15 / 16 / 22. The OS text size scales all of it (nsh multiplies every font and line height by `font_scale()`, the browser's default size on the web), so controls hold their text with `min_h`, not a fixed height. Digits that change in place (times, ages, counts, elapsed) are tabular (`tabular_nums`). A phone's titles are 22 (`fit::title`), clear of its 16 pt reading text. Two weights per screen: regular and medium or semibold. Monospace only for code, commands, tool names and keys.
- **Sizes:** sidebar 280, inspector 320, conversation column 760, rows 30, icon slot 16.

## Components (`src/ui/controls.rs`, `src/ui/kit.rs`)

`key_line`, `segmented` (two or three ways, all in view: a track with the chosen one raised), `btn`, `btn_icon`, `icon_btn`, `icon_btn_primary` (the one main action as a round filled glyph) (variants primary, secondary, ghost, outline, danger, success; sizes sm and md), `field`, `secret_field` (masked), `picture` and `file_chip` (files with messages), `badge`, `status`, `kbd`, `row`, `surface`, `separator`, `section_label`, `empty`, `working`, `code`, `banner`, `hint` and `with_hint`. Over `nsh_base`'s behaviour-only parts (`kit.rs`; keyboard, focus, roles and the terminal's rendering are nsh's): `toggle` (Switch: a setting that takes effect at once), `choice` (Select; searchable past 8 items; a sheet on phones), `sections` (Accordion, several open), `tab_bar` (Tabs), `data_table` (Table semantics, actions in the last column), `meter` (Progress), `confirm` (the platform's alert, before anything that cannot be undone), `with_menu` (ContextMenu: right click, long press, the menu key), `press_menu` (Menu: a glyph that opens one when pressed; a sheet on phones), `pick_list` / `pick_dialog` (Command in a Sheet, Popover or Dialog: one choice among named things with a line each, sections, search, checks and an action row; for projects, models, and anything that grows. `choice` stays for a few short words in a settings form, and on a phone opens the same kind of sheet: dimmed page, rounded top, tall rows, the chosen one washed), `toasts` (ToastStack and ToastManager: one stack for the app; the terminal's status line), and `split` / `pane` / `pane_rest` (resizable panes).

Over `nsh_base`'s larger views (`rich.rs`): `diff` (DiffView: one file's unified diff, line numbers, words changed, unchanged lines folded, highlighted by the file's name, side by side on request; only rows in view are made), `output` (Terminal: a program's output with its colours, scrolled back, following the end), `code_editor` (Editor: TOML and other code, growing to a cap), `day_bars` (plot: two stacked numbers per day, hover names the day), `file_tree` (Tree: files under folders, a folder of one folder joined), `suggest_field` (Combobox: free text with suggestions) and `with_preview` (HoverCard: more about a row while the pointer rests on it; never on touch or in a terminal).

Features compose these and layout primitives only. A missing look becomes a variant here first.

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
| Inspector, sidebar | Owner drags the line between panes | `split` (resizable panels; sidebar 200–420, inspector 260–520) |
| All | A page replaces another (conversation, Settings, Memory, System) | `motion::next_layout`: fade, `BASE` |
| Phone | Sidebar or inspector opens or closes | `ui::drawer` (nsh-base Sheet): slides in from its own edge (sidebar left, inspector right) over a dimmed page (`SCRIM`), back out on close; a tap on the dimmed page, Escape or back closes it; in place with reduced motion |
| Desktop | Sidebar or inspector toggles | At once: the resizable group keeps measured widths, so it is not animated |
| Conversation, Settings | Tools pill, sections unfold | `motion::next_unfold` (content fades in, what is below moves); sections `reveal` |
| Sidebar | Row menu opens and closes | Grows from where it was asked for (`opacity-0 scale-95`), `BASE` |
| All | A short note ("Copied", a setting saved, the view moved) | Toast at the bottom, `ENTER` in, leaves after 6 s, paused while hovered; Escape or a swipe dismisses |
| Settings | A change is refused | Danger banner that stays until dismissed or the next change |
| Settings, Memory | Remove, Forget, Restart | The platform's alert asks first |
| Sidebar | Right click or long press on a conversation | Menu: Open, Pin/Unpin, Stop its work (only while it works), Move to project… (the project picker over the page: No project first, then the projects with their folders, recent first past four; a phone's list closes for it), Rename (the row becomes a name field: Save, Cancel; empty goes back to the name from what was said), Archive / Bring back |
| Sidebar | Archived conversations | Out of every list; "Archived · N" at the end of the list opens them, "Back to your conversations" returns |
| Settings | True/false settings | A switch in the row, applied at once |
| Palette | Opens | Native dialog presentation |
| Conversation | Agent offers a view move while you type | Banner (accent tint) with Go (Cmd/Ctrl+G) and Dismiss; nothing moves until taken |
| Conversation | Agent moved the view | Toast, leaves after 6 s |
| Approval card | Brought up by a move | Border thickens to accent until answered |
| System | Approval arrives while the app is in the background | System notification (closed when answered anywhere); waiting count on the app icon; pressing it shows the card |

## Copy

Say what happened and what to do next, in the owner's words: "The daemon is not running", "Retrying in 3s", "Allow once", "Deny". No internal identifiers in headings. Session channels and tool names show as metadata, in monospace where they are identifiers.
| Conversation | Reading back while replies arrive | "Jump to latest" pill (bottom, centred) when the last row is out of view; follow-end only while at the end |
| Conversation | Stop pressed | Stop becomes "Stopping…" (warning status) until the turn ends |
| Sidebar | A reply lands elsewhere | Filled dot and full-strength title until opened; pinned (★) under Pinned |
| Steps | Work running | Elapsed time ticks each second |
| Sidebar | Home | One list, top to bottom: Needs you (from every project, only when any), Pinned, Projects (up to four, one line each: folder, name, count, chevron, then New project; past four, the three used last and All projects), then every other conversation under Recent, newest first, with no day headings and nothing held back: each row ends with when it last moved ("now", "5m", "3h" within a day, "18h" for last evening, then its date: "Nov 12"; with the year when not this one). A row is one line: a filled dot only for a new reply, its title, one faint tag where it adds something (its project, or "Telegram" / "Telegram group" when not from the app). Sources are one filter ("All ▾" beside Recent, the picker), shown only with more than one source. A phone's top line is Search, New conversation (one, a compose glyph) and Close |
| Sidebar | A project | A press on a project goes inside it (not a fold): Back, its name, "…" (Edit name or folder), its folder, its conversations newest first, each with its time. New conversation starts one there. Its menu (right click, long press): New conversation here, Edit name or folder. All projects lists every one with its folder, then New project (the form opens there) |
| Conversation | In a project | The header's second line names the project and its folder (a new conversation's project is picked on its page instead); the phone's + starts the next one in the same project |
| Conversation | New | The header bar holds the switch, Chat or Code, where a title will be. Both ways have one shape, a question over a room of one height, so switching changes the words and nothing moves. Chat: "What can I help with?", the room empty. Code: "Which project is this for?" over a card naming the project and its folder; a press opens the projects (`ui::pick_list`: a sheet from the bottom on a phone, a popover at the card in a window) under "Choose a project": past four, Recent (the three last worked in) then All projects by name, each with its folder and a check on the chosen one, a search past six, and New project last (the sidebar's form; a made project becomes the choice). It starts on the project this device last worked in; with none, nothing is chosen and Send waits ("Choose a project first"). With no project yet, New project in the list's place. No starters. It sits by the composer on a phone, centred in a window. The details button waits for a conversation; a phone's Conversations button is a speech bubble, not a second pane picture |
| Composer | Model and effort | One chip in the tools' row, after attach and the microphone ("Auto", or the model and "· High" when an effort is set); a press opens one picker: Model (each with its provider under it when there are several; Auto first) and Effort (only for a model that takes one), what is in effect checked. On a phone it gives the row to the working hint while the agent works |
| Composer | Send | A round filled arrow, no word (the word stays in a terminal); a phone's placeholder drops the hint about commands |
| Composer | Placeholder | Says what to write: "Ask anything, or tell it what to do" (new chat), "Describe the change, fix or task" (new code), "Choose a project first", "Say what to do with the marked part" (a marked-up picture attached), "Say what to do with it, or just send" (other files attached), "Reply" (ongoing), "Add to the work or queue what's next" (while it works); a window adds "(/ for commands)" |
| Conversation | Pictures | A press opens the picture viewer over the page (`ui::picture_viewer`): near-black backdrop, the picture as large as fits, its name, then Mark up, Share, Download (the web) or Open in browser (a window) and Close; a press on the picture or backdrop, Escape or back closes it. Share is the system's share sheet with the file (the web, where the browser has one; else it downloads). Mark up (the web, in the app's own conversations) draws red strokes with a finger or the mouse, with Undo, Cancel and Add to message, which puts `<name>-marked.jpg` on the message being written, uploaded with `marked_from` (the original's id), so the daemon tells the agent the red marks are the owner's and where the unmarked picture is |
| Composer | Attach | The paperclip opens a menu (a sheet on a phone): Take a picture (a phone's or tablet's browser, which opens the camera), Pictures, Files |
| All | The daemon refuses the sign-in | The connection says so (it signs in again, or reads as refused); no note is left on the page |
| Details | A conversation works on code (in a project with a folder, wrote a file, or has changes) | It gets the Changes tab and opens on it; on a wide window the details show beside it, and hide beside any other; a turn's first file write moves to Changes once per turn; on a phone the header's "Changes · N" opens them |
| Changes | Read | The last command run after the last change leads: Passed, Failed (exit N), or "Not checked" when files changed and nothing ran since |
| Conversation | A chat's conversation (Telegram, Slack, Discord) | The header names the platform and bot ("· group" when others are in it); the composer says "Also posted in …": what is sent here is posted there too |
| Conversation | A group chat | The composer is replaced by a note that the bot posts what is sent here for everyone, and "Post in this group"; once pressed, the composer shows for this conversation |
| Automations | Open | Reads on open, every 30 s and on Refresh; Cancel turns into "Cancel it for good?" with Yes, cancel / Keep it; History swaps for Hide history and opens the runs under the card |
| System | Chat bots | Each bot with its conversation count and its project (a small folder chip opening the project picker, No project first): its direct messages work in that project's folder unless moved; group chats join a project only when moved there |
| Changes | A tool that can change files finishes | The changes read again while the tab shows (else on its next showing); the files list as a tree under their folders; a press on a file fills the pane with its diff (`ui::diff`: line numbers, changed words, folded context that expands, highlighted by its name), with Previous/Next change, Side by side, Copy and All files |
| System | Open | Reads on open and every 30 s; "Needs a look" (warning tint, alert role) leads when anything fails |
| All | A reply finishes, or an approval arrives, in another conversation while the app is in front | Toast: "Done: <conversation>" or "Needs you in <conversation>: <what>" |
| System | A reply finishes while the app is in the background | "Done: <conversation>" notification; pressing it opens the conversation |
| Composer | Files dragged from the desktop | The border turns accent and says "Drop to attach"; dropped files attach as the paperclip's do (a folder or unreadable file says so) |
| Composer | Microphone (where the platform recognizes speech) | Speech writes into the composer after what is there as it is heard; the button becomes Stop listening (accent); a refusal says what to allow |
| Steps | A tool's output | Shown in a terminal (`ui::output`): colours kept, 8 rows or 24 with Show more, scrolled back for the rest, Copy |
| Settings | A value is edited | A TOML code editor (highlighted, brackets closed); Another setting suggests the known keys as its key is typed |
| System | Token use | Stacked bars per day (in grey, out teal), oldest at the left; the day under the pointer is named under the chart |
| Sidebar | Pointer rests on a conversation (desktop, web) | A card after 600 ms: title, where it is and when it moved, its last message |
| Conversation | A reply's turn used tools | One quiet disclosure line above the reply ("Used 4 tools: read_file ×2, … · 12s"); ▸ turns ▾ and lists every tool in order. The palette's "Tool activity" cycles hidden / one line / every tool (kept per device) |
| Conversation | Agent working, tools used | The working line adds "· N tools so far" |
| Composer | Typing while the agent works | Send (Enter) tells the running work now; Queue (Mod+Enter) sends after it; the hint says "Enter tells it now · ⌘↩ sends after · Esc stops" |
| Composer | Messages wait | "Waiting to send" above the composer: each marked Now (the running work reads it at its next step) or After (its own turn when the work ends), with Edit and Remove until the agent takes it |
| Settings, General | On this device: what Enter does while the agent works | "Tells it now (⌘↩ sends after)" or "Sends after it (⌘↩ tells it now)"; the composer's hint and second button (Queue or Now) follow |
| Changes | Undo a file | "Undo changes" under a file's diff header asks first (a new file is deleted, a changed one goes back to its last commit), then the list comes back read again |
| Automations | Runs since the last look | "N new" on the Automations row; "New since you last looked" on each card whose last run reported something or failed |
| Composer | Draft starts with `/` | The daemon's matching commands above the composer; Tab or a press completes the first |
| Composer | Up on an empty composer | The last message sent here comes back to edit |
| Conversation | Last turn failed or was stopped | Retry under it sends the same message again |
| Sidebar | A conversation waits on an approval, or works | Its row's end says so in place of its time: an amber dot and "Needs you" (and the conversation leads the list under Needs you), or a small accent ring that turns (`ui::spinner`; still with reduced motion) while it works, keeping its place by time. A phone's (or a folded sidebar's) list button carries a corner dot when another conversation needs the owner (amber) or works (accent) |
| Sidebar | Searching | Titles filter at once; the daemon's matches in every conversation follow under "In messages" with a snippet, who said it and when |
| System | A goal ran in the last day | "While you were away": each run with Done, Running, Failed, Stopped or "No report" (warning: finished without saying what it did); failed and silent runs also go to "Needs a look" |
| Terminal | Quit | The open conversation prints to the normal scrollback as Markdown |
| Settings | A change is sent | The daemon's answer stays as a note at the top until dismissed; guarded changes say they wait for approval (or for another device) and a banner counts them |
| Settings | Saved changes not yet in effect | Info banner with Reload, warning banner with Restart |
| First run | No daemon to sign in to | "Where should your agent run?": Run it here (status while it starts) or a pairing link for one elsewhere |
| Settings | Tabs | General (a short curated list: switches, choice lists, fields with Save), Models (providers, then the decision model), Connections, Advanced (every setting by table in plain words with its key beneath, the database, Reload/Restart, Undo) |
| Settings | Decision model | Backend list (Off, your chat model, Gemini, TypeSafe Jev) with a sentence on what each sends where; model and key fields only for the backends that take them; each decision's Off / Watch only / On |
| Settings | Providers | "In use" lists the main provider and each failover in order, each with its models and a status dot for its key or ChatGPT sign-in; Add/Replace key opens a masked field under its row; Remove asks first. "Add a provider" opens a form for the provider picked: a key with Check key (then the account's models as a list), or Sign in with ChatGPT (a card with the page's link, and a paste field when the browser is on another device); Make it the main provider and Add as failover send the key with the change |
| First run | Connect a model | Found models first with Use it; every provider says how it bills; a key form opens under its row; "Testing" while the test call runs; the daemon's refusal as a danger banner |
| First run | Model connected | "Connected" with a working line until the daemon restarts with it |
| First run | Chat step | Platform choices, then the token form; after connecting, "Waiting for your first message" until the bot has answered; Skip for now always |
| Conversation | The daemon offers prompts under a reply | Outline buttons above the composer until the owner says something |
| Conversation | The agent asks for a feature that is off | Info banner with Turn it on (or Set it up for voice) and Not now |
| Approval card | The first one on this device | One line saying what is asked about and what each answer does |

## Areas

- **First run:** where the agent runs, then the daemon's own setup steps (a model, then an optional chat), then the greeting's conversation opens.

- **Files:** the composer's paperclip offers the camera (phone and tablet browsers), pictures (both scaled to fit 2048 px) and files of any type as they are, and a pasted screenshot is attached; each shows as a chip ("attaching…", "failed") with a remove button until sent. Sent and received pictures show in place (up to 320 pt wide), other files as chips; pressing either opens it. Only the app's own conversations take files; the terminal shows chips, not pictures.
- **Conversation:** day breaks ("Today", "Yesterday", weekday, date), times under messages, Copy on every reply, Copy conversation as Markdown and Pin in the header.
- **Projects** (sidebar): a name and any folder on the daemon's computer (a site, a repository, designs, or none); a conversation in one works in its folder.
- **Steps** (the details pane): the open conversation's running work and its current turn's steps. What waits on the owner across conversations is the sidebar's Needs you.
- **Changes** (the details pane of a coding conversation): whether the last change was checked, then the uncommitted changes in each git repository the open conversation worked in: name, branch, files changed with +/−; files this conversation wrote first, marked with an accent dot; each file's diff on press. Changes by the owner or other sessions in the same tree show too, unmarked.
- **Automations:** everything that runs on its own (scheduled work and reminders, watching automations, health checks), next to run first: name, kind and cadence, next run, a badge when it is not simply on (Paused, Failed, Paused after failures, Waiting for you, Not confirmed yet), its last run (when, how it ended, what it said; red when it failed) or "It hasn't run yet.", and Pause or Resume (Try again when it failed), Run now (scheduled work), History (its last five runs, each with its end and what it said) and Cancel (asked again before it happens). Details opens its whole card (what it may do, when it tells you, when it ends, workspace, budget) and its last five runs. Edit, for scheduled work, opens its name, when it runs (in words), what each run does and Tell me (every run, only when something changes, only about problems); Save waits for a change, and the timed refresh waits while the form is open. Watching automations and health checks offer Change in chat. Paused after failures offers only Try again: its next good run turns it back on. Silence is never the only signal that something broke. The answer to a change brings the list and a note of what happened. New ones, and changes to when one runs, are asked for in a conversation: New starts one. The System page no longer lists schedules.
- **Memory:** what the daemon remembers about the owner, newest first, filterable, each with Forget.
- **Settings:** every setting of the daemon by config table (folded; filtering opens them), each with its value or "default (example: …)", "asks" when it needs approval, and when it takes effect; Edit in place as TOML, booleans as Turn on/off, secrets stored masked and never shown; any other key by path; the providers in use with their keys or sign-in, and adding one as main or failover; runtime bots and MCP servers; moving the database and rotating its key; Reload and Restart.
- **System:** daemon version, uptime and models; what needs a look; the chat bots and their projects; token use per day; model call times; scheduled goals; background jobs and queues; health checks.

Times are shown in the owner's zone (`ui::time`): the browser's offset on the web, the system's elsewhere.
