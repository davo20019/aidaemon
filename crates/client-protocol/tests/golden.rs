//! Golden fixtures: one `.golden` file per frame, command, result and event
//! variant. Each sample must serialize to exactly its fixture, and each
//! fixture must read back and serialize to itself. The exhaustive matches
//! below fail to compile when a variant is added without a sample.
//!
//! `UPDATE_GOLDEN=1 cargo test -p aidaemon-client-protocol --test golden`
//! rewrites the fixtures from the samples.

use std::path::PathBuf;

use aidaemon_client_protocol::*;
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn approval() -> Approval {
    Approval {
        id: "a-7f3e-1".into(),
        session_id: "app:6c1f".into(),
        kind: ApprovalKind::Command,
        risk: Risk::High,
        summary: "Your assistant wants to post the synthetic report to h.example.invalid".into(),
        detail: Some("POST https://h.example.invalid/report".into()),
        card: Some(ApprovalCard {
            asked_by: Some("Report helper".into()),
            run: None,
            action: CardAction::Use {
                what: "send a request to h.example.invalid".into(),
                detail: Some("{\"method\":\"POST\"}".into()),
            },
            reason: Some(AskReason::ReadOutsideContent {
                check_says: Some("The page asked for this, not you.".into()),
            }),
            impact: Some(Impact::Sends),
        }),
        warnings: vec!["Sends data outside this computer.".into()],
        options: vec![
            ApprovalDecision::ApproveOnce,
            ApprovalDecision::ApproveAlways,
            ApprovalDecision::Deny,
        ],
        requested_at_ms: 1_767_225_600_000,
        expires_at_ms: Some(1_767_225_900_000),
        channel: Some("telegram".into()),
    }
}

fn task() -> TaskSummary {
    TaskSummary {
        id: "b9e1-4".into(),
        session_id: "app:6c1f".into(),
        number: 4,
        description: "Working on your message".into(),
        state: TaskState::Running,
        started_at_ms: 1_767_225_600_000,
        finished_at_ms: None,
        error: None,
    }
}

fn session() -> SessionSummary {
    SessionSummary {
        group: Default::default(),
        archived: false,
        agent: None,
        id: "app:6c1f".into(),
        channel: "app".into(),
        title: Some("Synthetic report for Alice Rivera".into()),
        last_activity_ms: Some(1_767_225_600_000),
        writable: true,
        project_id: Some("project-acme".into()),
    }
}

fn project() -> ProjectSummary {
    ProjectSummary {
        channels: vec!["telegram:synthetic_coding_bot".into()],
        id: "project-acme".into(),
        name: "Acme site".into(),
        path: Some("/home/synthetic/acme-site".into()),
    }
}

fn handoff() -> Approval {
    Approval {
        id: "a-7f3e-2".into(),
        session_id: "app:6c1f".into(),
        kind: ApprovalKind::Handoff,
        risk: Risk::Low,
        summary: "Sign in to login.acme.example, then press Done.".into(),
        detail: None,
        card: Some(ApprovalCard {
            asked_by: None,
            run: None,
            action: CardAction::TakeOver {
                surface_id: "srf_7d1c".into(),
                what: "Sign in to your Acme account".into(),
                site: Some("login.acme.example".into()),
                keeps_sign_in: Some(true),
            },
            reason: None,
            impact: None,
        }),
        warnings: vec![],
        options: vec![ApprovalDecision::ApproveOnce, ApprovalDecision::Deny],
        requested_at_ms: 1_767_225_600_000,
        expires_at_ms: None,
        channel: None,
    }
}

fn click_on_site() -> Approval {
    Approval {
        id: "a-7f3e-3".into(),
        session_id: "app:6c1f".into(),
        kind: ApprovalKind::Command,
        risk: Risk::Medium,
        summary:
            "Your assistant wants to click a button on https://shop.acme.example in the browser."
                .into(),
        detail: Some("button#checkout".into()),
        card: Some(ApprovalCard {
            asked_by: None,
            run: None,
            action: CardAction::ClickOnSite {
                site: "https://shop.acme.example".into(),
                target: Some("button#checkout".into()),
            },
            reason: None,
            impact: None,
        }),
        warnings: vec![],
        options: vec![
            ApprovalDecision::ApproveOnce,
            ApprovalDecision::ApproveSession,
            ApprovalDecision::Deny,
        ],
        requested_at_ms: 1_767_225_600_000,
        expires_at_ms: None,
        channel: None,
    }
}

fn surface() -> SurfaceInfo {
    SurfaceInfo {
        id: "srf_7d1c".into(),
        kind: SurfaceKind::BrowserTab,
        session_id: Some("app:6c1f".into()),
        title: "Sign in - Acme".into(),
        site: Some("login.acme.example".into()),
        width: 1280,
        height: 720,
        controller: SurfaceController::Owner {
            device_id: "dev-phone-1".into(),
            device: Some("Synthetic phone".into()),
        },
        handoff_pending: true,
        interactive: true,
        opener: None,
    }
}

fn shell() -> ShellInfo {
    ShellInfo {
        id: "sh_4b2e".into(),
        session_id: "app:6c1f".into(),
        folder: "/home/synthetic/acme".into(),
        title: "acme".into(),
        cols: 100,
        rows: 30,
        exit_code: None,
        started_at_ms: 1_767_225_600_000,
    }
}

fn snapshot() -> Snapshot {
    Snapshot {
        sessions: vec![session()],
        running_tasks: vec![task()],
        pending_approvals: vec![approval(), handoff(), click_on_site()],
        live_turns: vec![LiveTurnState {
            session_id: "app:synthetic-1".into(),
            turn_id: "turn-7".into(),
            started_at_ms: 1_700_000_000_000,
            streamed: "Reading the synthetic notes".into(),
            status: Some(TurnStatus::tool_started(
                "terminal",
                "cargo test -q",
                Some(ToolKind::Run),
            )),
            tools: vec!["terminal".into()],
            calls: vec![ToolCallInfo {
                name: "terminal".into(),
                input: "cargo test -q".into(),
                output: String::new(),
                kind: Some(ToolKind::Run),
            }],
        }],
        surfaces: vec![surface()],
        shells: vec![shell()],
        runs: vec![run()],
    }
}

fn run() -> RunSummary {
    RunSummary {
        id: "run-3f2a".into(),
        session_id: "app:6c1f".into(),
        label: "Claude Code: prepare the synthetic release".into(),
        agent: Some("Claude Code".into()),
        phase: RunPhase::Running,
        background: true,
        step: "Running a command".into(),
        started_at_ms: 1_767_225_600_000,
        ended_at_ms: None,
        took: None,
        reason: None,
        stopped_by_owner: false,
        group: Some("task-9b1d".into()),
        edited: vec!["/w/synthetic/release.md".into()],
        continues: None,
        thread: None,
        can_continue: false,
        result: None,
        asked: None,
    }
}

fn client_name(frame: &ClientFrame) -> String {
    match frame {
        ClientFrame::Hello { .. } => "client_hello".into(),
        ClientFrame::Command { command, .. } => format!("client_command_{}", command_name(command)),
        ClientFrame::Ping => "client_ping".into(),
    }
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::SendMessage { .. } => "send_message",
        Command::StopTurn { .. } => "stop_turn",
        Command::ResolveApproval { .. } => "resolve_approval",
        Command::CancelTask { .. } => "cancel_task",
        Command::GetRun { .. } => "get_run",
        Command::ListSessions { .. } => "list_sessions",
        Command::GetHistory { .. } => "get_history",
        Command::ClearSession { .. } => "clear_session",
        Command::RunSlash { .. } => "run_slash",
        Command::GetGlance => "get_glance",
        Command::GetSnapshot => "get_snapshot",
        Command::GetSystem => "get_system",
        Command::SearchMessages { .. } => "search_messages",
        Command::ListCommands => "list_commands",
        Command::ListMemories => "list_memories",
        Command::ForgetMemory { .. } => "forget_memory",
        Command::UploadFile { .. } => "upload_file",
        Command::GetSettings => "get_settings",
        Command::ChangeSettings { .. } => "change_settings",
        Command::GetSetup => "get_setup",
        Command::Setup { .. } => "setup",
        Command::GetChanges { .. } => "get_changes",
        Command::ListProjects => "list_projects",
        Command::CreateProject { .. } => "create_project",
        Command::UpdateProject { .. } => "update_project",
        Command::DeleteProject { .. } => "delete_project",
        Command::SetSessionProject { .. } => "set_session_project",
        Command::SetChannelProject { .. } => "set_channel_project",
        Command::ListAutomations => "list_automations",
        Command::ChangeAutomation { .. } => "change_automation",
        Command::EditAutomation { .. } => "edit_automation",
        Command::ListModels => "list_models",
        Command::GetSessionModel { .. } => "get_session_model",
        Command::SetSessionModel { .. } => "set_session_model",
        Command::RenameSession { .. } => "rename_session",
        Command::RenameAgent { .. } => "rename_agent",
        Command::ArchiveSession { .. } => "archive_session",
        Command::RevertFile { .. } => "revert_file",
        Command::ListProjectFiles { .. } => "list_project_files",
        Command::ReadProjectFile { .. } => "read_project_file",
        Command::ListQueued { .. } => "list_queued",
        Command::EditQueued { .. } => "edit_queued",
        Command::ListSurfaces => "list_surfaces",
        Command::OpenSurface { .. } => "open_surface",
        Command::CloseSurface { .. } => "close_surface",
        Command::OpenShell { .. } => "open_shell",
        Command::CloseShell { .. } => "close_shell",
        Command::TakeControl { .. } => "take_control",
        Command::ReleaseControl { .. } => "release_control",
        Command::PointAt { .. } => "point_at",
        Command::ShareScreen { .. } => "share_screen",
        Command::ListDevices => "list_devices",
        Command::SetDeviceScopes { .. } => "set_device_scopes",
        Command::Unknown => "unknown",
    }
}

fn server_name(frame: &ServerFrame) -> String {
    match frame {
        ServerFrame::Welcome { .. } => "server_welcome".into(),
        ServerFrame::Snapshot { .. } => "server_snapshot".into(),
        ServerFrame::Event { event, .. } => format!("server_event_{}", event.kind()),
        ServerFrame::Ack { result, .. } => format!("server_ack_{}", result_name(result)),
        ServerFrame::Error { .. } => "server_error".into(),
        ServerFrame::Pong => "server_pong".into(),
    }
}

fn result_name(result: &CommandResult) -> &'static str {
    match result {
        CommandResult::MessageAccepted { .. } => "message_accepted",
        CommandResult::Stopped { .. } => "stopped",
        CommandResult::ApprovalResolved { .. } => "approval_resolved",
        CommandResult::Sessions { .. } => "sessions",
        CommandResult::History { .. } => "history",
        CommandResult::Reply { .. } => "reply",
        CommandResult::Glance { .. } => "glance",
        CommandResult::Snapshot { .. } => "snapshot",
        CommandResult::System { .. } => "system",
        CommandResult::SearchResults { .. } => "search_results",
        CommandResult::Commands { .. } => "commands",
        CommandResult::Memories { .. } => "memories",
        CommandResult::Forgotten { .. } => "forgotten",
        CommandResult::Uploaded { .. } => "uploaded",
        CommandResult::Settings { .. } => "settings",
        CommandResult::SettingsChanged { .. } => "settings_changed",
        CommandResult::Setup { .. } => "setup",
        CommandResult::Changes { .. } => "changes",
        CommandResult::ProjectDir { .. } => "project_dir",
        CommandResult::ProjectFile { .. } => "project_file",
        CommandResult::Projects { .. } => "projects",
        CommandResult::Automations { .. } => "automations",
        CommandResult::Run { .. } => "run",
        CommandResult::Project { .. } => "project",
        CommandResult::Models { .. } => "models",
        CommandResult::SessionModel { .. } => "session_model",
        CommandResult::Queued { .. } => "queued",
        CommandResult::Surfaces { .. } => "surfaces",
        CommandResult::SurfaceOpened { .. } => "surface_opened",
        CommandResult::SurfaceControl { .. } => "surface_control",
        CommandResult::ShellOpened { .. } => "shell_opened",
        CommandResult::ScreenShared { .. } => "screen_shared",
        CommandResult::Devices { .. } => "devices",
        CommandResult::Unknown => "unknown",
    }
}

fn command(request_id: &str, command: Command) -> ClientFrame {
    ClientFrame::Command {
        request_id: request_id.into(),
        command,
    }
}

fn client_samples() -> Vec<ClientFrame> {
    vec![
        ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: ClientKind::Watch,
                name: "aidaemon-wear".into(),
                version: Some("0.1.0".into()),
                device: Some("Synthetic Watch 1".into()),
            },
            capabilities: vec![Capability::Compact],
            resume_from: Some(41),
            token: None,
        },
        command(
            "5d0c2b1e-0001",
            Command::SendMessage {
                session_id: None,
                text: "Post the synthetic report for Alice Rivera".into(),
                attachments: vec![AttachmentRef {
                    marked_from: None,
                    height: None,
                    width: None,
                    url: None,
                    id: "up-1".into(),
                    name: "report.pdf".into(),
                    mime: Some("application/pdf".into()),
                    size_bytes: Some(2048),
                }],
                mid_run: Some(MidRun::Queue),
                project_id: Some("project-acme".into()),
                model: Some(ModelChoice::Pinned {
                    provider: "main".into(),
                    model: "synthetic-model-large".into(),
                }),
                effort: Some(Effort::High),
                in_run: None,
            },
        ),
        command(
            "5d0c2b1e-0002",
            Command::StopTurn {
                session_id: "app:6c1f".into(),
            },
        ),
        command(
            "5d0c2b1e-0003",
            Command::ResolveApproval {
                approval_id: "a-7f3e-1".into(),
                decision: ApprovalDecision::ApproveOnce,
            },
        ),
        command(
            "5d0c2b1e-0004",
            Command::CancelTask {
                task_id: "b9e1-4".into(),
            },
        ),
        command(
            "5d0c2b1e-0005",
            Command::ListSessions {
                cursor: Some("50".into()),
                limit: 20,
            },
        ),
        command(
            "5d0c2b1e-0006",
            Command::GetHistory {
                session_id: "app:6c1f".into(),
                before: Some("m-120".into()),
                limit: 30,
            },
        ),
        command(
            "5d0c2b1e-0007",
            Command::ClearSession {
                session_id: "app:6c1f".into(),
            },
        ),
        command(
            "5d0c2b1e-0008",
            Command::RunSlash {
                session_id: "app:6c1f".into(),
                command: "tasks".into(),
                args: String::new(),
            },
        ),
        command("5d0c2b1e-0009", Command::GetGlance),
        command("5d0c2b1e-000a", Command::GetSnapshot),
        command("5d0c2b1e-000c", Command::GetSystem),
        command(
            "5d0c2b1e-000d",
            Command::SearchMessages {
                query: "weekly summary".into(),
                limit: 20,
            },
        ),
        command("5d0c2b1e-000e", Command::ListCommands),
        command("5d0c2b1e-000f", Command::ListMemories),
        command(
            "5d0c2b1e-0010",
            Command::ForgetMemory {
                memory_id: "41".into(),
            },
        ),
        command(
            "5d0c2b1e-0013",
            Command::UploadFile {
                name: "receipt.png".into(),
                mime: Some("image/png".into()),
                data: "iVBORw0KGgo=".into(),
                marked_from: Some("0123456789abcdef01234567".into()),
            },
        ),
        command("5d0c2b1e-0011", Command::GetSettings),
        command(
            "5d0c2b1e-0012",
            Command::ChangeSettings {
                change: SettingsChange::Set {
                    key: "provider.models.default".into(),
                    value: Some("\"synthetic-model-1\"".into()),
                },
            },
        ),
        command("5d0c2b1e-0013", Command::GetSetup),
        command(
            "5d0c2b1e-0014",
            Command::Setup {
                action: SetupAction::ConnectModel {
                    provider: ProviderChoice {
                        preset: "openrouter".into(),
                        primary_model: "synthetic/model-1".into(),
                        ..ProviderChoice::default()
                    },
                    key: Some(SecretText("sk-synthetic-not-a-key".into())),
                },
            },
        ),
        command(
            "5d0c2b1e-0015",
            Command::GetChanges {
                session_id: "app:6c1f".into(),
            },
        ),
        command("5d0c2b1e-0016", Command::ListProjects),
        command(
            "5d0c2b1e-0016a",
            Command::ListProjectFiles {
                project_id: "p-acme".into(),
                path: Some("src".into()),
                ignored: false,
            },
        ),
        command(
            "5d0c2b1e-0016b",
            Command::ReadProjectFile {
                project_id: "p-acme".into(),
                path: "src/contact.rs".into(),
            },
        ),
        command(
            "5d0c2b1e-0017",
            Command::CreateProject {
                name: "Acme site".into(),
                path: Some("/home/synthetic/acme-site".into()),
                create_folder: true,
            },
        ),
        command(
            "5d0c2b1e-0018",
            Command::UpdateProject {
                project_id: "project-acme".into(),
                name: Some("Acme Corp site".into()),
                path: None,
                create_folder: false,
            },
        ),
        command(
            "5d0c2b1e-0018",
            Command::DeleteProject {
                project_id: "project-acme".into(),
            },
        ),
        command(
            "5d0c2b1e-0019",
            Command::SetSessionProject {
                session_id: "app:6c1f".into(),
                project_id: Some("project-acme".into()),
            },
        ),
        command(
            "5d0c2b1e-001a",
            Command::SetChannelProject {
                channel: "telegram:synthetic_coding_bot".into(),
                project_id: Some("project-acme".into()),
            },
        ),
        command("5d0c2b1e-001b", Command::ListAutomations),
        command(
            "5d0c2b1e-00r1",
            Command::GetRun {
                run_id: "run-3f2a".into(),
            },
        ),
        command(
            "5d0c2b1e-001c",
            Command::ChangeAutomation {
                automation_id: "goal-7f3e".into(),
                change: AutomationChange::Pause,
            },
        ),
        command(
            "5d0c2b1e-001d",
            Command::EditAutomation {
                automation_id: "goal-7f3e".into(),
                name: None,
                when: Some("every weekday at 8am".into()),
                instructions: Some("Summarize the synthetic inbox; skip newsletters".into()),
                notify: None,
            },
        ),
        command("5d0c2b1e-001e", Command::ListModels),
        command(
            "5d0c2b1e-001f",
            Command::GetSessionModel {
                session_id: "app:6c1f".into(),
            },
        ),
        command(
            "5d0c2b1e-0020",
            Command::SetSessionModel {
                session_id: "app:6c1f".into(),
                model: Some(ModelChoice::Auto),
                effort: Some(Effort::Low),
            },
        ),
        command(
            "5d0c2b1e-0021",
            Command::RenameSession {
                session_id: "app:6c1f".into(),
                title: Some("Acme invoice".into()),
            },
        ),
        command(
            "5d0c2b1e-0021a",
            Command::RenameAgent {
                session_id: "app:6c1f".into(),
                thread: "run-3f2a".into(),
                name: Some("Designer".into()),
            },
        ),
        command(
            "5d0c2b1e-0022",
            Command::ArchiveSession {
                session_id: "app:6c1f".into(),
                archived: true,
            },
        ),
        command(
            "5d0c2b1e-0023",
            Command::RevertFile {
                session_id: "app:6c1f".into(),
                root: "/home/synthetic/acme-site".into(),
                path: "src/contact.rs".into(),
            },
        ),
        command(
            "5d0c2b1e-0024",
            Command::ListQueued {
                session_id: "app:6c1f".into(),
            },
        ),
        command(
            "5d0c2b1e-0025",
            Command::EditQueued {
                session_id: "app:6c1f".into(),
                message_id: 7,
                text: Some("Also check the footer".into()),
                now: false,
            },
        ),
        command("5d0c2b1e-0026", Command::ListSurfaces),
        command(
            "5d0c2b1e-0027",
            Command::OpenSurface {
                surface_id: "srf_7d1c".into(),
                mode: SurfaceMode::Control,
                accept: StreamAccept {
                    codecs: vec!["jpeg".into()],
                    max_w: Some(1080),
                    max_h: None,
                    max_fps: Some(8),
                    quality: Some(60),
                    fit_w: Some(390),
                    fit_h: Some(560),
                },
            },
        ),
        command(
            "5d0c2b1e-0028",
            Command::CloseSurface {
                live_id: "live_3a9f".into(),
            },
        ),
        command(
            "5d0c2b1e-0061",
            Command::OpenShell {
                session_id: "app:6c1f".into(),
                cols: 100,
                rows: 30,
            },
        ),
        command(
            "5d0c2b1e-0062",
            Command::CloseShell {
                shell_id: "sh_4b2e".into(),
            },
        ),
        command(
            "5d0c2b1e-0029",
            Command::TakeControl {
                surface_id: "srf_7d1c".into(),
            },
        ),
        command(
            "5d0c2b1e-002a",
            Command::ReleaseControl {
                surface_id: "srf_7d1c".into(),
            },
        ),
        command(
            "5d0c2b1e-002d",
            Command::PointAt {
                surface_id: "srf_7d1c".into(),
                x: 0.5,
                y: 0.25,
                note: Some("This one".into()),
            },
        ),
        command(
            "5d0c2b1e-002e",
            Command::ShareScreen {
                name: "Synthetic studio Mac".into(),
                width: 2560,
                height: 1440,
            },
        ),
        command("5d0c2b1e-002b", Command::ListDevices),
        command(
            "5d0c2b1e-002c",
            Command::SetDeviceScopes {
                device_id: "dev-phone-1".into(),
                scopes: vec![Scope::Observe, Scope::Chat, Scope::Approve],
            },
        ),
        command("5d0c2b1e-000b", Command::Unknown),
        ClientFrame::Ping,
    ]
}

fn setup_state() -> SetupState {
    SetupState {
        step: SetupStep::Model,
        found: vec![FoundModel {
            preset: "openai_chatgpt".into(),
            name: "OpenAI (ChatGPT subscription) · gpt-synthetic".into(),
            source: "ChatGPT sign-in on this computer".into(),
            billing: Billing::Plan,
            works: true,
            detail: String::new(),
        }],
        providers: vec![ProviderOption {
            preset: "openrouter".into(),
            name: "OpenRouter".into(),
            billing: Billing::PerToken,
            needs_key: true,
            key_url: "https://openrouter.ai/keys".into(),
            needs_base_url: false,
            model: "synthetic/model-1".into(),
        }],
        model: None,
        checked: Some(CheckedKey {
            preset: "openrouter".into(),
            models: vec!["synthetic/model-1".into(), "synthetic/model-2".into()],
            suggested: Some("synthetic/model-1".into()),
        }),
        sign_in: Some(SignIn {
            url: "https://auth.example.invalid/authorize?state=synthetic".into(),
            paste: true,
        }),
        channels: vec![ChannelOption {
            platform: "telegram".into(),
            name: "Telegram".into(),
            help_url: "https://t.me/BotFather".into(),
            needs_app_token: false,
        }],
        channel: None,
        notices: vec![
            "Secrets are kept in /home/synthetic/.aidaemon/.env (no keychain here).".into(),
        ],
        on_server: true,
    }
}

fn event(seq: Option<u64>, event: Event) -> ServerFrame {
    ServerFrame::Event {
        seq,
        at_ms: 1_767_225_600_000,
        session_id: Some("app:6c1f".into()),
        event,
    }
}

fn ack(result: CommandResult) -> ServerFrame {
    ServerFrame::Ack {
        request_id: "5d0c2b1e-0001".into(),
        result,
    }
}

fn server_samples() -> Vec<ServerFrame> {
    vec![
        ServerFrame::Welcome {
            version: ProtocolVersion::CURRENT,
            server_capabilities: vec![Capability::Compact],
            owner: OwnerInfo {
                name: Some("Alice Rivera".into()),
            },
            latest_seq: 42,
            setup: Some(setup_state()),
        },
        ServerFrame::Snapshot {
            seq: 42,
            state: snapshot(),
        },
        event(Some(1), Event::SessionCreated { session: session() }),
        event(Some(2), Event::SessionUpdated { session: session() }),
        event(
            Some(3),
            Event::UserMessage {
                session_id: "app:6c1f".into(),
                text: "Post the synthetic report for Alice Rivera".into(),
                attachments: Vec::new(),
                origin: "app".into(),
            },
        ),
        event(
            Some(4),
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: "app:6c1f".into(),
                text: "Got it, I'll start on that as soon as this finishes.".into(),
                suggestions: Vec::new(),
            },
        ),
        event(
            Some(5),
            Event::TurnStarted {
                turn_id: "b9e1-4".into(),
                session_id: "app:6c1f".into(),
                origin: "app".into(),
            },
        ),
        event(
            None,
            Event::TurnDelta {
                turn_id: "b9e1-4".into(),
                text: "Posting".into(),
            },
        ),
        event(
            None,
            Event::TurnStatus {
                turn_id: "b9e1-4".into(),
                status: TurnStatus::tool_started("http_request", "POST h.example.invalid", None),
            },
        ),
        event(
            Some(6),
            Event::UserActionRequired {
                turn_id: "b9e1-4".into(),
                message: "Finish signing in at https://auth.example.invalid".into(),
            },
        ),
        event(
            Some(7),
            Event::TurnCompleted {
                turn_id: "b9e1-4".into(),
                text: "Posted the report.".into(),
                attachments: Vec::new(),
                state: TurnState::Completed,
                error: None,
                served_by: Some(aidaemon_client_protocol::ServedBy {
                    preset: "anthropic".into(),
                    provider: "Anthropic".into(),
                    model: "synthetic-model-1".into(),
                }),
            },
        ),
        event(
            Some(8),
            Event::ApprovalRequested {
                approval: approval(),
            },
        ),
        event(
            Some(9),
            Event::ApprovalResolved {
                approval_id: "a-7f3e-1".into(),
                outcome: ApprovalOutcome::ApprovedOnce,
                resolved_by: Some(ResolvedBy {
                    channel: "app".into(),
                    device: Some("Synthetic Watch 1".into()),
                }),
            },
        ),
        event(Some(10), Event::TaskUpdated { task: task() }),
        event(Some(10), Event::RunUpdated { run: run() }),
        event(
            None,
            Event::RunStep {
                run_id: "run-3f2a".into(),
                session_id: "app:6c1f".into(),
                step: "Running a command".into(),
                at_ms: 1_767_225_660_000,
            },
        ),
        event(Some(10), Event::ProjectUpdated { project: project() }),
        event(
            Some(10),
            Event::ProjectDeleted {
                project_id: "project-acme".into(),
            },
        ),
        event(
            Some(11),
            Event::GoalUpdated {
                goal: GoalSummary {
                    id: "g-1".into(),
                    description: "Water the plants every Monday".into(),
                    status: "active".into(),
                    session_id: Some("app:6c1f".into()),
                },
            },
        ),
        event(Some(12), Event::Unknown),
        event(
            None,
            Event::ViewAction {
                action: ViewAction::ShowApproval {
                    approval_id: "a-7f3e-1".into(),
                },
                reason: "You asked what is waiting".into(),
            },
        ),
        event(
            None,
            Event::SettingsUpdated {
                ok: true,
                message: "Updated provider.models.default".into(),
            },
        ),
        event(
            None,
            Event::SetupUpdated {
                setup: SetupState {
                    step: SetupStep::Channel,
                    model: Some("OpenAI (ChatGPT subscription) · gpt-synthetic".into()),
                    channel: Some(ChannelWait {
                        platform: "telegram".into(),
                        bot: "@synthetic_bot".into(),
                        connected: true,
                        claim_code: None,
                    }),
                    ..SetupState::default()
                },
            },
        ),
        event(
            Some(9),
            Event::FeatureRequested {
                session_id: "app:6c1f".into(),
                feature: Feature::Browser,
                reason: "to read the synthetic report page".into(),
            },
        ),
        event(
            None,
            Event::SurfaceUpdated {
                surface: surface(),
            },
        ),
        event(
            None,
            Event::SurfaceClosed {
                surface_id: "srf_7d1c".into(),
            },
        ),
        event(
            None,
            Event::ShellUpdated {
                shell: ShellInfo {
                    exit_code: Some(0),
                    ..shell()
                },
            },
        ),
        event(
            None,
            Event::ShellClosed {
                shell_id: "sh_4b2e".into(),
                session_id: "app:6c1f".into(),
            },
        ),
        ack(CommandResult::MessageAccepted {
            session_id: "app:6c1f".into(),
            new_session: true,
        }),
        ack(CommandResult::Stopped {
            report: "⏹️ Stopped:\n• Working on your message".into(),
        }),
        ack(CommandResult::ApprovalResolved {
            approval_id: "a-7f3e-1".into(),
            outcome: ApprovalOutcome::Denied,
        }),
        ack(CommandResult::Sessions {
            sessions: vec![session()],
            next_cursor: Some("1".into()),
        }),
        ack(CommandResult::History {
            session_id: "app:6c1f".into(),
            messages: vec![
                HistoryMessage {
                    attachments: Vec::new(),
                    id: "m-119".into(),
                    role: "user".into(),
                    text: "Post the synthetic report for Alice Rivera".into(),
                    at_ms: 1_767_225_600_000,
                    tools: Vec::new(),
                    calls: Vec::new(),
                    handed_off: false,
                    served_by: None,
                },
                HistoryMessage {
                    attachments: Vec::new(),
                    id: "m-120".into(),
                    role: "assistant".into(),
                    text: "Posted it.".into(),
                    at_ms: 1_767_225_660_000,
                    tools: vec!["http_request".into()],
                    calls: vec![ToolCallInfo {
                        name: "http_request".into(),
                        input: r#"{"method":"POST","url":"https://reports.example.invalid/weekly"}"#
                            .into(),
                        output: "201 Created".into(),
                        kind: Some(ToolKind::Web),
                    }],
                    handed_off: false,
                    served_by: None,
                },
                HistoryMessage {
                    attachments: Vec::new(),
                    id: "m-121".into(),
                    role: "assistant".into(),
                    text: "Started in the background: synthetic test run.".into(),
                    at_ms: 1_767_225_720_000,
                    tools: Vec::new(),
                    calls: Vec::new(),
                    handed_off: true,
                    served_by: None,
                },
            ],
            more: true,
            runs: Vec::new(),
            agent_names: [("run-3f2a".to_string(), "Designer".to_string())].into(),
        }),
        ack(CommandResult::Reply {
            text: "Nothing is running in this chat right now.".into(),
        }),
        ack(CommandResult::Glance {
            glance: Glance {
                running_tasks: vec![task()],
                pending_approvals: vec![approval().compact()],
                last_reply: Some(LastReply {
                    session_id: "app:6c1f".into(),
                    turn_id: "b9e1-3".into(),
                    text: "Posted the report.".into(),
                    at_ms: 1_767_225_600_000,
                }),
            },
        }),
        ack(CommandResult::Snapshot {
            seq: 42,
            state: snapshot(),
        }),
        ack(CommandResult::System {
            system: SystemStatus {
                version: "0.0.0-synthetic".into(),
                uptime_secs: 3_600,
                model: Some("model-a".into()),
                fallback_models: vec!["model-b".into()],
                usage: vec![UsageDay {
                    day: "2026-01-01".into(),
                    model: "model-a".into(),
                    input_tokens: 12_000,
                    output_tokens: 900,
                    requests: 14,
                }],
                model_calls: Some(ModelCalls {
                    total: 14,
                    failed: 1,
                    fell_back: 0,
                    p50_ms: 1_200,
                    p95_ms: 4_800,
                    max_ms: 6_100,
                }),
                schedules: vec![ScheduleInfo {
                    goal_id: "g-1".into(),
                    description: "Send the synthetic weekly summary".into(),
                    status: "active".into(),
                    schedule: "every monday at 9".into(),
                    paused: false,
                    one_shot: false,
                    next_run_ms: Some(1_767_600_000_000),
                    last_run_ms: None,
                }],
                jobs: vec![JobInfo {
                    name: "consolidation".into(),
                    every_secs: 3_600,
                    running: false,
                    consecutive_failures: 0,
                    last_success_ms: Some(1_767_225_000_000),
                    last_error: None,
                }],
                queues: vec![QueueInfo {
                    name: "triggers".into(),
                    depth: 0,
                    capacity: 64,
                    dropped: 0,
                    failed: 0,
                    overloaded: false,
                }],
                checks: vec![HealthCheckInfo {
                    name: "Synthetic site".into(),
                    target: "https://h.example.invalid".into(),
                    status: CheckStatus::Healthy,
                    paused: false,
                    consecutive_failures: 0,
                    latency_ms: Some(120),
                    checked_ms: Some(1_767_225_500_000),
                    error: None,
                }],
                runs: vec![GoalRunInfo {
                    goal_id: "g-1".into(),
                    description: "Send the synthetic weekly summary".into(),
                    status: "completed".into(),
                    outcome: Some("Sent the summary to Jordan Lee".into()),
                    started_ms: 1_767_225_000_000,
                    finished_ms: Some(1_767_225_120_000),
                    session_id: Some("telegram:synthetic-user-1".into()),
                }],
                nodes: vec![NodeInfo {
                    node_id: "node_synthetic_1".into(),
                    name: "Synthetic Companion".into(),
                    kind: "k10".into(),
                    online: true,
                    last_seen_ms: Some(1_767_225_300_000),
                    firmware_version: Some("0.8.12".into()),
                    connection: Some(NodeConnection::LocalNetwork),
                }],
            },
        }),
        ack(CommandResult::SearchResults {
            hits: vec![SearchHit {
                session_id: "app:6c1f".into(),
                message_id: "m-119".into(),
                role: "user".into(),
                snippet: "Post the synthetic report for Alice Rivera".into(),
                at_ms: 1_767_225_600_000,
            }],
        }),
        ack(CommandResult::Commands {
            commands: vec![SlashCommand {
                name: "tasks".into(),
                description: "What is running now".into(),
            }],
        }),
        ack(CommandResult::Memories {
            memories: vec![MemoryItem {
                id: "41".into(),
                category: "preference".into(),
                key: "coffee".into(),
                value: "Alice Rivera takes her coffee black".into(),
                source: "conversation".into(),
                updated_ms: 1_767_225_000_000,
            }],
        }),
        ack(CommandResult::Forgotten {
            memory_id: "41".into(),
        }),
        ack(CommandResult::Uploaded {
            attachment: AttachmentRef {
                marked_from: None,
                height: None,
                width: None,
                id: "f-3c9a".into(),
                name: "receipt.png".into(),
                mime: Some("image/png".into()),
                size_bytes: Some(8),
                url: Some("/client/v1/files/f-3c9a/5b1e/receipt.png".into()),
            },
        }),
        ack(CommandResult::Settings {
            settings: Settings {
                config_path: "/home/synthetic/.aidaemon/config.toml".into(),
                groups: vec![SettingGroup {
                    name: "provider".into(),
                    settings: vec![
                        Setting {
                            key: "provider.models.default".into(),
                            value: Some("\"synthetic-model-1\"".into()),
                            kind: SettingKind::Text,
                            secret_set: false,
                            description: "The model most turns use.".into(),
                            example: Some("\"openai/gpt-4o\"".into()),
                            applies: Applies::Reload,
                            guarded: false,
                        },
                        Setting {
                            key: "provider.api_key".into(),
                            value: None,
                            kind: SettingKind::Secret,
                            secret_set: true,
                            description: String::new(),
                            example: None,
                            applies: Applies::Reload,
                            guarded: true,
                        },
                    ],
                }],
                bots: vec![BotInfo {
                    id: 3,
                    platform: "telegram".into(),
                    allowed_user_ids: vec!["telegram:synthetic-user-1".into()],
                    created_at: "2026-01-01 00:00:00".into(),
                }],
                mcp_servers: vec![McpServerInfo {
                    name: "files".into(),
                    command: "synthetic-mcp --root /tmp".into(),
                    enabled: true,
                    tools: vec!["read_file".into()],
                    from_config: false,
                }],
                reload_pending: true,
                restart_pending: false,
                notices: vec![
                    "The database moves to /data/aidaemon.db at the next restart.".into(),
                ],
                database_path: "/home/synthetic/.aidaemon/aidaemon.db".into(),
                database_encrypted: true,
                providers: vec![ProviderOption {
                    preset: "openai".into(),
                    name: "OpenAI".into(),
                    billing: Billing::PerToken,
                    needs_key: true,
                    key_url: "https://platform.openai.com/api-keys".into(),
                    needs_base_url: false,
                    model: "gpt-x".into(),
                }],
                connected: vec![
                    ConnectedProvider {
                        failover: None,
                        preset: "openai_chatgpt".into(),
                        name: "OpenAI (ChatGPT subscription)".into(),
                        base_url: String::new(),
                        models: vec!["synthetic-model-1".into()],
                        hard_model: "synthetic-model-1".into(),
                        quick_model: "synthetic-model-2".into(),
                        offered_models: vec![
                            "synthetic-model-1".into(),
                            "synthetic-model-2".into(),
                            "synthetic-model-3".into(),
                        ],
                        credential: CredentialState::SignedIn,
                        key_setting: String::new(),
                        usage: Some(ProviderUsage {
                            plan: "plus".into(),
                            windows: vec![
                                UsageWindow {
                                    minutes: 300,
                                    used_percent: 42,
                                    resets_at_ms: Some(1_700_000_100_000),
                                },
                                UsageWindow {
                                    minutes: 10_080,
                                    used_percent: 18,
                                    resets_at_ms: None,
                                },
                            ],
                            balance: None,
                            read_at_ms: 1_700_000_000_000,
                        }),
                    },
                    ConnectedProvider {
                        failover: Some(0),
                        preset: "openai".into(),
                        name: "OpenAI".into(),
                        base_url: "https://api.example.com/v1".into(),
                        models: vec!["gpt-x".into()],
                        hard_model: String::new(),
                        quick_model: String::new(),
                        offered_models: Vec::new(),
                        credential: CredentialState::KeyStored,
                        key_setting: "provider.fallbacks[0].api_key".into(),
                        usage: Some(ProviderUsage {
                            balance: Some(UsageBalance {
                                used_cents: 350,
                                limit_cents: Some(1_000),
                                remaining_cents: Some(650),
                            }),
                            read_at_ms: 1_700_000_000_000,
                            ..ProviderUsage::default()
                        }),
                    },
                ],
                chatgpt_signed_in: true,
                sign_in: None,
                checked: Some(CheckedKey {
                    preset: "openai".into(),
                    models: vec!["gpt-x".into(), "gpt-y".into()],
                    suggested: Some("gpt-x".into()),
                }),
                coding_agents: vec![
                    CodingAgent {
                        name: "codex".into(),
                        program: "Codex".into(),
                        mode: Some(AgentMode::Checked),
                        modes: vec![
                            AgentMode::Checked,
                            AgentMode::EveryCommand,
                            AgentMode::FullAccess,
                        ],
                        note: String::new(),
                        readiness: AgentReadiness::NotUsable,
                        readiness_reason: "Codex isn't signed in on this machine.".into(),
                        readiness_fix: "Run `codex login` on this machine.".into(),
                        same_as_aidaemon: true,
                        within_level: vec![AgentMode::EveryCommand, AgentMode::Checked],
                        capped: false,
                        account: "Its own Codex sign-in.".into(),
                        takes_model: true,
                        model: Some("synthetic-large".into()),
                        default_model: Some("synthetic-small".into()),
                        models: vec![
                            AgentModelOption {
                                id: "synthetic-large".into(),
                                name: "Synthetic Large".into(),
                            },
                            AgentModelOption {
                                id: "synthetic-small".into(),
                                name: "Synthetic Small".into(),
                            },
                        ],
                    },
                    CodingAgent {
                        name: "aider".into(),
                        program: "Aider".into(),
                        mode: None,
                        modes: vec![AgentMode::FullAccess],
                        note: "AIDaemon cannot check this agent's actions.".into(),
                        readiness: AgentReadiness::Unchecked,
                        readiness_reason: String::new(),
                        readiness_fix: String::new(),
                        same_as_aidaemon: false,
                        within_level: Vec::new(),
                        capped: false,
                        account: String::new(),
                        takes_model: false,
                        model: None,
                        default_model: None,
                        models: Vec::new(),
                    },
                ],
                permission_level: Some(PermissionLevel::Standard),
            },
        }),
        ack(CommandResult::SettingsChanged {
            outcome: SettingsOutcome::AwaitingApproval {
                approval_id: "a-7f3e-2".into(),
                other_device: true,
            },
        }),
        ack(CommandResult::Setup {
            setup: setup_state(),
        }),
        ack(CommandResult::ProjectDir {
            dir: ProjectDir {
                project_id: "p-acme".into(),
                path: "src".into(),
                entries: vec![
                    ProjectEntry {
                        name: "forms".into(),
                        kind: EntryKind::Folder,
                        size_bytes: None,
                        modified_ms: Some(1_767_225_600_000),
                        status: None,
                        changed: true,
                        ignored: false,
                        secret: false,
                    },
                    ProjectEntry {
                        name: "contact.rs".into(),
                        kind: EntryKind::File,
                        size_bytes: Some(1_204),
                        modified_ms: Some(1_767_225_600_000),
                        status: Some(FileStatus::Modified),
                        changed: false,
                        ignored: false,
                        secret: false,
                    },
                    ProjectEntry {
                        name: ".env".into(),
                        kind: EntryKind::File,
                        size_bytes: Some(64),
                        modified_ms: None,
                        status: None,
                        changed: false,
                        ignored: false,
                        secret: true,
                    },
                ],
                truncated: false,
                branch: Some("feature/contact-form".into()),
            },
        }),
        ack(CommandResult::ProjectFile {
            file: ProjectFile {
                project_id: "p-acme".into(),
                path: "src/contact.rs".into(),
                size_bytes: 26,
                modified_ms: Some(1_767_225_600_000),
                version: "sha256:0f1e2d3c".into(),
                text: Some("fn send() {\n}\nfn main() {}\n".into()),
                truncated: false,
                withheld: None,
                status: Some(FileStatus::Modified),
                link: None,
            },
        }),
        ack(CommandResult::Changes {
            changes: SessionChanges {
                last_check: Some(CheckRun {
                    command: "cargo test -q contact".into(),
                    passed: false,
                    exit_code: Some(101),
                    at_ms: 1_767_225_600_000,
                }),
                repos: vec![RepoChanges {
                    root: "/home/synthetic/acme-site".into(),
                    branch: Some("feature/contact-form".into()),
                    files: vec![
                        FileChange {
                            path: "src/contact.rs".into(),
                            status: FileStatus::Modified,
                            added: 2,
                            removed: 1,
                            by_session: true,
                        },
                        FileChange {
                            path: "notes.md".into(),
                            status: FileStatus::Untracked,
                            added: 1,
                            removed: 0,
                            by_session: false,
                        },
                    ],
                    patch: "diff --git a/src/contact.rs b/src/contact.rs\n--- a/src/contact.rs\n+++ b/src/contact.rs\n@@ -1,2 +1,3 @@\n-fn send() {}\n+fn send() {\n+}\n fn main() {}\n".into(),
                    truncated: false,
                }],
            },
        }),
        ack(CommandResult::Run {
            run: RunDetail {
                run_id: "run-3f2a".into(),
                agent_session_id: Some("synthetic-session-1".into()),
                can_continue: true,
                model: Some("synthetic-large".into()),
                working_dir: Some("/w/synthetic".into()),
                task: "Prepare the synthetic release notes".into(),
                result: Some("Wrote release.md".into()),
                steps: vec![RunStepItem {
                    words: "Editing release.md".into(),
                    state: RunStepState::Done,
                    at_ms: Some(1_767_225_660_000),
                    by_helper: false,
                }],
                plan: vec![RunPlanItem {
                    text: "Write the notes".into(),
                    status: RunPlanStatus::Done,
                }],
                permissions: vec![RunPermission {
                    label: "write: /w/synthetic/release.md".into(),
                    outcome: RunPermissionOutcome::Allowed,
                    reason: None,
                }],
                usage: Some(RunUsage {
                    input_tokens: 1200,
                    output_tokens: 300,
                    cost_usd: Some(0.02),
                }),
            },
        }),
        ack(CommandResult::Automations {
            automations: vec![AutomationSummary {
                session_id: Some("telegram:synthetic-user-1".into()),
                id: "goal-7f3e".into(),
                kind: AutomationKind::Scheduled,
                name: Some("Inbox digest".into()),
                what: "Summarize the synthetic inbox".into(),
                how_often: "Every weekday at 9:00 AM ET".into(),
                status: "paused".into(),
                next_run_ms: None,
                recent_runs: vec![
                    AutomationRun {
                        started_ms: 1_767_088_800_000,
                        finished_ms: Some(1_767_092_400_000),
                        outcome: RunOutcome::Missed,
                        summary: Some("The daemon wasn't running when this was due.".into()),
                    },
                    AutomationRun {
                        started_ms: 1_767_002_400_000,
                        finished_ms: Some(1_767_002_460_000),
                        outcome: RunOutcome::Failed,
                        summary: Some("The synthetic mail account refused the sign-in.".into()),
                    },
                    AutomationRun {
                        started_ms: 1_766_916_000_000,
                        finished_ms: Some(1_766_916_042_000),
                        outcome: RunOutcome::Done,
                        summary: Some("3 new messages, 1 needs a reply.".into()),
                    },
                ],
                details: vec![
                    "Can do: Read only – makes no changes".into(),
                    "Tells you when: a run finishes, or it hits a problem".into(),
                ],
                edit: Some(AutomationEdit {
                    when: "every weekday at 9am".into(),
                    instructions: Some("Summarize the synthetic inbox".into()),
                    notify: Some(AutomationNotify::Changes),
                }),
            }],
            note: Some("Paused \"Inbox digest\" (scheduled). It won't run until you resume it.".into()),
            changed: true,
        }),
        ack(CommandResult::Projects {
            projects: vec![project()],
        }),
        ack(CommandResult::Project { project: project() }),
        ack(CommandResult::Models {
            options: ModelOptions {
                providers: vec![
                    ModelProvider {
                        id: "main".into(),
                        name: "ChatGPT".into(),
                        billing: Billing::Plan,
                        models: vec![ModelOption {
                            id: "synthetic-model-large".into(),
                            efforts: vec![Effort::Low, Effort::Medium, Effort::High],
                        }],
                    },
                    ModelProvider {
                        id: "failover-0".into(),
                        name: "Google Gemini".into(),
                        billing: Billing::PerToken,
                        models: vec![ModelOption {
                            id: "synthetic-model-flash".into(),
                            efforts: vec![],
                        }],
                    },
                ],
                agents: vec![AgentOption {
                    id: "codex".into(),
                    name: "Codex".into(),
                    mark: Some("openai".into()),
                }],
            },
        }),

        ack(CommandResult::SessionModel {
            session_model: SessionModel {
                session_id: "app:6c1f".into(),
                model: ModelChoice::Pinned {
                    provider: "main".into(),
                    model: "synthetic-model-large".into(),
                },
                effort: Effort::High,
            },
        }),
        ack(CommandResult::Queued {
            session_id: "app:6c1f".into(),
            messages: vec![QueuedMessage {
                id: 7,
                text: "Also check the footer".into(),
                queued_at_ms: 1_767_225_600_000,
                after: true,
            }],
        }),
        ack(CommandResult::Surfaces {
            surfaces: vec![surface()],
        }),
        ack(CommandResult::SurfaceOpened {
            live_id: "live_3a9f".into(),
            surface: surface(),
            transports: vec![SurfaceTransport {
                kind: TransportKind::WsFrames,
                url: "/client/v1/surfaces/live_3a9f/ws".into(),
                ticket: "tkt_synthetic_0001".into(),
                codecs: vec!["jpeg".into()],
            }],
        }),
        ack(CommandResult::ShellOpened {
            shell: shell(),
            transport: ShellTransport {
                url: "/client/v1/shells/sh_4b2e/ws".into(),
                ticket: "tkt_synthetic_0003".into(),
            },
        }),
        ack(CommandResult::SurfaceControl {
            surface: SurfaceInfo {
                controller: SurfaceController::Agent,
                ..surface()
            },
        }),
        ack(CommandResult::ScreenShared {
            surface_id: "srf_remote_1".into(),
            url: "/client/v1/surfaces/provide/srf_remote_1/ws".into(),
            ticket: "tkt_synthetic_0002".into(),
        }),
        ack(CommandResult::Devices {
            devices: vec![DeviceInfo {
                device_id: "dev-phone-1".into(),
                name: "Synthetic phone".into(),
                kind: ClientKind::Mobile,
                scopes: ALL_SCOPES.to_vec(),
                risk_ceiling: None,
                created_at_ms: 1_767_225_600_000,
                last_seen_ms: Some(1_767_225_900_000),
                revoked: false,
            }],
            can_change: true,
        }),
        ack(CommandResult::Unknown),
        ServerFrame::Error {
            request_id: Some("5d0c2b1e-0003".into()),
            error: ProtocolError::new(
                ErrorCode::Conflict,
                "This approval was already answered: approved_once",
            ),
        },
        ServerFrame::Pong,
    ]
}

fn check(name: &str, value: Value, roundtrip: impl Fn(&str) -> Value) {
    let path = fixtures().join(format!("{name}.golden"));
    let mut text = serde_json::to_string_pretty(&value).unwrap();
    text.push('\n');
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    let golden = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture {}", path.display()));
    let golden_value: Value = serde_json::from_str(&golden).unwrap();
    assert_eq!(value, golden_value, "{name}: sample differs from fixture");
    assert_eq!(
        roundtrip(&golden),
        golden_value,
        "{name}: fixture does not round-trip"
    );
}

#[test]
fn every_client_frame_matches_its_fixture() {
    let samples = client_samples();
    for frame in &samples {
        check(
            &client_name(frame),
            serde_json::to_value(frame).unwrap(),
            |text| {
                serde_json::to_value(serde_json::from_str::<ClientFrame>(text).unwrap()).unwrap()
            },
        );
    }
}

#[test]
fn every_server_frame_matches_its_fixture() {
    let samples = server_samples();
    for frame in &samples {
        check(
            &server_name(frame),
            serde_json::to_value(frame).unwrap(),
            |text| {
                serde_json::to_value(serde_json::from_str::<ServerFrame>(text).unwrap()).unwrap()
            },
        );
    }
}

#[test]
fn the_http_events_page_matches_its_fixture() {
    let page = EventsPage {
        frames: vec![event(
            Some(43),
            Event::TurnStarted {
                turn_id: "b9e1-5".into(),
                session_id: "app:6c1f".into(),
                origin: "telegram".into(),
            },
        )],
        next_since: 43,
    };
    check(
        "http_events_page",
        serde_json::to_value(&page).unwrap(),
        |text| serde_json::to_value(serde_json::from_str::<EventsPage>(text).unwrap()).unwrap(),
    );
}

fn http_samples() -> Vec<(&'static str, Value)> {
    vec![
        (
            "http_offer_request",
            serde_json::to_value(OfferRequest {
                confirm_in_session: Some("telegram:synthetic-user-1".into()),
            })
            .unwrap(),
        ),
        (
            "http_pairing_offer",
            serde_json::to_value(PairingOffer {
                offer_id: "of-1".into(),
                code: "K7Q2-9FZD".into(),
                expires_at_ms: 1_767_226_200_000,
                link: "aidaemon-pair:?gateway=http%3A%2F%2F127.0.0.1%3A8788&offer=of-1&code=K7Q2-9FZD&fp=3b1f".into(),
                fingerprint: "3b1f".into(),
                public_url: Some("https://app.example.com".into()),
                web_app: None,
            })
            .unwrap(),
        ),
        (
            "http_redeem_request",
            serde_json::to_value(RedeemRequest {
                offer_id: "of-1".into(),
                code: "K7Q2-9FZD".into(),
                device_name: "Synthetic Phone".into(),
                device_kind: ClientKind::Web,
                public_key: "BOr9...".into(),
                scopes: vec![Scope::Observe, Scope::Approve],
            })
            .unwrap(),
        ),
        (
            "http_enrollment",
            serde_json::to_value(Enrollment {
                offer_id: "of-1".into(),
                state: EnrollmentState::Approved,
                device_id: Some("dv-1".into()),
                device_name: Some("Synthetic Phone".into()),
                device_kind: Some(ClientKind::Web),
                scopes: ALL_SCOPES.to_vec(),
                fingerprint: "3b1f".into(),
            })
            .unwrap(),
        ),
        (
            "http_pairing_decision",
            serde_json::to_value(PairingDecision { approve: true }).unwrap(),
        ),
        (
            "http_challenge_request",
            serde_json::to_value(ChallengeRequest {
                device_id: "dv-1".into(),
                endpoint: None,
            })
            .unwrap(),
        ),
        (
            "http_challenge",
            serde_json::to_value(Challenge {
                challenge_id: "ch-1".into(),
                nonce: "n0nce".into(),
                expires_at_ms: 1_767_225_660_000,
            })
            .unwrap(),
        ),
        (
            "http_session_request",
            serde_json::to_value(SessionRequest {
                challenge_id: "ch-1".into(),
                signature: "c2ln...".into(),
                audience: None,
            })
            .unwrap(),
        ),
        (
            "http_socket_sign_in",
            serde_json::to_value(SocketSignIn {
                device_id: "dv-1".into(),
                issued_at_ms: 1_767_225_600_000,
                nonce: "0123456789abcdef0123456789abcdef".into(),
                signature: "c2ln...".into(),
            })
            .unwrap(),
        ),
        (
            "http_access_grant",
            serde_json::to_value(AccessGrant {
                access_token: "acs_synthetic".into(),
                expires_at_ms: 1_767_226_500_000,
                device_id: "dv-1".into(),
                scopes: ALL_SCOPES.to_vec(),
                risk_ceiling: Some(Risk::Medium),
            })
            .unwrap(),
        ),
        (
            "http_devices",
            serde_json::to_value(vec![DeviceInfo {
                device_id: "dv-1".into(),
                name: "Synthetic Phone".into(),
                kind: ClientKind::Web,
                scopes: ALL_SCOPES.to_vec(),
                risk_ceiling: None,
                created_at_ms: 1_767_225_600_000,
                last_seen_ms: Some(1_767_225_700_000),
                revoked: false,
            }])
            .unwrap(),
        ),
    ]
}

fn roundtrip_http(name: &str, text: &str) -> Value {
    fn again<T: serde::de::DeserializeOwned + serde::Serialize>(text: &str) -> Value {
        serde_json::to_value(serde_json::from_str::<T>(text).unwrap()).unwrap()
    }
    match name {
        "http_offer_request" => again::<OfferRequest>(text),
        "http_pairing_offer" => again::<PairingOffer>(text),
        "http_redeem_request" => again::<RedeemRequest>(text),
        "http_enrollment" => again::<Enrollment>(text),
        "http_pairing_decision" => again::<PairingDecision>(text),
        "http_challenge_request" => again::<ChallengeRequest>(text),
        "http_challenge" => again::<Challenge>(text),
        "http_session_request" => again::<SessionRequest>(text),
        "http_socket_sign_in" => again::<SocketSignIn>(text),
        "http_access_grant" => again::<AccessGrant>(text),
        "http_devices" => again::<Vec<DeviceInfo>>(text),
        other => panic!("no type for {other}"),
    }
}

#[test]
fn every_pairing_type_matches_its_fixture() {
    for (name, value) in http_samples() {
        check(name, value, |text| roundtrip_http(name, text));
    }
}

/// Every fixture on disk belongs to a sample, so a renamed variant cannot
/// leave a stale file behind.
#[test]
fn no_fixture_is_orphaned() {
    let mut known: Vec<String> = client_samples().iter().map(client_name).collect();
    known.extend(server_samples().iter().map(server_name));
    known.push("http_events_page".into());
    known.extend(http_samples().into_iter().map(|(name, _)| name.to_string()));
    for entry in std::fs::read_dir(fixtures()).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        let stem = name.trim_end_matches(".golden");
        assert!(known.iter().any(|k| k == stem), "orphaned fixture {name}");
    }
}

/// Every turn status shape round-trips (they ride on ephemeral events, so
/// they get no fixture of their own).
#[test]
fn every_turn_status_round_trips() {
    let statuses = [
        TurnStatus::Thinking,
        TurnStatus::tool_started("terminal", "", Some(ToolKind::Run))
            .with_detail("{\"command\": \"cargo test -q\"}"),
        TurnStatus::ToolProgress {
            name: "terminal".into(),
            chunk: "building".into(),
        },
        TurnStatus::tool_completed("terminal", "ok").with_detail("test result: ok"),
        TurnStatus::tool_handed_off("Coding agent", "Working on in the background")
            .with_detail("Started in the background: claude coding agent."),
        TurnStatus::PlanStep {
            step: 1,
            total: 3,
            description: "Draft the report".into(),
            state: StepState::Completed,
        },
        TurnStatus::Checklist {
            text: "- [x] Draft".into(),
        },
        TurnStatus::Note {
            text: "Helper started: report".into(),
        },
    ];
    for status in statuses {
        let text = serde_json::to_string(&status).unwrap();
        assert_eq!(serde_json::from_str::<TurnStatus>(&text).unwrap(), status);
    }
}

/// Every view action round-trips, and each is only a place on screen: an
/// exhaustive match fails to compile when a kind is added without being
/// listed here, and none carries a decision.
#[test]
fn every_view_action_round_trips_and_carries_no_decision() {
    let actions = [
        ViewAction::OpenSession {
            session_id: "app:6c1f".into(),
        },
        ViewAction::ShowApproval {
            approval_id: "a-7f3e-1".into(),
        },
        ViewAction::ShowTask {
            task_id: "b9e1-4".into(),
        },
        ViewAction::FocusPane { pane: Pane::Tasks },
        ViewAction::NewSession,
        ViewAction::ShowDevices,
        ViewAction::ShowSurface {
            surface_id: "srf_7d1c".into(),
        },
    ];
    for action in actions {
        match &action {
            ViewAction::OpenSession { .. }
            | ViewAction::ShowApproval { .. }
            | ViewAction::ShowTask { .. }
            | ViewAction::FocusPane { .. }
            | ViewAction::NewSession
            | ViewAction::ShowDevices
            | ViewAction::ShowSurface { .. }
            | ViewAction::Unknown => {}
        }
        let value = serde_json::to_value(&action).unwrap();
        for field in ["decision", "outcome", "text", "device_id", "command"] {
            assert!(value.get(field).is_none(), "{value} carries {field}");
        }
        assert_eq!(serde_json::from_value::<ViewAction>(value).unwrap(), action);
    }
}

/// A card action this build does not know (an older client meeting a newer
/// daemon's `click_on_site`, say) reads as `Unknown`, so the client falls
/// back to the approval's summary and detail.
#[test]
fn an_unknown_card_action_reads_as_unknown() {
    let action: CardAction =
        serde_json::from_str(r#"{"kind":"some_later_action","site":"https://acme.example"}"#)
            .unwrap();
    assert_eq!(action, CardAction::Unknown);
}
