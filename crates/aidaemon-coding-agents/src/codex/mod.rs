//! Codex, OpenAI's coding agent, in each permission mode.
//!
//! - Checked: a read-only run is `codex exec` in its read-only sandbox. A
//!   read-write run gets an invocation-local permission profile (minimal
//!   reads plus the run's paths, writes only to its write paths, no
//!   network) with the owner's own config ignored; hosted
//!   ([`host::CodexHost`]) where its requests can be answered, it asks
//!   before leaving that profile, and its network goes through Codex's
//!   proxy so a host is asked rather than failing. Unhosted it never asks.
//! - Every command: its sandbox read-only, every command and change but its
//!   own read-only ones asked (`untrusted`); needs the host.
//! - Full access: `--dangerously-bypass-approvals-and-sandbox`.
//!
//! Every run keeps its thread where a later run can continue it, unless the
//! launch says `--ephemeral`: the owner's `CODEX_HOME` (`~/.codex` by
//! default) holds the threads of both launches. An `exec` run writes there
//! directly; a hosted run's own `CODEX_HOME` links its `sessions` folder
//! there and nothing else of the owner's (see [`host`]). So a thread either
//! launch started can be continued by either. An `exec` run continues one
//! as `codex exec <options> resume <thread id> <task>`: the options stay
//! before `resume`, which refuses `--sandbox` after it. A hosted run stays
//! hosted, and still asks: it continues the thread with `thread/resume`
//! under its own sandbox, model and approval policy, whatever the thread
//! last ran with.
//!
//! A run asked for a report ([`RunRequest::report`]) holds its final
//! message to the report schema: an `exec` run with `--output-schema
//! <file>` (a file this run keeps, among the options before `resume`), a
//! hosted run with `turn/start`'s `outputSchema`. Its final message is
//! then the report's JSON. Either launch is read through a host that writes
//! the turn's final message on the line ending it ([`events`]); an `exec`
//! run's ([`exec::ExecHost`]) answers nothing. An `exec` run whose stdin
//! can't stay open (not a local process) has neither: the schema file is
//! this machine's, and no host reads it.
//!
//! Verified against Codex CLI 0.170.0 (live probes, 2026-11-18; continuing
//! threads across launches, 2026-11-23; the report, 2026-11-23).

pub mod events;
pub mod exec;
pub mod host;

use crate::{
    CodingAgent, Discovery, Launch, LaunchArguments, PermissionMode, Plan, PromptPlacement,
    RunRequest,
};

/// Codex.
#[derive(Debug, Clone, Copy, Default)]
pub struct Codex;

/// The flags that set what a launch may do, removed so a mode's own are
/// the only ones (exact command-line flags: a machine format).
fn drop_access_flags(args: &mut Vec<String>, start: usize) {
    let mut kept: Vec<String> = args[..start.min(args.len())].to_vec();
    let mut index = start;
    while index < args.len() {
        let argument = &args[index];
        let drops_value = matches!(
            argument.as_str(),
            "--sandbox" | "-s" | "--add-dir" | "--output-last-message" | "-o"
        );
        let drops_flag = matches!(
            argument.as_str(),
            "--dangerously-bypass-approvals-and-sandbox"
                | "--dangerously-bypass-hook-trust"
                | "--full-auto"
                | "--ignore-user-config"
                | "--skip-git-repo-check"
        ) || argument.starts_with("--sandbox=")
            || argument.starts_with("-s=")
            || argument.starts_with("--add-dir=")
            || argument.starts_with("--output-last-message=");
        if drops_value {
            index += 2;
            continue;
        }
        if !drops_flag {
            kept.push(argument.clone());
        }
        index += 1;
    }
    *args = kept;
}

/// Its own sandbox, read-only. The owner's config (and its trusted
/// projects) is ignored, so a folder outside a git repository is allowed:
/// the sandbox is the boundary.
fn read_only(args: &mut Vec<String>, start: usize) {
    drop_access_flags(args, start);
    args.extend(
        [
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
        ]
        .iter()
        .map(|arg| arg.to_string()),
    );
}

fn profile_path(path: &str) -> Result<&str, String> {
    if path.contains(['\n', '\r', '=']) {
        Err(format!(
            "path cannot be represented safely in a permission profile: {path:?}"
        ))
    } else {
        Ok(path)
    }
}

/// An invocation-local split read/write profile: Codex refuses a profile it
/// cannot enforce instead of widening access.
fn scoped(
    args: &mut Vec<String>,
    start: usize,
    read_paths: &[String],
    write_paths: &[String],
) -> Result<(), String> {
    if write_paths.is_empty() {
        return Err(
            "a read-write delegation requires at least one explicit write path".to_string(),
        );
    }
    drop_access_flags(args, start);
    const PROFILE: &str = "aidaemon-task";
    let toml = |value: &str| serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
    let mut entries = vec![(":minimal", "read")];
    for path in read_paths {
        entries.push((profile_path(path)?, "read"));
    }
    for path in write_paths {
        let path = profile_path(path)?;
        match entries.iter_mut().find(|(existing, _)| *existing == path) {
            Some(entry) => entry.1 = "write",
            None => entries.push((path, "write")),
        }
    }
    let filesystem = entries
        .into_iter()
        .map(|(path, access)| format!("{}={}", toml(path), toml(access)))
        .collect::<Vec<_>>()
        .join(",");
    args.extend([
        "--ignore-user-config".to_string(),
        "--skip-git-repo-check".to_string(),
        "-c".to_string(),
        format!("default_permissions={}", toml(PROFILE)),
        "-c".to_string(),
        format!("permissions.{PROFILE}.filesystem={{{filesystem}}}"),
        "-c".to_string(),
        format!("permissions.{PROFILE}.network.enabled=false"),
        "-c".to_string(),
        "approval_policy=\"never\"".to_string(),
    ]);
    Ok(())
}

/// The launch says not to keep its thread (`--ephemeral`, Codex's own flag).
fn ephemeral(launch: &Launch) -> bool {
    launch.args[launch.start.min(launch.args.len())..]
        .iter()
        .any(|arg| arg == "--ephemeral")
}

/// Makes a hosted run's `CODEX_HOME` ([`host::CodexHome::prepare`]; a
/// test's own in tests).
type MakeHome<'a> = &'a dyn Fn() -> Result<host::CodexHome, String>;

/// The same run behind `codex app-server`, its requests answered by the
/// host, with its own `CODEX_HOME`: a new thread, or the one `request`
/// continues.
fn hosted(
    launch: &mut Launch,
    request: &RunRequest<'_>,
    asks: host::Asks,
    home: MakeHome<'_>,
) -> Result<Plan, String> {
    let args = host::app_server_args(&launch.args, launch.start, asks)?;
    let home = home()?;
    home.write_mcp_servers(request.mcp_servers)?;
    let thread = match request.resume {
        Some(id) => host::ThreadRequest::Resume {
            id: id.to_string(),
            settings: host::thread_settings(&args[launch.start.min(args.len())..]),
        },
        None => host::ThreadRequest::Start {
            ephemeral: ephemeral(launch),
        },
    };
    launch.args = args;
    tracing::info!("A Codex run is hosted through its app-server");
    Ok(Plan {
        prompt: PromptPlacement::Host,
        env: vec![(
            "CODEX_HOME".to_string(),
            home.path().to_string_lossy().into_owned(),
        )],
        host: Some(Box::new(host::CodexHostStarter {
            working_dir: request.working_dir.map(str::to_string),
            asks,
            thread,
            output_schema: request.report.then(crate::report::schema),
        })),
        keep_alive: Some(Box::new(home)),
        ..Plan::plain()
    })
}

impl CodingAgent for Codex {
    fn reports_steps(&self) -> bool {
        true
    }

    /// `--output-schema` (`exec`) and `outputSchema` (hosted) hold its
    /// final message to the report schema.
    fn reports_in_fields(&self) -> bool {
        true
    }

    fn read_events(&self, line: &str) -> Vec<crate::AgentEvent> {
        events::read_events(line)
    }

    fn name(&self) -> &str {
        "Codex"
    }

    fn mark(&self) -> Option<&str> {
        Some("openai")
    }

    fn launches(&self, program: &str) -> bool {
        program
            .trim_end_matches(".exe")
            .eq_ignore_ascii_case("codex")
    }

    fn modes(&self) -> &[PermissionMode] {
        &[
            PermissionMode::Checked,
            PermissionMode::EveryCommand,
            PermissionMode::FullAccess,
        ]
    }

    fn discovery(&self) -> Option<Discovery> {
        Some(Discovery {
            name: "codex",
            command: "codex",
            args: &["exec", "--json"],
            description: "Codex CLI — OpenAI's AI coding agent",
        })
    }

    fn launch_arguments(&self) -> Option<LaunchArguments> {
        Some(LaunchArguments {
            flags: &["--json", "--skip-git-repo-check", "--ephemeral"],
            valued: &["--model", "-m", "--color"],
            subcommands: &["exec"],
        })
    }

    /// Checked runs unhosted too (its sandbox holds, it just never asks).
    fn can_run(&self, mode: PermissionMode, _read_only: bool, can_host: bool) -> bool {
        mode != PermissionMode::EveryCommand || can_host
    }

    fn account(&self) -> &str {
        "Its own Codex sign-in: ChatGPT or an OpenAI API key."
    }

    /// OpenAI allows ChatGPT plans in third-party apps but points
    /// programmatic Codex use at API keys: a grey area for delegated runs
    /// on a ChatGPT sign-in.
    fn delegation(&self, sign_in: crate::SignInMethod) -> crate::Delegation {
        match sign_in {
            crate::SignInMethod::Own => crate::Delegation::Warned(crate::SignInNote {
                text: "On a ChatGPT sign-in, OpenAI recommends an API key for automated Codex use; \
                       work AIdaemon hands it counts.",
                link: "https://help.openai.com/en/articles/20001542-using-your-chatgpt-plan-in-other-apps-and-sites",
            }),
            _ => crate::Delegation::Allowed,
        }
    }

    fn takes_model(&self) -> bool {
        true
    }

    fn resumes(&self) -> bool {
        true
    }

    fn usage_check(&self) -> Option<crate::UsageCheck> {
        Some(crate::UsageCheck::Codex)
    }

    fn model_listing(&self) -> Option<crate::ModelListing> {
        Some(crate::ModelListing::Codex)
    }

    /// `codex login status` exits 0 when signed in, 1 when not (verified
    /// with an empty `CODEX_HOME`, 0.170.0).
    fn readiness_probe(&self) -> Option<crate::Probe> {
        Some(crate::Probe::Status {
            args: vec!["login".into(), "status".into()],
            read: |code, _| match code {
                Some(0) => crate::Readiness::Ready,
                Some(1) => crate::Readiness::NotUsable {
                    reason: "Codex isn't signed in on this machine.".into(),
                    fix: Some("Run `codex login` on this machine.".into()),
                },
                _ => crate::Readiness::Unchecked,
            },
        })
    }

    fn prepare(&self, request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
        prepare(request, launch, &host::CodexHome::prepare)
    }
}

/// [`Codex::prepare`], its hosted runs' homes made by `home`.
fn prepare(
    request: &RunRequest<'_>,
    launch: &mut Launch,
    home: MakeHome<'_>,
) -> Result<Plan, String> {
    if let Some(model) = request.model {
        crate::agent::set_option(&mut launch.args, launch.start, "--model", &["-m"], model);
    }
    // Read before shaping: a hosted launch no longer carries the flag.
    let ephemeral = ephemeral(launch);
    let mut plan = shape(request, launch, home)?;
    // Both launches keep the thread in the owner's sessions, unless the
    // launch says not to.
    plan.keeps_session = !ephemeral;
    if plan.host.is_some() {
        // Hosted: its thread and schema go in its requests.
        return Ok(plan);
    }
    // `exec`: options first.
    if !request.mcp_servers.is_empty() {
        for server in request.mcp_servers {
            // Codex splits a `-c` key on `.` and keeps quotes, so a quoted
            // segment would name the server `"x"`: the key must be bare.
            let qname = server.name.as_str();
            if qname.is_empty()
                || !qname
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err(format!(
                    "MCP server name `{qname}` is not a bare config key"
                ));
            }
            let qcmd = serde_json::to_string(&server.command).unwrap_or_else(|_| "\"\"".into());
            let qargs = serde_json::to_string(&server.args).unwrap_or_else(|_| "[]".into());
            launch.args.extend([
                "-c".to_string(),
                format!("mcp_servers.{qname}.command={qcmd}"),
                "-c".to_string(),
                format!("mcp_servers.{qname}.args={qargs}"),
            ]);
            if !server.env.is_empty() {
                let env_names: Vec<&str> = server.env.iter().map(|(k, _)| k.as_str()).collect();
                let qvars = serde_json::to_string(&env_names).unwrap_or_else(|_| "[]".into());
                launch.args.extend([
                    "-c".to_string(),
                    format!("mcp_servers.{qname}.env_vars={qvars}"),
                ]);
                plan.env.extend(server.env.iter().cloned());
            }
        }
    }
    if request.report && request.stdin_open {
        match SchemaFile::write() {
            Ok(file) => {
                let path = file.path.to_string_lossy().into_owned();
                crate::agent::set_option(
                    &mut launch.args,
                    launch.start,
                    "--output-schema",
                    &[],
                    &path,
                );
                plan.keep_alive = Some(match plan.keep_alive.take() {
                    Some(kept) => Box::new((kept, file)),
                    None => Box::new(file),
                });
            }
            Err(error) => {
                tracing::warn!(%error, "A Codex run reports in text: its schema file could not be written");
            }
        }
    }
    if let Some(session) = request.resume {
        // After every option, before the task (its last argument). A
        // hosted run names it in its `thread/resume` instead.
        launch
            .args
            .extend(["resume".to_string(), session.to_string()]);
    }
    if request.stdin_open {
        plan.host = Some(Box::new(exec::ExecHostStarter));
    }
    Ok(plan)
}

/// The report schema in a file of its own for `--output-schema`, removed
/// when the run is over.
#[derive(Debug)]
struct SchemaFile {
    path: std::path::PathBuf,
}

impl SchemaFile {
    fn write() -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "aidaemon-codex-report-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, crate::report::schema().to_string())?;
        Ok(Self { path })
    }
}

impl Drop for SchemaFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `launch` in `request`'s mode, and how the run goes.
fn shape(
    request: &RunRequest<'_>,
    launch: &mut Launch,
    home: MakeHome<'_>,
) -> Result<Plan, String> {
    if !request.enforce {
        return Ok(Plan::plain());
    }
    match request.mode {
        PermissionMode::FullAccess => {
            drop_access_flags(&mut launch.args, launch.start);
            launch
                .args
                .push("--dangerously-bypass-approvals-and-sandbox".to_string());
            Ok(Plan::plain())
        }
        PermissionMode::EveryCommand => {
            if !request.can_host {
                return Err(
                    "Codex can check every command only where its requests can be answered"
                        .to_string(),
                );
            }
            read_only(&mut launch.args, launch.start);
            hosted(launch, request, host::Asks::EveryCommand, home)
        }
        PermissionMode::Checked if request.read_only => {
            read_only(&mut launch.args, launch.start);
            Ok(Plan::plain())
        }
        PermissionMode::Checked => {
            // No write paths: it may change anything in its folder (a
            // conversation set to it), so its folder is its write root.
            let folder: Vec<String> = match (request.write_paths, request.working_dir) {
                ([], Some(dir)) => vec![dir.to_string()],
                _ => Vec::new(),
            };
            let write_paths = if folder.is_empty() {
                request.write_paths
            } else {
                folder.as_slice()
            };
            scoped(
                &mut launch.args,
                launch.start,
                request.read_paths,
                write_paths,
            )?;
            if !request.can_host {
                return Ok(Plan::plain());
            }
            // A launch the app-server cannot take, or no sign-in to
            // link, stays on `exec`: confined, it just never asks. It
            // continues a thread all the same, as `exec ... resume`.
            let exec = launch.clone();
            hosted(launch, request, host::Asks::BeyondSandbox, home).or_else(|reason| {
                tracing::info!(%reason, "A Codex run stays on exec");
                *launch = exec;
                Ok(Plan::plain())
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn request<'a>(
        mode: PermissionMode,
        read_only: bool,
        write_paths: &'a [String],
    ) -> RunRequest<'a> {
        RunRequest {
            mode,
            enforce: true,
            read_only,
            working_dir: Some("/w/project"),
            read_paths: &[],
            write_paths,
            can_host: false,
            stdin_open: true,
            model: None,
            resume: None,
            report: false,
            mcp_servers: &[],
        }
    }

    #[test]
    fn checked_unhosted_is_its_own_sandbox_and_never_asks() {
        let mut launch = Launch {
            args: strings(&[
                "exec",
                "--json",
                "--full-auto",
                "--sandbox",
                "danger-full-access",
            ]),
            start: 0,
        };
        Codex
            .prepare(&request(PermissionMode::Checked, true, &[]), &mut launch)
            .unwrap();
        let joined = launch.args.join(" ");
        assert!(joined.ends_with("--sandbox read-only"), "{joined}");
        assert!(!joined.contains("--full-auto") && !joined.contains("danger-full-access"));

        let writes = ["/w/project/out.md".to_string()];
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = Codex
            .prepare(
                &request(PermissionMode::Checked, false, &writes),
                &mut launch,
            )
            .unwrap();
        let joined = launch.args.join(" ");
        assert!(
            joined.contains("\"/w/project/out.md\"=\"write\""),
            "{joined}"
        );
        assert!(joined.contains("network.enabled=false"));
        assert!(joined.contains("approval_policy=\"never\""));
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);
        // A write with no path given may change anything in its folder: its
        // folder is its write root (a conversation set to it).
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        Codex
            .prepare(&request(PermissionMode::Checked, false, &[]), &mut launch)
            .unwrap();
        let joined = launch.args.join(" ");
        assert!(joined.contains("\"/w/project\"=\"write\""), "{joined}");
        // With no folder either, it is refused.
        let mut nowhere = request(PermissionMode::Checked, false, &[]);
        nowhere.working_dir = None;
        assert!(Codex.prepare(&nowhere, &mut launch).is_err());
    }

    #[test]
    fn every_command_needs_its_host_and_full_access_bypasses_everything() {
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        assert!(
            Codex
                .prepare(
                    &request(PermissionMode::EveryCommand, false, &[]),
                    &mut launch
                )
                .is_err()
        );
        assert!(!Codex.can_run(PermissionMode::EveryCommand, false, false));
        assert!(Codex.can_run(PermissionMode::Checked, false, false));

        let mut launch = Launch {
            args: strings(&["exec", "--json", "--sandbox", "read-only"]),
            start: 0,
        };
        let plan = Codex
            .prepare(
                &request(PermissionMode::FullAccess, false, &[]),
                &mut launch,
            )
            .unwrap();
        assert_eq!(
            launch.args,
            strings(&[
                "exec",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox"
            ])
        );
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);
    }

    #[test]
    fn an_env_launcher_is_kept_in_front() {
        let mut launch = Launch {
            args: strings(&["FOO=1", "codex", "exec", "--json"]),
            start: 2,
        };
        Codex
            .prepare(&request(PermissionMode::Checked, true, &[]), &mut launch)
            .unwrap();
        assert_eq!(&launch.args[..3], &strings(&["FOO=1", "codex", "exec"])[..]);
    }

    #[test]
    fn a_run_continues_the_thread_it_names_with_its_options_before_resume() {
        assert!(Codex.resumes());
        let mut asked = request(PermissionMode::FullAccess, false, &[]);
        asked.resume = Some("s-1");
        asked.model = Some("m-2");
        let mut launch = Launch {
            args: strings(&["exec", "--json", "-m", "m-1"]),
            start: 0,
        };
        let plan = Codex.prepare(&asked, &mut launch).unwrap();
        assert_eq!(
            launch.args,
            strings(&[
                "exec",
                "--json",
                "--model",
                "m-2",
                "--dangerously-bypass-approvals-and-sandbox",
                "resume",
                "s-1",
            ])
        );
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);

        // Read-only checked: its sandbox, then the thread.
        let mut asked = request(PermissionMode::Checked, true, &[]);
        asked.resume = Some("s-1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        Codex.prepare(&asked, &mut launch).unwrap();
        assert!(
            launch
                .args
                .join(" ")
                .ends_with("--sandbox read-only resume s-1"),
            "{:?}",
            launch.args
        );

        // Outside the checks, too.
        let mut asked = request(PermissionMode::Checked, false, &[]);
        asked.enforce = false;
        asked.resume = Some("s-1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        Codex.prepare(&asked, &mut launch).unwrap();
        assert_eq!(launch.args, strings(&["exec", "--json", "resume", "s-1"]));
    }

    /// A signed-in owner's Codex home in a folder of its own, removed by
    /// the caller.
    fn owner_home() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("aidaemon-test-owner-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("auth.json"), "{}").unwrap();
        path
    }

    struct AllowAll;

    #[async_trait::async_trait]
    impl crate::ActionJudge for AllowAll {
        async fn judge(&self, _action: crate::AgentAction) -> crate::Verdict {
            crate::Verdict::Allow
        }
    }

    /// The thread request a hosted plan's host sends once Codex answers
    /// its handshake.
    #[cfg(unix)]
    async fn thread_request(plan: Plan) -> serde_json::Value {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut child = tokio::process::Command::new("cat")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("cat");
        let host = plan
            .host
            .expect("a hosted plan")
            .start(
                child.stdin.take().unwrap(),
                std::sync::Arc::new(AllowAll),
                "go on",
            )
            .await;
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        // Its handshake, then the thread once Codex answers it.
        assert!(
            sent.next_line()
                .await
                .unwrap()
                .unwrap()
                .contains("initialize")
        );
        host.on_line(r#"{"id":1,"result":{}}"#).await;
        assert!(
            sent.next_line()
                .await
                .unwrap()
                .unwrap()
                .contains("initialized")
        );
        let line = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        host.close().await;
        let _ = child.wait().await;
        line
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_hosted_run_keeps_its_thread_and_continues_one_through_its_host() {
        let owner = owner_home();
        let home = || host::CodexHome::linked_to(&owner);
        let writes = ["/w/project/out.md".to_string()];

        // Checked: a new stored thread.
        let mut asked = request(PermissionMode::Checked, false, &writes);
        asked.can_host = true;
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        assert_eq!(launch.args[0], "app-server");
        assert!(plan.keeps_session, "its thread is in the owner's sessions");
        let start = thread_request(plan).await;
        assert_eq!(start["method"], "thread/start");
        assert!(start["params"].get("ephemeral").is_none());

        // Checked, continuing a thread: still hosted, under its profile and
        // model whatever the thread last ran with.
        asked.resume = Some("thread-a1");
        asked.model = Some("m-2");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        assert_eq!(launch.args[0], "app-server");
        assert!(
            !launch
                .args
                .iter()
                .any(|arg| arg == "resume" || arg == "thread-a1")
        );
        assert_eq!(plan.prompt, PromptPlacement::Host);
        assert!(plan.keeps_session);
        let resume = thread_request(plan).await;
        assert_eq!(resume["method"], "thread/resume");
        assert_eq!(resume["params"]["threadId"], "thread-a1");
        assert_eq!(resume["params"]["approvalPolicy"], "on-request");
        assert_eq!(
            resume["params"]["config"],
            serde_json::json!({"model": "m-2", "default_permissions": "aidaemon-task"})
        );

        // Every command, continuing a thread: hosted, read-only, every
        // command asked.
        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        asked.resume = Some("thread-a1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        assert!(plan.keeps_session);
        let resume = thread_request(plan).await;
        assert_eq!(resume["method"], "thread/resume");
        assert_eq!(resume["params"]["approvalPolicy"], "untrusted");
        assert_eq!(
            resume["params"]["config"],
            serde_json::json!({"sandbox_mode": "read-only"})
        );

        // A launch told not to keep its thread keeps none, hosted too.
        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        let mut launch = Launch {
            args: strings(&["exec", "--json", "--ephemeral"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        assert!(!plan.keeps_session);
        let start = thread_request(plan).await;
        assert_eq!(start["params"]["ephemeral"], true);
        std::fs::remove_dir_all(&owner).unwrap();
    }

    #[test]
    fn without_a_sign_in_to_link_a_checked_run_continues_on_exec() {
        let no_home = || Err::<host::CodexHome, _>("no Codex sign-in".to_string());
        let writes = ["/w/project/out.md".to_string()];
        let mut asked = request(PermissionMode::Checked, false, &writes);
        asked.can_host = true;
        asked.resume = Some("thread-a1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &no_home).unwrap();
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);
        assert!(plan.keeps_session);
        assert_eq!(launch.args[0], "exec");
        assert_eq!(
            &launch.args[launch.args.len() - 2..],
            &strings(&["resume", "thread-a1"])[..]
        );

        // Every command needs its host.
        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        asked.resume = Some("thread-a1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        assert!(prepare(&asked, &mut launch, &no_home).is_err());

        // An exec run told not to keep its thread keeps none.
        let asked = request(PermissionMode::Checked, true, &[]);
        let mut launch = Launch {
            args: strings(&["exec", "--json", "--ephemeral"]),
            start: 0,
        };
        assert!(
            !prepare(&asked, &mut launch, &no_home)
                .unwrap()
                .keeps_session
        );
    }

    #[test]
    fn an_exec_run_asked_for_a_report_names_its_schema_file_before_resume() {
        assert!(Codex.reports_in_fields());
        let mut asked = request(PermissionMode::Checked, true, &[]);
        asked.report = true;
        asked.resume = Some("s-1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = Codex.prepare(&asked, &mut launch).unwrap();
        let args = &launch.args;
        let at = args
            .iter()
            .position(|arg| arg == "--output-schema")
            .expect("its schema");
        let path = std::path::PathBuf::from(&args[at + 1]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(&path).unwrap())
                .unwrap(),
            crate::report::schema()
        );
        // Options before `resume`, the thread last (the task follows it).
        assert_eq!(&args[at + 2..], &strings(&["resume", "s-1"])[..]);
        assert!(args.join(" ").contains("--sandbox read-only"));
        // Read through its exec host; the task stays its last argument.
        assert!(plan.host.is_some());
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);
        // The file lasts as long as the run.
        drop(plan);
        assert!(!path.exists());

        // No report asked: no schema, still read through its host.
        asked.report = false;
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = Codex.prepare(&asked, &mut launch).unwrap();
        assert!(!launch.args.iter().any(|arg| arg == "--output-schema"));
        assert!(plan.host.is_some() && plan.keep_alive.is_none());

        // Not a local process: no file of this machine's, no host.
        asked.report = true;
        asked.stdin_open = false;
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = Codex.prepare(&asked, &mut launch).unwrap();
        assert!(!launch.args.iter().any(|arg| arg == "--output-schema"));
        assert!(plan.host.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_hosted_run_asked_for_a_report_sends_its_schema_with_its_turn() {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let owner = owner_home();
        let home = || host::CodexHome::linked_to(&owner);
        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        asked.report = true;
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        assert_eq!(launch.args[0], "app-server");
        assert!(!launch.args.iter().any(|arg| arg == "--output-schema"));
        let mut child = tokio::process::Command::new("cat")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("cat");
        let host = plan
            .host
            .expect("a hosted plan")
            .start(
                child.stdin.take().unwrap(),
                std::sync::Arc::new(AllowAll),
                "go on",
            )
            .await;
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        sent.next_line().await.unwrap();
        host.on_line(r#"{"id":1,"result":{}}"#).await;
        sent.next_line().await.unwrap();
        sent.next_line().await.unwrap();
        host.on_line(r#"{"id":2,"result":{"thread":{"id":"t1"}}}"#)
            .await;
        let turn: serde_json::Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(turn["method"], "turn/start");
        assert_eq!(turn["params"]["outputSchema"], crate::report::schema());
        host.close().await;
        let _ = child.wait().await;
        std::fs::remove_dir_all(&owner).unwrap();
    }

    #[test]
    fn injected_mcp_servers_write_codex_home_config_and_exec_env_vars() {
        let endpoints = vec![crate::McpEndpoint {
            name: "aidaemon_peers".into(),
            command: "/usr/local/bin/aidaemon".into(),
            args: vec![
                "mcp-bridge".into(),
                "--addr".into(),
                "127.0.0.1:4242".into(),
            ],
            env: vec![("AIDAEMON_MCP_TOKEN".into(), "synthetic-token".into())],
        }];

        // Hosted path: writes config.toml inside ephemeral CODEX_HOME (not in argv).
        let owner =
            std::env::temp_dir().join(format!("aidaemon-codex-mcp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&owner).unwrap();
        std::fs::write(owner.join("auth.json"), "{}").unwrap();
        let home = || host::CodexHome::linked_to(&owner);
        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        asked.mcp_servers = &endpoints;
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = prepare(&asked, &mut launch, &home).unwrap();
        let codex_home = plan
            .env
            .iter()
            .find(|(k, _)| k == "CODEX_HOME")
            .map(|(_, v)| std::path::PathBuf::from(v))
            .expect("CODEX_HOME env");
        let cfg = std::fs::read_to_string(codex_home.join("config.toml")).unwrap();
        assert!(cfg.contains("[mcp_servers.\"aidaemon_peers\"]"), "{cfg}");
        assert!(cfg.contains("AIDAEMON_MCP_TOKEN"), "{cfg}");
        drop(plan);
        std::fs::remove_dir_all(&owner).unwrap();

        // Exec path: keeps token in Plan::env and uses env_vars in -c args.
        let mut exec_req = request(PermissionMode::FullAccess, false, &[]);
        exec_req.mcp_servers = &endpoints;
        let mut exec_launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let exec_plan = prepare(&exec_req, &mut exec_launch, &host::CodexHome::prepare).unwrap();
        let joined = exec_launch.args.join(" ");
        assert!(
            joined.contains("mcp_servers.aidaemon_peers.command="),
            "{joined}"
        );
        assert!(
            joined.contains("env_vars=[\"AIDAEMON_MCP_TOKEN\"]"),
            "{joined}"
        );
        assert!(
            !joined.contains("synthetic-token"),
            "token must never appear in argv"
        );
        assert!(
            exec_plan
                .env
                .iter()
                .any(|(k, v)| k == "AIDAEMON_MCP_TOKEN" && v == "synthetic-token")
        );
    }
}
