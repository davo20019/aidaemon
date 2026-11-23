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
//! A run continues an earlier session as `codex exec <options> resume
//! <thread id> <task>`: the options stay before `resume`, which refuses
//! `--sandbox` after it. Only an `exec` run keeps its session where a later
//! one finds it (the owner's `CODEX_HOME`); a hosted run's thread is
//! ephemeral, in a `CODEX_HOME` of its own, so a run that continues one
//! stays on `exec`: checked, it just never asks, and every-command mode,
//! which needs the host, cannot continue one.
//!
//! Verified against Codex CLI 0.170.0 (live probes, 2026-11-18).

pub mod events;
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

/// The same run behind `codex app-server`, its requests answered by the
/// host, with its own `CODEX_HOME`.
fn hosted(launch: &mut Launch, request: &RunRequest<'_>, asks: host::Asks) -> Result<Plan, String> {
    let args = host::app_server_args(&launch.args, launch.start, asks)?;
    let home = host::CodexHome::prepare()?;
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
        })),
        keep_alive: Some(Box::new(home)),
        ..Plan::plain()
    })
}

impl CodingAgent for Codex {
    fn reports_steps(&self) -> bool {
        true
    }

    fn read_events(&self, line: &str) -> Vec<crate::AgentEvent> {
        events::read_events(line)
    }

    fn name(&self) -> &str {
        "Codex"
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

    fn takes_model(&self) -> bool {
        true
    }

    fn resumes(&self) -> bool {
        true
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
        if let Some(model) = request.model {
            crate::agent::set_option(&mut launch.args, launch.start, "--model", &["-m"], model);
        }
        let mut plan = shape(request, launch)?;
        // A hosted run's thread is ephemeral, in a home of its own thrown
        // away at its end; `exec` keeps it in the owner's, unless the
        // launch says not to (`--ephemeral`, its own flag).
        plan.keeps_session = plan.host.is_none()
            && !launch.args[launch.start..]
                .iter()
                .any(|a| a == "--ephemeral");
        if let Some(session) = request.resume {
            // After every option, before the task (its last argument).
            launch
                .args
                .extend(["resume".to_string(), session.to_string()]);
        }
        Ok(plan)
    }
}

/// `launch` in `request`'s mode, and how the run goes.
fn shape(request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
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
            if request.resume.is_some() {
                return Err(
                    "Codex can't continue an earlier session while it checks every command: \
                         its prompts need a host, and a hosted run can't continue one"
                        .to_string(),
                );
            }
            if !request.can_host {
                return Err(
                    "Codex can check every command only where its requests can be answered"
                        .to_string(),
                );
            }
            read_only(&mut launch.args, launch.start);
            hosted(launch, request, host::Asks::EveryCommand)
        }
        PermissionMode::Checked if request.read_only => {
            read_only(&mut launch.args, launch.start);
            Ok(Plan::plain())
        }
        PermissionMode::Checked => {
            scoped(
                &mut launch.args,
                launch.start,
                request.read_paths,
                request.write_paths,
            )?;
            if !request.can_host || request.resume.is_some() {
                return Ok(Plan::plain());
            }
            // A launch the app-server cannot take, or no sign-in to
            // link, stays on `exec`: confined, it just never asks.
            let exec = launch.clone();
            hosted(launch, request, host::Asks::BeyondSandbox).or_else(|reason| {
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
        assert!(plan.host.is_none());
        // A write with no path to write is refused.
        assert!(
            Codex
                .prepare(&request(PermissionMode::Checked, false, &[]), &mut launch)
                .is_err()
        );
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
        assert!(plan.host.is_none());
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

    #[test]
    fn a_continued_run_stays_on_exec_and_every_command_cannot_continue_one() {
        let writes = ["/w/project/out.md".to_string()];
        let mut asked = request(PermissionMode::Checked, false, &writes);
        asked.can_host = true;
        asked.resume = Some("s-1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        let plan = Codex.prepare(&asked, &mut launch).unwrap();
        assert!(plan.host.is_none());
        assert!(
            plan.keeps_session,
            "exec keeps its thread for the next follow-up"
        );
        assert_eq!(launch.args[0], "exec");
        assert!(!launch.args.iter().any(|arg| arg == "app-server"));
        assert_eq!(
            &launch.args[launch.args.len() - 2..],
            &strings(&["resume", "s-1"])[..]
        );

        let mut asked = request(PermissionMode::EveryCommand, false, &[]);
        asked.can_host = true;
        asked.resume = Some("s-1");
        let mut launch = Launch {
            args: strings(&["exec", "--json"]),
            start: 0,
        };
        assert!(Codex.prepare(&asked, &mut launch).is_err());

        // A launch told not to keep its thread keeps none.
        let mut asked = request(PermissionMode::Checked, true, &[]);
        asked.resume = None;
        let mut launch = Launch {
            args: strings(&["exec", "--json", "--ephemeral"]),
            start: 0,
        };
        assert!(!Codex.prepare(&asked, &mut launch).unwrap().keeps_session);
    }
}
