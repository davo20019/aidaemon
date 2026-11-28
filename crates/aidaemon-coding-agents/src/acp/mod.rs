//! Any coding agent that speaks the Agent Client Protocol: each action it
//! takes is asked first ([`host::AcpHost`]), so every one can be judged.
//!
//! - Every command (its default): each permission request is judged.
//! - Full access: each is allowed as asked, unjudged.
//!
//! There is no checked mode: the protocol has no sandbox of its own, and
//! an agent's own one is not trusted unless verified (`agy --sandbox` still
//! wrote outside its folder once a command was allowed; live probe,
//! 2026-11-18).
//!
//! A run can continue a session an earlier run started: its host loads it
//! (`session/load`) in place of opening a new one. Only an agent whose
//! `initialize` reply says it can load sessions (`loadSession`) does, so
//! that is known once it runs: its session event says whether a later run
//! can continue it, and a run asked to continue one with an agent that
//! can't ends before its task is sent.
//!
//! [`AcpAgent::antigravity`], [`AcpAgent::gemini`] and [`AcpAgent::opencode`]
//! are the ones known;
//! [`AcpAgent::custom`] is any other the owner names.

pub mod events;
pub mod host;
mod opencode;

use crate::{
    CodingAgent, Discovery, Launch, LaunchArguments, PermissionMode, Plan, PromptPlacement,
    RunRequest,
};

/// Where `agy acp` keeps its data when it runs on a Gemini API key: a
/// folder of its own under `~/.gemini` (`--app-data-dir`), whose
/// `settings.json` names the key as its sign-in. Its own folder
/// (`antigravity-cli`) keeps the owner's saved sign-in untouched.
const AGY_KEY_FOLDER: &str = "aidaemon-agy";

/// A coding agent run over the Agent Client Protocol.
#[derive(Debug, Clone)]
pub struct AcpAgent {
    name: String,
    program: String,
    /// The arguments that start its protocol server (`acp`, `--acp`).
    serve: Vec<String>,
    discovery: Option<Discovery>,
    arguments: Option<LaunchArguments>,
    /// Whose account it works with, in plain words.
    account: &'static str,
    /// What a client draws for it ([`CodingAgent::mark`]).
    mark: Option<&'static str>,
    /// Whether this agent can authenticate with a Gemini API key supplied by
    /// AIdaemon (`GEMINI_API_KEY`).
    supports_gemini_key: bool,
    /// When set, the agent keeps its Gemini-API-key settings in a dedicated
    /// folder under `~/.gemini` (`--app-data-dir`), apart from the owner's own
    /// sessions with it.
    gemini_key_folder: Option<&'static str>,
    /// Sign-in method used when the owner has not chosen one explicitly.
    default_sign_in: crate::SignInMethod,
    /// Whether the embedding agent may hand it work on its own sign-in
    /// (its maker's terms); a key AIdaemon hands it is always allowed.
    own_delegation: crate::Delegation,
    /// Its own sign-in holds one per model provider (OpenCode): work handed
    /// to it runs without those on a subscription plan.
    per_provider_sign_ins: bool,
}

/// Google's Antigravity FAQ and Terms (section 6): wrapping `agy` in a
/// third-party agent on a consumer sign-in is not supported and risks
/// suspension; a Gemini API key is the supported path. Whether an agent the
/// owner watches may hand it work is unsettled, so that alone is allowed,
/// with this word each time.
const AGY_OWN_SIGN_IN: crate::SignInNote = crate::SignInNote {
    text: "Google's terms may not allow other AI agents to hand it work on its own sign-in: \
           AIdaemon never does in the background, and agents in a Code session you watch do so \
           at your account's risk. A Gemini API key avoids that.",
    link: "https://antigravity.google/docs/faq",
};

/// OpenCode signs in to each model provider on its own. A subscription
/// plan's OAuth sign-in (a Claude plan's terms bar other tools from using
/// it) is turned off when another agent hands it work.
const OPENCODE_OWN_SIGN_INS: crate::SignInNote = crate::SignInNote {
    text: "When another agent hands it work, AIdaemon turns off the providers it is signed in \
           to with a subscription plan (their terms may not allow it), so that work runs only \
           on its API-key providers. Your own prompts to it use them all.",
    link: "https://opencode.ai/docs/providers/",
};

impl AcpAgent {
    /// Any agent: its name, the program it launches, and the arguments
    /// that start its protocol server (none when its launch already does).
    pub fn custom(name: &str, program: &str, serve: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            program: program.to_string(),
            serve: serve.iter().map(|arg| arg.to_string()).collect(),
            discovery: None,
            arguments: None,
            account: "",
            mark: None,
            supports_gemini_key: false,
            gemini_key_folder: None,
            default_sign_in: crate::SignInMethod::Own,
            own_delegation: crate::Delegation::Allowed,
            per_provider_sign_ins: false,
        }
    }

    /// Antigravity CLI (`agy acp`). Uses its own sign-in by default; owners
    /// can switch to a Gemini API key in Settings (see [`CodingAgent::sign_in_note`]).
    pub fn antigravity() -> Self {
        Self {
            discovery: Some(Discovery {
                name: "agy",
                command: "agy",
                args: &["acp"],
                description: "Antigravity CLI — Google's AI coding agent",
            }),
            arguments: Some(LaunchArguments {
                flags: &[],
                valued: &["--agent", "--model"],
                subcommands: &["acp"],
            }),
            account: "Its own Antigravity sign-in, or a Gemini API key you give AIdaemon.",
            mark: Some("gemini"),
            supports_gemini_key: true,
            gemini_key_folder: Some(AGY_KEY_FOLDER),
            default_sign_in: crate::SignInMethod::Own,
            own_delegation: crate::Delegation::WhileWatched(AGY_OWN_SIGN_IN),
            ..Self::custom("Antigravity CLI", "agy", &["acp"])
        }
    }

    /// Gemini CLI (`gemini --acp`). Since 2026-06-18 Google serves it only
    /// to enterprise licences and paid API keys: a personal sign-in is
    /// refused a session, which its readiness check reports. Owners may also
    /// supply a Gemini API key directly in Settings.
    pub fn gemini() -> Self {
        Self {
            discovery: Some(Discovery {
                name: "gemini",
                command: "gemini",
                args: &["--acp"],
                description: "Gemini CLI — Google's AI coding agent",
            }),
            arguments: Some(LaunchArguments {
                flags: &["--acp", "--experimental-acp", "--debug", "-d"],
                valued: &["--model", "-m"],
                subcommands: &[],
            }),
            account: "Its own Gemini CLI sign-in (enterprise or API key), or a Gemini API key you give AIdaemon.",
            mark: Some("gemini"),
            supports_gemini_key: true,
            ..Self::custom("Gemini CLI", "gemini", &["--acp"])
        }
    }
    /// OpenCode (`opencode acp`): any model provider it is logged in to.
    /// Work another agent hands it runs only on its API-key providers
    /// (`opencode::delegated`).
    pub fn opencode() -> Self {
        Self {
            discovery: Some(Discovery {
                name: "opencode",
                command: "opencode",
                args: &["acp"],
                description: "OpenCode — an open-source coding agent for any model provider",
            }),
            arguments: Some(LaunchArguments {
                flags: &["--pure", "--print-logs"],
                valued: &["--log-level"],
                subcommands: &["acp"],
            }),
            account: "Its own OpenCode sign-ins, one per model provider (API keys or plans).",
            own_delegation: crate::Delegation::Warned(OPENCODE_OWN_SIGN_INS),
            per_provider_sign_ins: true,
            ..Self::custom("OpenCode", "opencode", &["acp"])
        }
    }
}

impl CodingAgent for AcpAgent {
    fn reports_steps(&self) -> bool {
        true
    }

    fn read_events(&self, line: &str) -> Vec<crate::AgentEvent> {
        events::read_events(line)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn mark(&self) -> Option<&str> {
        self.mark
    }

    fn launches(&self, program: &str) -> bool {
        program
            .trim_end_matches(".exe")
            .eq_ignore_ascii_case(&self.program)
    }

    fn modes(&self) -> &[PermissionMode] {
        &[PermissionMode::EveryCommand, PermissionMode::FullAccess]
    }

    /// Every action asked: the safest it offers.
    fn default_mode(&self) -> Option<PermissionMode> {
        Some(PermissionMode::EveryCommand)
    }

    fn note(&self) -> &str {
        "It has no sandbox AIDaemon can rely on, so every action it takes is checked."
    }

    fn discovery(&self) -> Option<Discovery> {
        self.discovery
    }

    fn launch_arguments(&self) -> Option<LaunchArguments> {
        self.arguments
    }

    /// Its protocol needs stdin held open; checking also needs a judge.
    fn can_run(&self, mode: PermissionMode, _read_only: bool, can_host: bool) -> bool {
        match mode {
            PermissionMode::FullAccess => true,
            PermissionMode::EveryCommand => can_host,
            PermissionMode::Checked => false,
        }
    }

    fn account(&self) -> &str {
        self.account
    }

    fn sign_ins(&self) -> &[crate::SignInMethod] {
        if self.supports_gemini_key {
            &[crate::SignInMethod::GeminiApiKey]
        } else {
            &[]
        }
    }

    fn default_sign_in(&self) -> crate::SignInMethod {
        self.default_sign_in
    }

    /// Its own sign-in follows its maker's terms (`own_delegation`); a key
    /// AIdaemon hands it is an API sign-in, which they allow.
    fn delegation(&self, sign_in: crate::SignInMethod) -> crate::Delegation {
        match sign_in {
            crate::SignInMethod::Own => self.own_delegation,
            crate::SignInMethod::GeminiApiKey => crate::Delegation::Allowed,
        }
    }

    /// Its own sign-in without the providers on a plan when it holds one
    /// per provider (OpenCode); otherwise as [`Self::signed_in`].
    fn signed_in_delegated(&self, sign_in: &crate::SignIn) -> Result<crate::SignedIn, String> {
        match sign_in {
            crate::SignIn::Own if self.per_provider_sign_ins => opencode::delegated(),
            other => self.signed_in(other),
        }
    }

    /// On a Gemini API key: when a dedicated data folder is configured
    /// (`agy`), its settings name the key as its sign-in (`agy acp` reads its
    /// sign-in from there and the key from `GEMINI_API_KEY`); otherwise
    /// (`gemini`) the key is passed via `GEMINI_API_KEY`. In both cases other
    /// Google credential environment variables are stripped.
    fn signed_in(&self, sign_in: &crate::SignIn) -> Result<crate::SignedIn, String> {
        match (sign_in, self.supports_gemini_key, self.gemini_key_folder) {
            (crate::SignIn::Own, _, _) => Ok(crate::SignedIn::default()),
            (crate::SignIn::GeminiApiKey(key), true, Some(folder)) => {
                let gemini = dirs::home_dir()
                    .ok_or("there is no home folder to keep its settings in")?
                    .join(".gemini");
                key_settings(&gemini.join(folder))?;
                Ok(crate::SignedIn {
                    args: vec!["--app-data-dir".into(), folder.into()],
                    env: vec![("GEMINI_API_KEY".into(), key.clone())],
                    env_remove: crate::sign_in::GOOGLE_CREDENTIAL_VARS
                        .iter()
                        .map(|name| name.to_string())
                        .collect(),
                })
            }
            (crate::SignIn::GeminiApiKey(key), true, None) => Ok(crate::SignedIn {
                args: Vec::new(),
                env: vec![("GEMINI_API_KEY".into(), key.clone())],
                env_remove: crate::sign_in::GOOGLE_CREDENTIAL_VARS
                    .iter()
                    .map(|name| name.to_string())
                    .collect(),
            }),
            (other, false, _) => Err(format!(
                "{} can't sign in with {}",
                self.name,
                other.method().as_str()
            )),
        }
    }

    /// Set over the protocol once its session opens (`session/set_model`).
    fn takes_model(&self) -> bool {
        true
    }

    /// Over the protocol (`session/load`), when the agent offers it: that
    /// is known only once it runs, so its session event says whether a
    /// run's session can be continued, and its host ends a run asked to
    /// continue one with an agent that can't.
    fn resumes(&self) -> bool {
        true
    }

    /// A session it opens (no prompt sent), or its own reason not to.
    fn readiness_probe(&self) -> Option<crate::Probe> {
        Some(crate::Probe::AcpSession {
            serve: self.serve.clone(),
            signed: crate::SignedIn::default(),
        })
    }

    fn prepare(&self, request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
        if !request.stdin_open {
            return Err(format!(
                "{} runs only where its stdin can stay open (a local process)",
                self.name
            ));
        }
        if request.enforce && request.mode != PermissionMode::FullAccess && !request.can_host {
            return Err(format!(
                "{}'s requests can't be answered here (its run needs a checked turn)",
                self.name
            ));
        }
        if request.enforce && request.mode == PermissionMode::Checked {
            return Err(format!("{} has no checked mode", self.name));
        }
        // Its protocol server, unless the launch already starts it.
        let own = &launch.args[launch.start.min(launch.args.len())..];
        if !self.serve.iter().all(|arg| own.contains(arg)) {
            launch.args.extend(self.serve.iter().cloned());
        }
        Ok(Plan {
            prompt: PromptPlacement::Host,
            host: Some(Box::new(host::AcpHostStarter {
                agent: self.name.clone(),
                working_dir: request.working_dir.map(str::to_string),
                model: request.model.map(str::to_string),
                resume: request.resume.map(str::to_string),
                mcp_servers: request.mcp_servers.to_vec(),
            })),
            // Whether its session can be continued is the agent's word once
            // it runs (its session event's `resumable`).
            ..Plan::plain()
        })
    }
}

/// Make `folder`'s `settings.json` name a Gemini API key as its sign-in,
/// keeping whatever else it holds.
fn key_settings(folder: &std::path::Path) -> Result<(), String> {
    let path = folder.join("settings.json");
    let mut settings = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    if settings
        .pointer("/auth/type")
        .and_then(serde_json::Value::as_str)
        == Some("gemini-api-key")
    {
        return Ok(());
    }
    if let Some(object) = settings.as_object_mut() {
        let auth = object
            .entry("auth")
            .or_insert_with(|| serde_json::json!({}));
        if !auth.is_object() {
            *auth = serde_json::json!({});
        }
        auth["type"] = serde_json::json!("gemini-api-key");
    }
    let failed = |error: std::io::Error| {
        format!(
            "its settings in {} could not be written: {error}",
            folder.display()
        )
    };
    std::fs::create_dir_all(folder).map_err(failed)?;
    let text = serde_json::to_string_pretty(&settings).unwrap_or_default();
    std::fs::write(&path, text + "\n").map_err(failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: PermissionMode, can_host: bool) -> RunRequest<'static> {
        RunRequest {
            mode,
            enforce: true,
            read_only: false,
            working_dir: Some("/w/project"),
            read_paths: &[],
            write_paths: &[],
            can_host,
            stdin_open: true,
            model: None,
            resume: None,
            report: false,
            mcp_servers: &[],
        }
    }

    #[test]
    fn it_starts_its_protocol_server_and_is_hosted_in_every_mode_it_offers() {
        let agent = AcpAgent::antigravity();
        for mode in [PermissionMode::EveryCommand, PermissionMode::FullAccess] {
            let mut launch = Launch {
                args: Vec::new(),
                start: 0,
            };
            let plan = agent.prepare(&request(mode, true), &mut launch).unwrap();
            assert_eq!(launch.args, ["acp"]);
            assert_eq!(plan.prompt, PromptPlacement::Host);
            assert!(plan.host.is_some());
        }
        // A launch that already starts it is left as it is.
        let mut launch = Launch {
            args: vec!["acp".into(), "--model".into(), "m".into()],
            start: 0,
        };
        agent
            .prepare(&request(PermissionMode::EveryCommand, true), &mut launch)
            .unwrap();
        assert_eq!(launch.args, ["acp", "--model", "m"]);
    }

    #[test]
    fn checking_needs_a_judge_and_there_is_no_checked_mode() {
        let agent = AcpAgent::custom("Synthetic Agent", "synthetic-agent", &["--acp"]);
        let mut launch = Launch {
            args: Vec::new(),
            start: 0,
        };
        assert!(
            agent
                .prepare(&request(PermissionMode::EveryCommand, false), &mut launch)
                .is_err()
        );
        assert!(
            agent
                .prepare(&request(PermissionMode::FullAccess, false), &mut launch)
                .is_ok()
        );
        assert!(!agent.can_run(PermissionMode::Checked, false, true));
        assert_eq!(agent.default_mode(), Some(PermissionMode::EveryCommand));
        assert!(!agent.modes().contains(&PermissionMode::Checked));
    }

    #[test]
    fn a_run_may_ask_to_continue_an_earlier_session() {
        let agent = AcpAgent::custom("Synthetic Agent", "synthetic-agent", &["--acp"]);
        assert!(agent.resumes());
        let mut asked = request(PermissionMode::FullAccess, true);
        asked.resume = Some("synthetic-session-1");
        let mut launch = Launch {
            args: Vec::new(),
            start: 0,
        };
        let plan = agent.prepare(&asked, &mut launch).unwrap();
        assert_eq!(launch.args, ["--acp"]);
        assert!(plan.host.is_some());
        // Whether it can is the agent's own word, once it runs.
        assert!(!plan.keeps_session);
    }

    /// Delegation follows each maker's terms: Antigravity takes work an
    /// agent hands it on its own (consumer) sign-in only while the owner
    /// watches, and anywhere on a key;
    /// Claude Code and Codex take it on theirs with a warning; an agent the
    /// owner added takes it.
    #[test]
    fn delegation_follows_each_makers_terms_by_sign_in() {
        use crate::{CodingAgent as _, Delegation, SignInMethod};
        let agy = AcpAgent::antigravity();
        // Only in a session the owner watches.
        assert!(matches!(
            agy.delegation(SignInMethod::Own),
            Delegation::WhileWatched(_)
        ));
        assert!(agy.delegation(SignInMethod::Own).permits(true));
        assert!(!agy.delegation(SignInMethod::Own).permits(false));
        assert_eq!(
            agy.delegation(SignInMethod::GeminiApiKey),
            Delegation::Allowed
        );
        assert_eq!(agy.sign_in_note(), agy.delegation(SignInMethod::Own).note());
        assert_eq!(
            AcpAgent::gemini().delegation(SignInMethod::Own),
            Delegation::Allowed
        );
        for agent in [
            &crate::claude::ClaudeCode as &dyn crate::CodingAgent,
            &crate::codex::Codex,
        ] {
            assert!(matches!(
                agent.delegation(SignInMethod::Own),
                Delegation::Warned(_)
            ));
            assert!(agent.sign_in_note().is_some());
        }
        let other = AcpAgent::custom("Synthetic Agent", "synthetic-agent", &["--acp"]);
        assert_eq!(other.delegation(SignInMethod::Own), Delegation::Allowed);
    }

    /// OpenCode takes handed work on its own sign-ins with a word: that work
    /// runs without its plan sign-ins (`opencode::delegated`), the owner's
    /// own runs with all of them.
    #[test]
    fn opencode_is_handed_work_without_its_plan_sign_ins() {
        use crate::{CodingAgent as _, Delegation, SignInMethod};
        let agent = AcpAgent::opencode();
        assert!(agent.launches("opencode"));
        assert!(matches!(
            agent.delegation(SignInMethod::Own),
            Delegation::Warned(_)
        ));
        assert!(agent.delegation(SignInMethod::Own).permits(false));
        assert!(agent.sign_in_note().is_some());
        assert!(agent.sign_ins().is_empty());
        assert_eq!(
            agent.signed_in(&crate::SignIn::Own).unwrap(),
            crate::SignedIn::default()
        );
        let mut launch = Launch {
            args: Vec::new(),
            start: 0,
        };
        agent
            .prepare(&request(PermissionMode::EveryCommand, true), &mut launch)
            .unwrap();
        assert_eq!(launch.args, ["acp"]);
        // Other agents hand nothing different to delegated work.
        let gemini = AcpAgent::gemini();
        assert_eq!(
            gemini.signed_in_delegated(&crate::SignIn::Own).unwrap(),
            crate::SignedIn::default()
        );
    }

    #[test]
    fn agy_and_gemini_default_to_own_and_offer_gemini_key() {
        let agent = AcpAgent::antigravity();
        assert_eq!(agent.sign_ins(), [crate::SignInMethod::GeminiApiKey]);
        assert_eq!(agent.default_sign_in(), crate::SignInMethod::Own);
        assert!(agent.sign_in_note().is_some());
        // Its own sign-in adds nothing.
        assert_eq!(
            agent.signed_in(&crate::SignIn::Own).unwrap(),
            crate::SignedIn::default()
        );
        // Gemini CLI defaults to its own sign-in and also accepts a Gemini API key.
        let gemini = AcpAgent::gemini();
        assert_eq!(gemini.sign_ins(), [crate::SignInMethod::GeminiApiKey]);
        assert_eq!(gemini.default_sign_in(), crate::SignInMethod::Own);
        assert!(gemini.sign_in_note().is_none());
        let gemini_signed = gemini
            .signed_in(&crate::SignIn::GeminiApiKey("synthetic-key".into()))
            .unwrap();
        assert!(gemini_signed.args.is_empty());
        assert_eq!(
            gemini_signed.env,
            vec![("GEMINI_API_KEY".into(), "synthetic-key".into())]
        );
        assert!(
            gemini_signed
                .env_remove
                .iter()
                .any(|v| v == "GOOGLE_API_KEY")
        );
        // Another agent offers none.
        let other = AcpAgent::custom("Synthetic Agent", "synthetic-agent", &["--acp"]);
        assert!(other.sign_ins().is_empty());
        assert_eq!(other.default_sign_in(), crate::SignInMethod::Own);
        assert!(other.sign_in_note().is_none());
        assert!(
            other
                .signed_in(&crate::SignIn::GeminiApiKey("synthetic-key".into()))
                .is_err()
        );
    }

    #[test]
    fn key_settings_name_the_key_and_keep_the_rest() {
        let folder = std::env::temp_dir().join(format!("aidaemon-agy-{}", uuid::Uuid::new_v4()));
        key_settings(&folder).unwrap();
        let read = || -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(folder.join("settings.json")).unwrap())
                .unwrap()
        };
        assert_eq!(read()["auth"]["type"], "gemini-api-key");
        std::fs::write(
            folder.join("settings.json"),
            r#"{"trustedWorkspaces":["/synthetic"],"auth":{"type":"oauth-personal"}}"#,
        )
        .unwrap();
        key_settings(&folder).unwrap();
        assert_eq!(read()["auth"]["type"], "gemini-api-key");
        assert_eq!(read()["trustedWorkspaces"][0], "/synthetic");
        let _ = std::fs::remove_dir_all(&folder);
    }
}
