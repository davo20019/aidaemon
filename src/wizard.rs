use std::path::Path;

use dialoguer::{Confirm, Input, MultiSelect, Select};

struct ProviderPreset {
    name: &'static str,
    base_url: &'static str,
    // Kind field for new config
    kind: &'static str,
    primary: &'static str,
    fast: &'static str,
    smart: &'static str,
    needs_key: bool,
    key_url: &'static str,
    key_hint: &'static str,
    prompt_base_url: bool,
    supports_gateway_token: bool,
}

const PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        name: "Google AI Studio — recommended",
        base_url: "",
        kind: "google_genai",
        primary: "gemini-3-flash-preview",
        fast: "gemini-2.5-flash-lite",
        smart: "gemini-3-pro-preview",
        needs_key: true,
        key_url: "https://aistudio.google.com/apikey",
        key_hint: "Get a free API key from Google AI Studio (free tier, includes web grounding)",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        kind: "openai_compatible",
        primary: "gpt-4o",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_key: true,
        key_url: "https://platform.openai.com/api-keys",
        key_hint: "Starts with \"sk-\"",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "Anthropic (Native)",
        base_url: "https://api.anthropic.com/v1",
        kind: "anthropic",
        primary: "claude-sonnet-4-20250514",
        fast: "claude-haiku-4-20250414",
        smart: "claude-opus-4-20250414",
        needs_key: true,
        key_url: "https://console.anthropic.com/settings/keys",
        key_hint: "Starts with \"sk-ant-\"",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "Anthropic (via OpenRouter)",
        base_url: "https://openrouter.ai/api/v1",
        kind: "openai_compatible",
        primary: "anthropic/claude-sonnet-4",
        fast: "anthropic/claude-haiku-4",
        smart: "anthropic/claude-opus-4",
        needs_key: true,
        key_url: "https://openrouter.ai/keys",
        key_hint: "Starts with \"sk-or-\"",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        kind: "openai_compatible",
        primary: "openai/gpt-4o",
        fast: "openai/gpt-4o-mini",
        smart: "anthropic/claude-sonnet-4",
        needs_key: true,
        key_url: "https://openrouter.ai/keys",
        key_hint: "Starts with \"sk-or-\". Gives access to many providers in one key.",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "DeepSeek",
        base_url: "https://api.deepseek.com",
        kind: "openai_compatible",
        primary: "deepseek-chat",
        fast: "deepseek-chat",
        smart: "deepseek-reasoner",
        needs_key: true,
        key_url: "https://platform.deepseek.com/api_keys",
        key_hint: "Starts with \"sk-\". Affordable pricing with strong reasoning models.",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "Moonshot AI (Kimi)",
        base_url: "https://api.moonshot.ai/v1",
        kind: "openai_compatible",
        primary: "kimi-k2.5",
        fast: "kimi-k2.5",
        smart: "kimi-k2-thinking",
        needs_key: true,
        key_url: "https://platform.moonshot.ai/",
        key_hint: "Moonshot OpenAI-compatible endpoint. Use your Moonshot API key.",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "MiniMax",
        base_url: "https://api.minimax.io/v1",
        kind: "openai_compatible",
        primary: "MiniMax-M2.5",
        fast: "MiniMax-M2.5-highspeed",
        smart: "MiniMax-M2.5",
        needs_key: true,
        key_url: "https://platform.minimax.io/user-center/basic-information",
        key_hint: "MiniMax OpenAI-compatible endpoint. Use your MiniMax API key.",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "Cloudflare AI Gateway",
        base_url: "https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/<GATEWAY_ID>/compat",
        kind: "openai_compatible",
        primary: "gpt-4o-mini",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_key: true,
        key_url: "https://dash.cloudflare.com/",
        key_hint: "Use your upstream LLM provider API key. You can optionally add a Cloudflare gateway token in the next step.",
        prompt_base_url: true,
        supports_gateway_token: true,
    },
    ProviderPreset {
        name: "Ollama (local)",
        base_url: "http://localhost:11434/v1",
        kind: "openai_compatible",
        primary: "llama3.1",
        fast: "llama3.1",
        smart: "llama3.1",
        needs_key: false,
        key_url: "",
        key_hint: "",
        prompt_base_url: false,
        supports_gateway_token: false,
    },
    ProviderPreset {
        name: "Custom (OpenAI Compatible)",
        base_url: "https://api.example.com/v1",
        kind: "openai_compatible",
        primary: "model-name",
        fast: "model-name",
        smart: "model-name",
        needs_key: true,
        key_url: "",
        key_hint: "",
        prompt_base_url: true,
        supports_gateway_token: false,
    },
];

#[derive(Clone, Copy)]
enum ChannelSetupChoice {
    Telegram,
    #[cfg(feature = "discord")]
    Discord,
    #[cfg(feature = "slack")]
    Slack,
}

/// Returns true if the wizard ran and we should start the daemon immediately.
pub fn run_wizard(config_path: &Path) -> anyhow::Result<bool> {
    println!();
    println!("  Welcome to aidaemon!");
    println!("  --------------------");
    println!("  Your personal AI agent that runs 24/7 in the background.");
    println!("  This wizard will get you up and running in a few steps.");
    println!();

    // ── Step 1: Provider ────────────────────────────────────────────────
    println!("  STEP 1 of 2 — Choose your AI provider");
    println!("  ──────────────────────────────────────");
    println!();

    let provider_names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    let selection = Select::new()
        .with_prompt("Select your LLM provider")
        .items(&provider_names)
        .default(0)
        .interact()?;

    let preset = &PRESETS[selection];
    let mut base_url = preset.base_url.to_string();
    let mut primary = preset.primary.to_string();
    let mut fast = preset.fast.to_string();
    let mut smart = preset.smart.to_string();

    // Optional base URL prompt (custom providers, Cloudflare gateway)
    if preset.prompt_base_url {
        base_url = Input::new()
            .with_prompt("Base URL (OpenAI-compatible endpoint)")
            .default(base_url)
            .interact_text()?;
    }

    // API key with guidance
    let api_key = if preset.needs_key {
        println!();
        if !preset.key_url.is_empty() {
            println!("  Get your API key here: {}", preset.key_url);
        }
        if !preset.key_hint.is_empty() {
            println!("  Hint: {}", preset.key_hint);
        }
        println!();

        let key: String = Input::new()
            .with_prompt(format!("Paste your {} API key", preset.name))
            .interact_text()?;

        // Validate API key
        println!();
        println!("  Checking API key...");
        match validate_api_key(&key, preset.kind, &base_url) {
            Ok(()) => println!("  API key is valid!"),
            Err(e) => {
                println!("  Warning: Could not verify API key ({})", e);
                println!("  The key has been saved — you can fix it later in config.toml.");
            }
        }

        key
    } else {
        "ollama".to_string()
    };

    let mut gateway_token: Option<String> = None;
    if preset.supports_gateway_token {
        println!();
        println!("  Optional: enable Cloudflare Authenticated Gateway token.");
        let use_gateway_token = Confirm::new()
            .with_prompt("Add `cf-aig-authorization` gateway token?")
            .default(false)
            .interact()?;
        if use_gateway_token {
            let token: String = Input::new()
                .with_prompt("Paste Cloudflare gateway token")
                .interact_text()?;
            gateway_token = Some(token);
        }
    }

    // Ollama auto-discovery
    if preset.name == "Ollama (local)" {
        println!();
        println!("  Checking for local Ollama models...");
        match discover_ollama_models() {
            Ok(models) if !models.is_empty() => {
                println!("  Found {} model(s):", models.len());
                let model_selection = Select::new()
                    .with_prompt("Select primary model")
                    .items(&models)
                    .default(0)
                    .interact()?;
                primary = models[model_selection].clone();
                fast = primary.clone();
                smart = primary.clone();
            }
            Ok(_) => println!("  No models found. Using default: {}", primary),
            Err(e) => println!("  Could not reach Ollama: {}. Using default.", e),
        }
    }

    // Custom model names
    if preset.name == "Custom (OpenAI Compatible)" {
        primary = Input::new()
            .with_prompt("Primary model name")
            .default(primary)
            .interact_text()?;
        fast = Input::new()
            .with_prompt("Fast model name")
            .default(fast)
            .interact_text()?;
        smart = Input::new()
            .with_prompt("Smart model name")
            .default(smart)
            .interact_text()?;
    }

    // ── Step 2: Channels ────────────────────────────────────────────────
    println!();
    println!("  STEP 2 of 2 — Connect channels");
    println!("  ───────────────────────────────");
    println!();
    println!("  Select one or more channels to set up now.");
    println!("  You can always add more later with /connect.");
    println!();

    #[cfg(not(feature = "discord"))]
    println!("  TIP: Rebuild with `--features discord` to enable Discord onboarding.");
    #[cfg(not(feature = "slack"))]
    println!("  TIP: Rebuild with `--features slack` to enable Slack onboarding.");

    #[allow(unused_mut)]
    let mut channel_choices: Vec<(String, ChannelSetupChoice)> = vec![(
        "Telegram (recommended baseline)".to_string(),
        ChannelSetupChoice::Telegram,
    )];
    #[cfg(feature = "discord")]
    channel_choices.push((
        "Discord (bot + slash commands)".to_string(),
        ChannelSetupChoice::Discord,
    ));
    #[cfg(feature = "slack")]
    channel_choices.push((
        "Slack (Socket Mode + threads)".to_string(),
        ChannelSetupChoice::Slack,
    ));

    let channel_labels: Vec<&str> = channel_choices
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    let channel_defaults: Vec<bool> = channel_choices
        .iter()
        .map(|(_, choice)| matches!(choice, ChannelSetupChoice::Telegram))
        .collect();

    let selected_indices = loop {
        let selected = MultiSelect::new()
            .with_prompt("Which channels do you want to configure now?")
            .items(&channel_labels)
            .defaults(&channel_defaults)
            .interact()?;
        if selected.is_empty() {
            println!("  Please select at least one channel.");
            continue;
        }
        break selected;
    };

    let mut selected_channel_names: Vec<String> = Vec::new();

    let mut telegram_bot_token: Option<String> = None;

    #[cfg(feature = "discord")]
    let mut discord_bot_token: Option<String> = None;
    #[cfg(feature = "discord")]
    let mut discord_guild_id: Option<u64> = None;

    #[cfg(feature = "slack")]
    let mut slack_app_token: Option<String> = None;
    #[cfg(feature = "slack")]
    let mut slack_bot_token: Option<String> = None;
    #[cfg(feature = "slack")]
    let mut slack_use_threads: Option<bool> = None;

    for idx in selected_indices {
        match channel_choices[idx].1 {
            ChannelSetupChoice::Telegram => {
                let bot_token = prompt_telegram_setup()?;
                telegram_bot_token = Some(bot_token);
                selected_channel_names.push("Telegram".to_string());
            }
            #[cfg(feature = "discord")]
            ChannelSetupChoice::Discord => {
                let (bot_token, guild_id) = prompt_discord_setup()?;
                discord_bot_token = Some(bot_token);
                discord_guild_id = guild_id;
                selected_channel_names.push("Discord".to_string());
            }
            #[cfg(feature = "slack")]
            ChannelSetupChoice::Slack => {
                let (app_token, bot_token, use_threads) = prompt_slack_setup()?;
                slack_app_token = Some(app_token);
                slack_bot_token = Some(bot_token);
                slack_use_threads = Some(use_threads);
                selected_channel_names.push("Slack".to_string());
            }
        }
    }

    // ── Browser setup (optional) ──────────────────────────────────────
    #[allow(unused_mut)]
    let mut browser_enabled = false;
    #[allow(unused_mut)]
    let mut browser_headless = true;

    #[cfg(feature = "browser")]
    {
        println!();
        println!("  Optional: Browser automation");
        println!("  The browser tool lets the agent navigate websites, fill forms,");
        println!("  take screenshots, and interact with authenticated services.");
        println!();

        let enable = Confirm::new()
            .with_prompt("  Enable the browser tool?")
            .default(false)
            .interact()?;

        if enable {
            browser_enabled = true;

            let mode_choices = vec![
                "Visible (recommended for personal computers)",
                "Headless (recommended for servers)",
            ];
            let mode_idx = Select::new()
                .with_prompt("  Browser mode")
                .items(&mode_choices)
                .default(0)
                .interact()?;
            browser_headless = mode_idx == 1;

            println!();
            println!(
                "  Browser tool enabled ({} mode).",
                if browser_headless {
                    "headless"
                } else {
                    "visible"
                }
            );
            println!("  Run `aidaemon browser login` after setup to log into your services.");
        }
    }

    // ── Store secrets in OS keychain if available ───────────────────────
    let mut any_secret_in_keychain = false;

    let (api_key_config, api_stored) = secret_config_value("api_key", &api_key);
    if api_stored {
        any_secret_in_keychain = true;
        println!("  API key stored in OS keychain.");
    }

    let mut gateway_token_config: Option<String> = None;
    if let Some(token) = gateway_token.as_deref() {
        let (value, stored) = secret_config_value("gateway_token", token);
        gateway_token_config = Some(value);
        if stored {
            any_secret_in_keychain = true;
            println!("  Cloudflare gateway token stored in OS keychain.");
        }
    }

    let mut telegram_bot_token_config: Option<String> = None;
    if let Some(token) = telegram_bot_token.as_deref() {
        let (value, stored) = secret_config_value("bot_token", token);
        telegram_bot_token_config = Some(value);
        if stored {
            any_secret_in_keychain = true;
            println!("  Telegram bot token stored in OS keychain.");
        }
    }

    #[cfg(feature = "discord")]
    let mut discord_bot_token_config: Option<String> = None;
    #[cfg(feature = "discord")]
    if let Some(token) = discord_bot_token.as_deref() {
        let (value, stored) = secret_config_value("discord_bot_token", token);
        discord_bot_token_config = Some(value);
        if stored {
            any_secret_in_keychain = true;
            println!("  Discord bot token stored in OS keychain.");
        }
    }

    #[cfg(feature = "slack")]
    let mut slack_app_token_config: Option<String> = None;
    #[cfg(feature = "slack")]
    if let Some(token) = slack_app_token.as_deref() {
        let (value, stored) = secret_config_value("slack_app_token", token);
        slack_app_token_config = Some(value);
        if stored {
            any_secret_in_keychain = true;
            println!("  Slack app token stored in OS keychain.");
        }
    }

    #[cfg(feature = "slack")]
    let mut slack_bot_token_config: Option<String> = None;
    #[cfg(feature = "slack")]
    if let Some(token) = slack_bot_token.as_deref() {
        let (value, stored) = secret_config_value("slack_bot_token", token);
        slack_bot_token_config = Some(value);
        if stored {
            any_secret_in_keychain = true;
            println!("  Slack bot token stored in OS keychain.");
        }
    }

    if any_secret_in_keychain {
        println!();
        println!("  Secrets stored in OS keychain (not in config file).");
        println!("  To move to a different machine, use environment variables instead.");
    }

    let mut channel_sections = String::new();

    if let Some(bot_token) = telegram_bot_token_config {
        channel_sections.push_str("\n[telegram]\n");
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str("allowed_user_ids = []\n");
    }

    #[cfg(feature = "discord")]
    if let Some(bot_token) = discord_bot_token_config {
        channel_sections.push_str("\n[discord]\n");
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str("allowed_user_ids = []\n");
        if let Some(guild_id) = discord_guild_id {
            channel_sections.push_str(&format!("guild_id = {guild_id}\n"));
        }
    }

    #[cfg(feature = "slack")]
    if let (Some(app_token), Some(bot_token), Some(use_threads)) = (
        slack_app_token_config,
        slack_bot_token_config,
        slack_use_threads,
    ) {
        channel_sections.push_str("\n[slack]\n");
        channel_sections.push_str(&format!("app_token = {app_token}\n"));
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str("allowed_user_ids = []\n");
        channel_sections.push_str(&format!("use_threads = {use_threads}\n"));
    }

    // ── Write config ────────────────────────────────────────────────────
    let base_url_line = if base_url.is_empty() {
        String::new()
    } else {
        format!("base_url = \"{base_url}\"\n")
    };
    let gateway_token_line = gateway_token_config
        .as_ref()
        .map(|v| format!("gateway_token = {v}\n"))
        .unwrap_or_default();
    let browser_section = if browser_enabled {
        format!("\n[browser]\nenabled = true\nheadless = {browser_headless}\n",)
    } else {
        "\n# Uncomment to enable browser tool:\n# [browser]\n# enabled = true\n".to_string()
    };

    let config = format!(
        r#"[provider]
kind = "{}"
api_key = {api_key_config}
{gateway_token_line}
{base_url_line}
[provider.models]
primary = "{primary}"
fast = "{fast}"
smart = "{smart}"{channel_sections}

[state]
db_path = "aidaemon.db"
working_memory_cap = 50

[terminal]
allowed_prefixes = ["ls", "cat", "echo", "date", "whoami", "uname", "df", "du", "ps"]

[daemon]
health_port = 8080
{browser_section}
# Uncomment to enable email triggers:
# [triggers.email]
# host = "imap.gmail.com"
# port = 993
# username = "you@gmail.com"
# password = "app-password"
# folder = "INBOX"

# Uncomment to enable MCP servers:
# [mcp.filesystem]
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
"#,
        preset.kind,
        channel_sections = channel_sections
    );

    std::fs::write(config_path, &config)?;

    println!();
    println!(
        "  Setup complete! Config saved to {}",
        config_path.display()
    );
    println!();
    println!(
        "  You can edit {} anytime to change settings.",
        config_path.display()
    );
    println!("  Run `aidaemon install-service` to start on boot (systemd/launchd).");
    if browser_enabled {
        println!("  Run `aidaemon browser login` to log into services for the agent.");
    }
    println!();
    println!(
        "  Starting... Send a message on {} to get started!",
        human_list(&selected_channel_names)
    );
    println!();

    Ok(true)
}

fn prompt_telegram_setup() -> anyhow::Result<String> {
    println!();
    println!("  Telegram setup");
    println!("  1. Open Telegram and search for @BotFather");
    println!("  2. Send /newbot and follow the prompts");
    println!("  3. Copy the bot token (looks like 123456:ABC-DEF...)");
    println!();

    let bot_token: String = Input::new()
        .with_prompt("Telegram bot token")
        .interact_text()?;

    println!();
    println!("  Checking Telegram token...");
    match validate_bot_token(&bot_token) {
        Ok(bot_name) => {
            println!("  Connected to Telegram bot @{}", bot_name);
            println!();
            println!("  Send a DM to @{} to get started!", bot_name);
        }
        Err(e) => {
            println!("  Warning: Could not verify Telegram token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    Ok(bot_token)
}

#[cfg(feature = "discord")]
fn prompt_discord_setup() -> anyhow::Result<(String, Option<u64>)> {
    println!();
    println!("  Discord setup");
    println!("  1. Create an app at https://discord.com/developers/applications");
    println!("  2. Add a bot user and copy its token");
    println!("  3. Invite the bot to your server (or use DMs)");
    println!();

    let bot_token: String = Input::new()
        .with_prompt("Discord bot token")
        .interact_text()?;

    println!();
    println!("  Checking Discord token...");
    match validate_discord_bot_token(&bot_token) {
        Ok(bot_name) => {
            println!("  Connected to Discord bot {}", bot_name);
            println!();
            println!("  Send a DM to {} to get started!", bot_name);
        }
        Err(e) => {
            println!("  Warning: Could not verify Discord token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    let guild_input: String = Input::new()
        .with_prompt("Optional Discord server (guild) ID (leave blank for any)")
        .allow_empty(true)
        .validate_with(|input: &String| -> Result<(), &str> {
            if input.trim().is_empty() || input.trim().parse::<u64>().is_ok() {
                Ok(())
            } else {
                Err("Please enter a numeric guild ID or leave blank")
            }
        })
        .interact_text()?;
    let guild_id = if guild_input.trim().is_empty() {
        None
    } else {
        Some(guild_input.trim().parse().expect("already validated"))
    };

    Ok((bot_token, guild_id))
}

#[cfg(feature = "slack")]
fn prompt_slack_setup() -> anyhow::Result<(String, String, bool)> {
    println!();
    println!("  Slack setup");
    println!("  1. Create a Slack app at https://api.slack.com/apps");
    println!("  2. Enable Socket Mode and copy the app token (xapp-...)");
    println!("  3. Install the app and copy the bot token (xoxb-...)");
    println!();

    let bot_token: String = Input::new()
        .with_prompt("Slack bot token (xoxb-...)")
        .interact_text()?;
    let app_token: String = Input::new()
        .with_prompt("Slack app token (xapp-...)")
        .interact_text()?;

    println!();
    println!("  Checking Slack bot token...");
    match validate_slack_bot_token(&bot_token) {
        Ok((bot_name, team_name)) => {
            println!("  Connected to Slack bot {} in {}", bot_name, team_name);
            println!();
            println!(
                "  Send a DM to @{} in {} to get started!",
                bot_name, team_name
            );
        }
        Err(e) => {
            println!("  Warning: Could not verify Slack bot token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    println!("  Checking Slack app token...");
    match validate_slack_app_token(&app_token) {
        Ok(()) => println!("  Slack app token is valid."),
        Err(e) => {
            println!("  Warning: Could not verify Slack app token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    let use_threads: bool = {
        use dialoguer::Confirm;
        Confirm::new()
            .with_prompt("Reply in Slack threads?")
            .default(true)
            .interact()?
    };

    Ok((app_token, bot_token, use_threads))
}

fn quote_toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn secret_config_value(keychain_key: &str, secret: &str) -> (String, bool) {
    match crate::config::store_in_keychain(keychain_key, secret) {
        Ok(()) => ("\"keychain\"".to_string(), true),
        Err(e) => {
            tracing::debug!("Keychain unavailable for {}: {}", keychain_key, e);
            (quote_toml_string(secret), false)
        }
    }
}

fn human_list(items: &[String]) -> String {
    match items {
        [] => "your configured channel".to_string(),
        [one] => one.clone(),
        [first, second] => format!("{first} and {second}"),
        _ => {
            let head = &items[..items.len() - 1];
            let last = &items[items.len() - 1];
            format!("{}, and {}", head.join(", "), last)
        }
    }
}

fn is_cloudflare_ai_gateway_base(base_url: &str) -> bool {
    let parsed = match reqwest::Url::parse(base_url) {
        Ok(url) => url,
        Err(_) => return false,
    };
    matches!(
        parsed.host_str(),
        Some(host) if host.eq_ignore_ascii_case("gateway.ai.cloudflare.com")
    )
}

fn cloudflare_models_probe_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/compat") {
        format!("{trimmed}/v1/models")
    } else {
        format!("{trimmed}/compat/v1/models")
    }
}

/// Validate an API key by making a lightweight request to the provider.
fn validate_api_key(key: &str, kind: &str, base_url: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    match kind {
        "anthropic" => {
            // Anthropic: hit /v1/messages with an empty body to see if auth works.
            // A 400 (bad request) means the key is valid; 401 means invalid.
            let resp = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .body(r#"{"model":"claude-haiku-4-20250414","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}"#)
                .send()?;
            let status = resp.status().as_u16();
            if status == 401 || status == 403 {
                anyhow::bail!("invalid API key");
            }
            // 200, 400, 429 all mean the key itself is recognized
            Ok(())
        }
        "google_genai" => {
            // Google AI: list models endpoint
            // Use header-based auth to avoid API key in URL logs
            let url = "https://generativelanguage.googleapis.com/v1beta/models";
            let resp = client.get(url).header("x-goog-api-key", key).send()?;
            let status = resp.status().as_u16();
            if status == 400 || status == 401 || status == 403 {
                anyhow::bail!("invalid API key");
            }
            Ok(())
        }
        _ => {
            // OpenAI-compatible: hit /models
            let url = format!("{}/models", base_url.trim_end_matches('/'));
            let status = client
                .get(&url)
                .header("Authorization", format!("Bearer {}", key))
                .send()?
                .status()
                .as_u16();
            if status == 401 || status == 403 {
                anyhow::bail!("invalid API key");
            }

            if is_cloudflare_ai_gateway_base(base_url) && (status == 404 || status == 405) {
                let cf_url = cloudflare_models_probe_url(base_url);
                let cf_status = client
                    .get(&cf_url)
                    .header("Authorization", format!("Bearer {}", key))
                    .send()?
                    .status()
                    .as_u16();
                if cf_status == 401 || cf_status == 403 {
                    anyhow::bail!("invalid API key");
                }
                if cf_status == 404 || cf_status == 405 {
                    anyhow::bail!(
                        "could not verify API key: models endpoint unavailable at '{}' and '{}'",
                        url,
                        cf_url
                    );
                }
            }

            Ok(())
        }
    }
}

/// Validate a Telegram bot token by calling getMe. Returns the bot username.
fn validate_bot_token(token: &str) -> anyhow::Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let url = format!("https://api.telegram.org/bot{}/getMe", token);
    let resp = client.get(&url).send()?;
    let data: serde_json::Value = resp.json()?;

    if data["ok"].as_bool() != Some(true) {
        anyhow::bail!(
            "{}",
            data["description"].as_str().unwrap_or("invalid token")
        );
    }

    let username = data["result"]["username"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    Ok(username)
}

#[cfg(feature = "discord")]
fn validate_discord_bot_token(token: &str) -> anyhow::Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client
        .get("https://discord.com/api/v10/users/@me")
        .header("Authorization", format!("Bot {}", token))
        .send()?;
    let status = resp.status();
    let data: serde_json::Value = resp.json().unwrap_or_default();

    if !status.is_success() {
        let message = data["message"].as_str().unwrap_or("invalid token");
        anyhow::bail!("{} (HTTP {})", message, status.as_u16());
    }

    let username = data["username"].as_str().unwrap_or("unknown").to_string();
    Ok(username)
}

#[cfg(feature = "slack")]
fn validate_slack_bot_token(token: &str) -> anyhow::Result<(String, String)> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client
        .get("https://slack.com/api/auth.test")
        .header("Authorization", format!("Bearer {}", token))
        .send()?;
    let data: serde_json::Value = resp.json()?;

    if data["ok"].as_bool() != Some(true) {
        anyhow::bail!(
            "{}",
            data["error"].as_str().unwrap_or("invalid Slack bot token")
        );
    }

    let user = data["user"].as_str().unwrap_or("unknown").to_string();
    let team = data["team"].as_str().unwrap_or("unknown").to_string();
    Ok((user, team))
}

#[cfg(feature = "slack")]
fn validate_slack_app_token(token: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client
        .post("https://slack.com/api/apps.connections.open")
        .header("Authorization", format!("Bearer {}", token))
        .send()?;
    let data: serde_json::Value = resp.json()?;

    if data["ok"].as_bool() != Some(true) {
        anyhow::bail!(
            "{}",
            data["error"].as_str().unwrap_or("invalid Slack app token")
        );
    }
    Ok(())
}

fn discover_ollama_models() -> anyhow::Result<Vec<String>> {
    // Blocking HTTP request (wizard runs before async runtime is fully up)
    let resp = reqwest::blocking::get("http://localhost:11434/api/tags")?;
    let data: serde_json::Value = resp.json()?;
    let models: Vec<String> = data["models"]
        .as_array()
        .map(|arr: &Vec<serde_json::Value>| {
            arr.iter()
                .filter_map(|m: &serde_json::Value| m["name"].as_str().map(|s: &str| s.to_string()))
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::{cloudflare_models_probe_url, is_cloudflare_ai_gateway_base};

    #[test]
    fn cloudflare_gateway_host_detection() {
        assert!(is_cloudflare_ai_gateway_base(
            "https://gateway.ai.cloudflare.com/v1/account/gateway/compat"
        ));
        assert!(!is_cloudflare_ai_gateway_base("https://api.openai.com/v1"));
    }

    #[test]
    fn cloudflare_models_probe_url_with_compat_suffix() {
        assert_eq!(
            cloudflare_models_probe_url(
                "https://gateway.ai.cloudflare.com/v1/account/gateway/compat"
            ),
            "https://gateway.ai.cloudflare.com/v1/account/gateway/compat/v1/models"
        );
    }

    #[test]
    fn cloudflare_models_probe_url_without_compat_suffix() {
        assert_eq!(
            cloudflare_models_probe_url("https://gateway.ai.cloudflare.com/v1/account/gateway"),
            "https://gateway.ai.cloudflare.com/v1/account/gateway/compat/v1/models"
        );
    }
}
