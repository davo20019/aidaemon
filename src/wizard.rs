use std::path::Path;
#[cfg(feature = "browser")]
use std::path::PathBuf;

#[cfg(feature = "browser")]
use dialoguer::{Confirm, Input, MultiSelect, Select};
#[cfg(not(feature = "browser"))]
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
    println!("  STEP 1 of 3 — Choose your AI provider");
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

    // Custom base URL
    if preset.name == "Custom (OpenAI Compatible)" {
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
    println!("  STEP 2 of 3 — Connect channels");
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
    let mut telegram_user_id: Option<u64> = None;

    #[cfg(feature = "discord")]
    let mut discord_bot_token: Option<String> = None;
    #[cfg(feature = "discord")]
    let mut discord_user_id: Option<u64> = None;
    #[cfg(feature = "discord")]
    let mut discord_guild_id: Option<u64> = None;

    #[cfg(feature = "slack")]
    let mut slack_app_token: Option<String> = None;
    #[cfg(feature = "slack")]
    let mut slack_bot_token: Option<String> = None;
    #[cfg(feature = "slack")]
    let mut slack_user_id: Option<String> = None;
    #[cfg(feature = "slack")]
    let mut slack_use_threads: Option<bool> = None;

    for idx in selected_indices {
        match channel_choices[idx].1 {
            ChannelSetupChoice::Telegram => {
                let (bot_token, user_id) = prompt_telegram_setup()?;
                telegram_bot_token = Some(bot_token);
                telegram_user_id = Some(user_id);
                selected_channel_names.push("Telegram".to_string());
            }
            #[cfg(feature = "discord")]
            ChannelSetupChoice::Discord => {
                let (bot_token, user_id, guild_id) = prompt_discord_setup()?;
                discord_bot_token = Some(bot_token);
                discord_user_id = Some(user_id);
                discord_guild_id = guild_id;
                selected_channel_names.push("Discord".to_string());
            }
            #[cfg(feature = "slack")]
            ChannelSetupChoice::Slack => {
                let (app_token, bot_token, user_id, use_threads) = prompt_slack_setup()?;
                slack_app_token = Some(app_token);
                slack_bot_token = Some(bot_token);
                slack_user_id = Some(user_id);
                slack_use_threads = Some(use_threads);
                selected_channel_names.push("Slack".to_string());
            }
        }
    }

    // ── Step 3: Browser (optional) ──────────────────────────────────────
    println!();
    println!("  STEP 3 of 3 — Optional features");
    println!("  ────────────────────────────────");

    let browser_section;
    #[cfg(feature = "browser")]
    {
        println!();
        let browser_enabled = Confirm::new()
            .with_prompt("Enable browser tool? (lets the agent browse the web, fill forms, take screenshots)")
            .default(false)
            .interact()?;

        browser_section = if browser_enabled {
            let mut user_data_dir = String::new();
            let mut profile_name = String::new();

            // Try to detect Chrome profiles
            let profiles = discover_chrome_profiles();
            if !profiles.is_empty() {
                let use_profile = Confirm::new()
                    .with_prompt("Chrome detected! Use an existing Chrome profile? (inherits your login sessions)")
                    .default(true)
                    .interact()?;

                if use_profile {
                    let display_names: Vec<String> = profiles
                        .iter()
                        .map(|p| {
                            if p.display_name.is_empty() {
                                p.dir_name.clone()
                            } else {
                                format!("{} ({})", p.display_name, p.dir_name)
                            }
                        })
                        .collect();

                    let profile_selection = Select::new()
                        .with_prompt("Select Chrome profile")
                        .items(&display_names)
                        .default(0)
                        .interact()?;

                    let selected = &profiles[profile_selection];
                    user_data_dir = selected.user_data_dir.clone();
                    profile_name = selected.dir_name.clone();

                    println!();
                    println!("  NOTE: Chrome locks its profile while running.");
                    println!(
                        "  The agent will only be able to use the browser when Chrome is closed,"
                    );
                    println!("  or you can copy the profile to a separate directory.");
                }
            } else {
                println!(
                    "  No Chrome installation detected. The browser will use a fresh profile."
                );
                println!("  You can set user_data_dir in config.toml later to reuse an existing profile.");
            }

            let headless = Confirm::new()
                .with_prompt("Run browser in headless mode? (no visible window)")
                .default(true)
                .interact()?;

            let mut section = format!("\n[browser]\nenabled = true\nheadless = {headless}\n");
            if !user_data_dir.is_empty() {
                section.push_str(&format!("user_data_dir = \"{user_data_dir}\"\n"));
                section.push_str(&format!("profile = \"{profile_name}\"\n"));
            }
            section
        } else {
            String::new()
        };
    }
    #[cfg(not(feature = "browser"))]
    {
        println!();
        println!("  TIP: Browser tool available with `cargo install aidaemon --features browser`");
        browser_section = String::new();
    }

    // ── Store secrets in OS keychain if available ───────────────────────
    let mut any_secret_in_keychain = false;

    let (api_key_config, api_stored) = secret_config_value("api_key", &api_key);
    if api_stored {
        any_secret_in_keychain = true;
        println!("  API key stored in OS keychain.");
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
    let mut owner_ids: Vec<(String, String)> = Vec::new();

    if let (Some(bot_token), Some(user_id)) = (telegram_bot_token_config, telegram_user_id) {
        channel_sections.push_str("\n[telegram]\n");
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str(&format!("allowed_user_ids = [{user_id}]\n"));
        owner_ids.push(("telegram".to_string(), user_id.to_string()));
    }

    #[cfg(feature = "discord")]
    if let (Some(bot_token), Some(user_id)) = (discord_bot_token_config, discord_user_id) {
        channel_sections.push_str("\n[discord]\n");
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str(&format!("allowed_user_ids = [{user_id}]\n"));
        if let Some(guild_id) = discord_guild_id {
            channel_sections.push_str(&format!("guild_id = {guild_id}\n"));
        }
        owner_ids.push(("discord".to_string(), user_id.to_string()));
    }

    #[cfg(feature = "slack")]
    if let (Some(app_token), Some(bot_token), Some(user_id), Some(use_threads)) = (
        slack_app_token_config,
        slack_bot_token_config,
        slack_user_id,
        slack_use_threads,
    ) {
        channel_sections.push_str("\n[slack]\n");
        channel_sections.push_str(&format!("app_token = {app_token}\n"));
        channel_sections.push_str(&format!("bot_token = {bot_token}\n"));
        channel_sections.push_str(&format!(
            "allowed_user_ids = [{}]\n",
            quote_toml_string(&user_id)
        ));
        channel_sections.push_str(&format!("use_threads = {use_threads}\n"));
        owner_ids.push(("slack".to_string(), user_id));
    }

    if !owner_ids.is_empty() {
        channel_sections.push_str("\n[users.owner_ids]\n");
        for (platform, user_id) in owner_ids {
            channel_sections.push_str(&format!("{platform} = [{}]\n", quote_toml_string(&user_id)));
        }
    }

    // ── Write config ────────────────────────────────────────────────────
    let base_url_line = if base_url.is_empty() {
        String::new()
    } else {
        format!("base_url = \"{base_url}\"\n")
    };
    let config = format!(
        r#"[provider]
kind = "{}"
api_key = {api_key_config}
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
    println!();

    // Ask whether to start now
    let start_now = Confirm::new()
        .with_prompt("Start aidaemon now?")
        .default(true)
        .interact()?;

    if start_now {
        println!();
        println!(
            "  Starting... Send a message on {} to begin!",
            human_list(&selected_channel_names)
        );
        println!();
    } else {
        println!();
        println!("  Run `aidaemon` whenever you're ready to start.");
        println!();
    }

    Ok(start_now)
}

fn prompt_telegram_setup() -> anyhow::Result<(String, u64)> {
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
        }
        Err(e) => {
            println!("  Warning: Could not verify Telegram token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    println!();
    println!("  Find your Telegram user ID via @userinfobot or @RawDataBot.");
    println!();

    let user_id: u64 = Input::new()
        .with_prompt("Your Telegram user ID (numeric)")
        .validate_with(|input: &String| -> Result<(), &str> {
            input
                .trim()
                .parse::<u64>()
                .map(|_| ())
                .map_err(|_| "Please enter a valid numeric user ID")
        })
        .interact_text()?
        .trim()
        .parse()
        .expect("already validated");

    Ok((bot_token, user_id))
}

#[cfg(feature = "discord")]
fn prompt_discord_setup() -> anyhow::Result<(String, u64, Option<u64>)> {
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
        Ok(bot_name) => println!("  Connected to Discord bot {}", bot_name),
        Err(e) => {
            println!("  Warning: Could not verify Discord token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    println!();
    println!("  Turn on Developer Mode in Discord to copy your numeric user ID.");
    println!();
    let user_id: u64 = Input::new()
        .with_prompt("Your Discord user ID (numeric)")
        .validate_with(|input: &String| -> Result<(), &str> {
            input
                .trim()
                .parse::<u64>()
                .map(|_| ())
                .map_err(|_| "Please enter a valid numeric user ID")
        })
        .interact_text()?
        .trim()
        .parse()
        .expect("already validated");

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

    Ok((bot_token, user_id, guild_id))
}

#[cfg(feature = "slack")]
fn prompt_slack_setup() -> anyhow::Result<(String, String, String, bool)> {
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
            println!("  Connected to Slack bot {} in {}", bot_name, team_name)
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

    println!();
    println!("  Find your Slack user ID in your Slack profile (or use /whoami app tools).");
    println!();
    let user_id: String = Input::new()
        .with_prompt("Your Slack user ID (e.g., U12345678)")
        .validate_with(|input: &String| -> Result<(), &str> {
            if input.trim().is_empty() {
                Err("Please enter a Slack user ID")
            } else {
                Ok(())
            }
        })
        .interact_text()?;
    let use_threads = Confirm::new()
        .with_prompt("Reply in Slack threads?")
        .default(true)
        .interact()?;

    Ok((
        app_token,
        bot_token,
        user_id.trim().to_string(),
        use_threads,
    ))
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
            let resp = client
                .get(&url)
                .header("Authorization", format!("Bearer {}", key))
                .send()?;
            let status = resp.status().as_u16();
            if status == 401 || status == 403 {
                anyhow::bail!("invalid API key");
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

#[cfg(feature = "browser")]
struct ChromeProfile {
    user_data_dir: String,
    dir_name: String,
    display_name: String,
}

/// Auto-detect Chrome profiles on the system.
#[cfg(feature = "browser")]
fn discover_chrome_profiles() -> Vec<ChromeProfile> {
    let mut profiles = Vec::new();

    // Known Chrome user data directories per platform
    let candidates: Vec<PathBuf> = if cfg!(target_os = "macos") {
        vec![
            dirs::home_dir()
                .map(|h| h.join("Library/Application Support/Google/Chrome"))
                .unwrap_or_default(),
            dirs::home_dir()
                .map(|h| h.join("Library/Application Support/Chromium"))
                .unwrap_or_default(),
        ]
    } else if cfg!(target_os = "linux") {
        vec![
            dirs::config_dir()
                .map(|c| c.join("google-chrome"))
                .unwrap_or_default(),
            dirs::config_dir()
                .map(|c| c.join("chromium"))
                .unwrap_or_default(),
        ]
    } else {
        // Windows or other
        vec![dirs::data_local_dir()
            .map(|d| d.join("Google/Chrome/User Data"))
            .unwrap_or_default()]
    };

    for user_data_dir in candidates {
        if !user_data_dir.exists() {
            continue;
        }

        // Look for profile directories (Default, Profile 1, Profile 2, ...)
        let entries = match std::fs::read_dir(&user_data_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if dir_name == "Default" || dir_name.starts_with("Profile ") {
                let prefs_path = entry.path().join("Preferences");
                let display_name = read_chrome_profile_name(&prefs_path);

                profiles.push(ChromeProfile {
                    user_data_dir: user_data_dir.to_string_lossy().to_string(),
                    dir_name,
                    display_name,
                });
            }
        }
    }

    profiles
}

/// Read the human-readable profile name from Chrome's Preferences JSON.
#[cfg(feature = "browser")]
fn read_chrome_profile_name(prefs_path: &Path) -> String {
    let content = match std::fs::read_to_string(prefs_path) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    json.get("profile")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string()
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
