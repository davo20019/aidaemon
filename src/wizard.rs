use std::path::Path;
#[cfg(feature = "browser")]
use std::path::PathBuf;

#[cfg(feature = "browser")]
use dialoguer::{Confirm, Input, Select};
#[cfg(not(feature = "browser"))]
use dialoguer::{Confirm, Input, Select};

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

    // ── Step 2: Telegram ────────────────────────────────────────────────
    println!();
    println!("  STEP 2 of 3 — Connect Telegram");
    println!("  ───────────────────────────────");
    println!();
    println!("  You'll control your AI agent through a Telegram bot.");
    println!("  Setting this up takes about 60 seconds:");
    println!();
    println!("  1. Open Telegram and search for @BotFather");
    println!("  2. Send /newbot and follow the prompts to name your bot");
    println!("  3. BotFather will give you a token like 123456:ABC-DEF...");
    println!();

    let bot_token: String = Input::new()
        .with_prompt("Paste your bot token from BotFather")
        .interact_text()?;

    // Validate bot token
    println!();
    println!("  Checking bot token...");
    match validate_bot_token(&bot_token) {
        Ok(bot_name) => {
            println!("  Connected to bot @{}", bot_name);
        }
        Err(e) => {
            println!("  Warning: Could not verify bot token ({})", e);
            println!("  The token has been saved — you can fix it later in config.toml.");
        }
    }

    println!();
    println!("  Now we need your Telegram user ID (a number) so only YOU can");
    println!("  control the bot. Here's how to find it:");
    println!();
    println!("  Option A: Search for @userinfobot on Telegram, send /start");
    println!("  Option B: Search for @RawDataBot on Telegram, send /start");
    println!("  Either will reply with your numeric user ID.");
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
    let (api_key_config, bot_token_config) = {
        let mut api_val = format!("\"{}\"", api_key);
        let mut bot_val = format!("\"{}\"", bot_token);

        match crate::config::store_in_keychain("api_key", &api_key) {
            Ok(()) => {
                api_val = "\"keychain\"".to_string();
                println!("  API key stored in OS keychain.");
            }
            Err(e) => {
                tracing::debug!("Keychain unavailable for api_key: {}", e);
            }
        }
        match crate::config::store_in_keychain("bot_token", &bot_token) {
            Ok(()) => {
                bot_val = "\"keychain\"".to_string();
                println!("  Bot token stored in OS keychain.");
            }
            Err(e) => {
                tracing::debug!("Keychain unavailable for bot_token: {}", e);
            }
        }

        if api_val == "\"keychain\"" || bot_val == "\"keychain\"" {
            println!();
            println!("  Secrets stored in OS keychain (not in config file).");
            println!("  To move to a different machine, use environment variables instead.");
        }

        (api_val, bot_val)
    };

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
smart = "{smart}"

[telegram]
bot_token = {bot_token_config}
allowed_user_ids = [{user_id}]

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
        preset.kind
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
        println!("  Starting... Send a message to your bot on Telegram!");
        println!();
    } else {
        println!();
        println!("  Run `aidaemon` whenever you're ready to start.");
        println!();
    }

    Ok(start_now)
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
            let resp = client
                .get(url)
                .header("x-goog-api-key", key)
                .send()?;
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
