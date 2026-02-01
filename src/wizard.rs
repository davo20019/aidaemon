use std::path::{Path, PathBuf};

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
}

const PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        kind: "openai_compatible",
        primary: "gpt-4o",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_key: true,
    },
    ProviderPreset {
        name: "Anthropic (Native)",
        base_url: "https://api.anthropic.com/v1", // managed by provider but good for documentation
        kind: "anthropic",
        primary: "claude-3-5-sonnet-20240620",
        fast: "claude-3-haiku-20240307",
        smart: "claude-3-opus-20240229",
        needs_key: true,
    },
    ProviderPreset {
        name: "Anthropic (via OpenRouter)",
        base_url: "https://openrouter.ai/api/v1",
        kind: "openai_compatible",
        primary: "anthropic/claude-3.5-sonnet",
        fast: "anthropic/claude-3-haiku",
        smart: "anthropic/claude-3-opus",
        needs_key: true,
    },
    ProviderPreset {
        name: "Google AI Studio (Native)",
        base_url: "", // not used — GoogleGenAiProvider has its own base URL
        kind: "google_genai",
        primary: "gemini-3-flash-preview",
        fast: "gemini-2.5-flash-lite",
        smart: "gemini-3-pro-preview",
        needs_key: true,
    },
    ProviderPreset {
        name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        kind: "openai_compatible",
        primary: "openai/gpt-4o",
        fast: "openai/gpt-4o-mini",
        smart: "anthropic/claude-3.5-sonnet",
        needs_key: true,
    },
    ProviderPreset {
        name: "Ollama (local)",
        base_url: "http://localhost:11434/v1",
        kind: "openai_compatible",
        primary: "llama3.1",
        fast: "llama3.1",
        smart: "llama3.1",
        needs_key: false,
    },
    ProviderPreset {
        name: "Custom (OpenAI Compatible)",
        base_url: "https://api.example.com/v1",
        kind: "openai_compatible",
        primary: "model-name",
        fast: "model-name",
        smart: "model-name",
        needs_key: true,
    },
];

pub fn run_wizard(config_path: &Path) -> anyhow::Result<()> {
    println!();
    println!("=== aidaemon Setup Wizard ===");
    println!();

    // 1. Provider selection
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
    if preset.name == "Custom" {
        base_url = Input::new()
            .with_prompt("Base URL (OpenAI-compatible endpoint)")
            .default(base_url)
            .interact_text()?;
    }

    // 2. API key
    let api_key = if preset.needs_key {
        Input::new()
            .with_prompt(format!("Paste your API key for {}", preset.name))
            .interact_text()?
    } else {
        "ollama".to_string()
    };

    // 3. Ollama auto-discovery
    if preset.name == "Ollama (local)" {
        println!("Checking for local Ollama models...");
        match discover_ollama_models() {
            Ok(models) if !models.is_empty() => {
                println!("Found {} model(s):", models.len());
                let model_selection = Select::new()
                    .with_prompt("Select primary model")
                    .items(&models)
                    .default(0)
                    .interact()?;
                primary = models[model_selection].clone();
                fast = primary.clone();
                smart = primary.clone();
            }
            Ok(_) => println!("No models found. Using default: {}", primary),
            Err(e) => println!("Could not reach Ollama: {}. Using default.", e),
        }
    }

    // Custom model names
    if preset.name == "Custom" {
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

    // 4. Telegram setup
    println!();
    let bot_token: String = Input::new()
        .with_prompt("Telegram bot token (from @BotFather)")
        .interact_text()?;

    let user_id: u64 = Input::new()
        .with_prompt("Your Telegram user ID (for authorization, numeric)")
        .validate_with(|input: &String| -> Result<(), &str> {
            input.trim().parse::<u64>()
                .map(|_| ())
                .map_err(|_| "Please enter a valid numeric user ID")
        })
        .interact_text()?
        .trim()
        .parse()
        .expect("already validated");

    // 5. Browser tool setup
    println!();
    let browser_enabled = Confirm::new()
        .with_prompt("Enable browser tool? (lets the agent browse the web, fill forms, take screenshots)")
        .default(false)
        .interact()?;

    let mut browser_section = String::new();
    if browser_enabled {
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
                println!("NOTE: Chrome locks its profile while running.");
                println!("The agent will only be able to use the browser when Chrome is closed,");
                println!("or you can copy the profile to a separate directory.");
            }
        } else {
            println!("No Chrome installation detected. The browser will use a fresh profile.");
            println!("You can set user_data_dir in config.toml later to reuse an existing profile.");
        }

        let headless = Confirm::new()
            .with_prompt("Run browser in headless mode? (no visible window)")
            .default(true)
            .interact()?;

        browser_section = format!("\n[browser]\nenabled = true\nheadless = {headless}\n");
        if !user_data_dir.is_empty() {
            browser_section.push_str(&format!("user_data_dir = \"{user_data_dir}\"\n"));
            browser_section.push_str(&format!("profile = \"{profile_name}\"\n"));
        }
    }

    // 6. Write config.toml
    let base_url_line = if base_url.is_empty() {
        String::new()
    } else {
        format!("base_url = \"{base_url}\"\n")
    };
    let config = format!(
        r#"[provider]
kind = "{}"
api_key = "{api_key}"
{base_url_line}
[provider.models]
primary = "{primary}"
fast = "{fast}"
smart = "{smart}"

[telegram]
bot_token = "{bot_token}"
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
"#
    , preset.kind);

    std::fs::write(config_path, config)?;

    println!();
    println!("Setup complete! Config written to {}", config_path.display());
    println!("Run `aidaemon` to start.");
    println!();

    Ok(())
}

struct ChromeProfile {
    user_data_dir: String,
    dir_name: String,
    display_name: String,
}

/// Auto-detect Chrome profiles on the system.
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
        vec![
            dirs::data_local_dir()
                .map(|d| d.join("Google/Chrome/User Data"))
                .unwrap_or_default(),
        ]
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
