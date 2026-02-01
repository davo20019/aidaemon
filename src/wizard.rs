use std::path::Path;

use dialoguer::{Input, Select};

struct ProviderPreset {
    name: &'static str,
    base_url: &'static str,
    primary: &'static str,
    fast: &'static str,
    smart: &'static str,
    needs_key: bool,
}

const PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        primary: "gpt-4o",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_key: true,
    },
    ProviderPreset {
        name: "Anthropic (via OpenRouter)",
        base_url: "https://openrouter.ai/api/v1",
        primary: "anthropic/claude-sonnet-4",
        fast: "anthropic/claude-haiku-3.5",
        smart: "anthropic/claude-sonnet-4",
        needs_key: true,
    },
    ProviderPreset {
        name: "Google AI Studio",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        primary: "gemini-2.0-flash",
        fast: "gemini-2.0-flash-lite",
        smart: "gemini-2.5-pro-preview-06-05",
        needs_key: true,
    },
    ProviderPreset {
        name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        primary: "openai/gpt-4o",
        fast: "openai/gpt-4o-mini",
        smart: "anthropic/claude-sonnet-4",
        needs_key: true,
    },
    ProviderPreset {
        name: "Ollama (local)",
        base_url: "http://localhost:11434/v1",
        primary: "llama3.1",
        fast: "llama3.1",
        smart: "llama3.1",
        needs_key: false,
    },
    ProviderPreset {
        name: "Custom",
        base_url: "https://api.example.com/v1",
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

    // 5. Write config.toml
    let config = format!(
        r#"[provider]
api_key = "{api_key}"
base_url = "{base_url}"

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
    );

    std::fs::write(config_path, config)?;

    println!();
    println!("Setup complete! Config written to {}", config_path.display());
    println!("Run `aidaemon` to start.");
    println!();

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
