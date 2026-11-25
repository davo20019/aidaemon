use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use base64::Engine;
use dialoguer::{Confirm, Input, MultiSelect, Select};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TelegramBotTarget {
    Legacy,
    ArrayIndex(usize),
    /// A dynamic bot (added via /connect, stored in DB). The i64 is the DB row id.
    Dynamic(i64),
}

#[derive(Debug, Clone, Default)]
struct ExistingWebhookSettings {
    enabled: bool,
    public_url: Option<String>,
    listen_addr: Option<String>,
    path: Option<String>,
}

#[derive(Debug, Clone)]
struct TelegramBotEntry {
    target: TelegramBotTarget,
    display_name: String,
    slug: String,
    existing: ExistingWebhookSettings,
}

#[derive(Debug, Clone)]
struct TelegramWebhookPlan {
    target: TelegramBotTarget,
    display_name: String,
    public_url: String,
    listen_addr: String,
    path: String,
    max_connections: u8,
    drop_pending_updates: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SetupCommandSpec {
    pub program: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CloudflaredIngressRoute {
    pub hostname: String,
    pub service: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LowLatencyEnvironment {
    WranglerAndCloudflared,
    WranglerOnly,
    CloudflaredOnly,
    Neither,
}

#[derive(Debug, Clone, Copy, Default)]
struct RuntimeProbe {
    wrangler_installed: bool,
    wrangler_authenticated: bool,
    cloudflared_installed: bool,
    cloudflared_authenticated: bool,
    cloudflared_tunnel_exists: bool,
    cloudflared_local_tunnel_running: Option<bool>,
}

impl RuntimeProbe {
    fn environment(self) -> LowLatencyEnvironment {
        match (self.wrangler_authenticated, self.cloudflared_authenticated) {
            (true, true) => LowLatencyEnvironment::WranglerAndCloudflared,
            (true, false) => LowLatencyEnvironment::WranglerOnly,
            (false, true) => LowLatencyEnvironment::CloudflaredOnly,
            (false, false) => LowLatencyEnvironment::Neither,
        }
    }
}

pub fn run_low_latency_setup(config_path: &Path) -> anyhow::Result<()> {
    let config_raw = std::fs::read_to_string(config_path)?;
    let mut doc: toml::Table = config_raw.parse()?;
    let all_bots = collect_telegram_bot_entries(&doc);
    if all_bots.is_empty() {
        anyhow::bail!(
            "No Telegram bots found in config. Configure [telegram] or [[telegram_bots]] first."
        );
    }

    println!();
    println!("  Low-Latency Telegram Setup (Opt-in)");
    println!("  ------------------------------------");
    println!("  Default behavior remains Telegram long polling.");
    println!();

    let probe = probe_runtime_environment();
    print_runtime_probe_summary(probe);

    let bot_labels: Vec<&str> = all_bots.iter().map(|b| b.display_name.as_str()).collect();
    let defaults = vec![true; bot_labels.len()];
    let selected_indices = loop {
        let selected = MultiSelect::new()
            .with_prompt("Select Telegram bot(s) to enable webhook low-latency mode")
            .items(&bot_labels)
            .defaults(&defaults)
            .interact()?;
        if selected.is_empty() {
            println!("  Please select at least one bot.");
            continue;
        }
        break selected;
    };
    let selected_bots: Vec<TelegramBotEntry> = selected_indices
        .into_iter()
        .map(|idx| all_bots[idx].clone())
        .collect();

    let hostname_mode_items = [
        "Use one base domain and auto-generate per-bot hostnames (recommended)",
        "Enter exact hostname per bot",
    ];
    let hostname_mode = Select::new()
        .with_prompt("How do you want to assign public HTTPS hostnames?")
        .items(hostname_mode_items)
        .default(0)
        .interact()?;

    let mut hostnames: HashMap<TelegramBotTarget, String> = HashMap::new();
    if hostname_mode == 0 {
        let base_domain: String = Input::new()
            .with_prompt("Base domain (example: bots.example.com)")
            .validate_with(|input: &String| -> Result<(), String> {
                normalize_domain_input(input)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })
            .interact_text()?;
        let normalized_domain = normalize_domain_input(&base_domain)?;
        let mut used_labels = HashSet::new();
        for bot in &selected_bots {
            let label = unique_dns_label(&bot.slug, &mut used_labels);
            hostnames.insert(
                bot.target,
                format!("{}.{}", label, normalized_domain.to_ascii_lowercase()),
            );
        }
    } else {
        for bot in &selected_bots {
            let default_host = bot
                .existing
                .public_url
                .as_deref()
                .and_then(extract_hostname_from_url)
                .unwrap_or_else(|| format!("{}.example.com", bot.slug));
            let input: String = Input::new()
                .with_prompt(format!("Hostname for {}", bot.display_name))
                .default(default_host)
                .validate_with(|value: &String| -> Result<(), String> {
                    normalize_hostname_input(value)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                })
                .interact_text()?;
            hostnames.insert(bot.target, normalize_hostname_input(&input)?);
        }
    }

    let bind_local_only = Confirm::new()
        .with_prompt("Bind local webhook listener to 127.0.0.1 only? (recommended)")
        .default(true)
        .interact()?;
    let listen_host = if bind_local_only {
        "127.0.0.1"
    } else {
        "0.0.0.0"
    };

    let max_connections_raw: String = Input::new()
        .with_prompt("Telegram max_connections (1-100)")
        .default("40".to_string())
        .validate_with(|value: &String| -> Result<(), String> {
            parse_max_connections(value)
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
        .interact_text()?;
    let max_connections = parse_max_connections(&max_connections_raw)?;

    let drop_pending_updates = Confirm::new()
        .with_prompt("Drop pending Telegram updates when enabling webhook?")
        .default(false)
        .interact()?;

    let keep_polling_fallback = Confirm::new()
        .with_prompt("Keep automatic fallback to polling if webhook setup fails?")
        .default(true)
        .interact()?;
    if !keep_polling_fallback {
        println!("  Safety note: aidaemon currently keeps polling fallback enabled.");
        println!("  This setup keeps fallback enabled to avoid regressions.");
    }

    let store_cf_api_token = Confirm::new()
        .with_prompt("Store Cloudflare API token in keychain for future automation?")
        .default(false)
        .interact()?;
    if store_cf_api_token {
        let token: String = Input::new()
            .with_prompt("Cloudflare API token")
            .interact_text()?;
        match crate::config::store_in_keychain("cloudflare_api_token", token.trim()) {
            Ok(()) => println!("  Cloudflare API token stored in OS keychain."),
            Err(err) => println!("  Warning: could not store Cloudflare API token ({err})."),
        }
    }

    let port_usage = collect_existing_listen_port_usage(&all_bots);
    let listen_ports = assign_listen_ports(&selected_bots, &port_usage)?;

    let mut plans = Vec::with_capacity(selected_bots.len());
    for bot in selected_bots {
        let hostname = hostnames.get(&bot.target).cloned().ok_or_else(|| {
            anyhow::anyhow!("missing hostname assignment for {}", bot.display_name)
        })?;
        let port = listen_ports
            .get(&bot.target)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing listen port for {}", bot.display_name))?;
        let path = bot
            .existing
            .path
            .as_deref()
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| format!("/telegram/{}", bot.slug));
        plans.push(TelegramWebhookPlan {
            target: bot.target,
            display_name: bot.display_name,
            public_url: format!("https://{}", hostname),
            listen_addr: format!("{}:{}", listen_host, port),
            path,
            max_connections,
            drop_pending_updates,
        });
    }

    validate_webhook_plans(&plans)?;

    println!();
    println!("  Planned webhook updates:");
    for plan in &plans {
        println!("  - {}", plan.display_name);
        println!("      public_url: {}", plan.public_url);
        println!("      listen_addr: {}", plan.listen_addr);
        println!("      path: {}", plan.path);
    }
    println!();
    let confirm_apply = Confirm::new()
        .with_prompt("Apply these webhook settings to config.toml?")
        .default(true)
        .interact()?;
    if !confirm_apply {
        println!("  No changes were made.");
        return Ok(());
    }

    apply_webhook_plans(&mut doc, &plans)?;
    let backup_path = write_config_with_backup(config_path, &doc)?;

    println!();
    println!(
        "  Updated {} and created backup at {}",
        config_path.display(),
        backup_path.display()
    );
    println!("  Webhook mode is now enabled for selected bot(s).");
    println!("  Polling fallback remains enabled for safety.");
    println!();
    print_low_latency_next_steps(probe.environment(), &plans);
    Ok(())
}

pub fn low_latency_status_summary(config_path: &Path) -> anyhow::Result<String> {
    let doc = load_config_doc(config_path)?;
    let bots = collect_telegram_bot_entries(&doc);
    if bots.is_empty() {
        anyhow::bail!(
            "No Telegram bots found in config. Configure [telegram] or [[telegram_bots]] first."
        );
    }

    let probe = probe_runtime_environment();
    let mut out = String::new();
    out.push_str("Low-latency webhook status\n\n");
    out.push_str(&format!(
        "Runtime mode: {}\n",
        runtime_mode_label(probe.environment())
    ));
    out.push_str(&format!(
        "- wrangler installed: {}\n",
        yes_no(probe.wrangler_installed)
    ));
    out.push_str(&format!(
        "- wrangler authenticated: {}\n",
        yes_no(probe.wrangler_authenticated)
    ));
    out.push_str(&format!(
        "- cloudflared installed: {}\n",
        yes_no(probe.cloudflared_installed)
    ));
    out.push_str(&format!(
        "- cloudflared authenticated: {}\n",
        yes_no(probe.cloudflared_authenticated)
    ));
    out.push_str(&format!(
        "- tunnel `aidaemon-telegram` exists: {}\n",
        yes_no(probe.cloudflared_tunnel_exists)
    ));
    out.push_str(&format!(
        "- local tunnel process: {}\n",
        local_tunnel_process_status(probe.cloudflared_local_tunnel_running)
    ));
    out.push('\n');
    out.push_str("Bots:\n");
    for bot in bots {
        let enabled = yes_no(bot.existing.enabled);
        let public_url = bot.existing.public_url.as_deref().unwrap_or("n/a");
        let listen_addr = bot.existing.listen_addr.as_deref().unwrap_or("n/a");
        let path = bot.existing.path.as_deref().unwrap_or("n/a");
        out.push_str(&format!(
            "- {} | webhook enabled: {}\n  public_url: {}\n  listen_addr: {}\n  path: {}\n",
            bot.display_name, enabled, public_url, listen_addr, path
        ));
    }
    out.push('\n');
    out.push_str(
        "Use `/setup lowlatency plan <base-domain>` to preview and `/setup lowlatency apply <base-domain>` to apply local webhook config and prepare tunnel/DNS steps.",
    );
    Ok(out)
}

pub fn low_latency_plan_from_base_domain_with_dynamic(
    config_path: &Path,
    base_domain: &str,
    dynamic_bots: &[crate::traits::DynamicBot],
) -> anyhow::Result<String> {
    let doc = load_config_doc(config_path)?;
    let plans =
        build_low_latency_plans_for_all_bots(&doc, base_domain, 40, false, true, dynamic_bots)?;
    let mut out = String::new();
    out.push_str("Low-latency webhook plan (dry run)\n\n");
    out.push_str(&format!(
        "Runtime mode: {}\n\n",
        runtime_mode_label(probe_runtime_environment().environment())
    ));
    append_plan_lines(&mut out, &plans);
    Ok(out)
}

pub fn low_latency_apply_from_base_domain_with_dynamic(
    config_path: &Path,
    base_domain: &str,
    dynamic_bots: &[crate::traits::DynamicBot],
) -> anyhow::Result<String> {
    let mut doc = load_config_doc(config_path)?;
    let plans =
        build_low_latency_plans_for_all_bots(&doc, base_domain, 40, false, true, dynamic_bots)?;
    apply_webhook_plans(&mut doc, &plans)?;

    // Write [telegram_webhook_defaults] so dynamic bots auto-derive on restart.
    let has_dynamic = plans
        .iter()
        .any(|p| matches!(p.target, TelegramBotTarget::Dynamic(_)));
    if has_dynamic {
        apply_telegram_webhook_defaults(&mut doc, base_domain);
    }

    let backup_path = write_config_with_backup(config_path, &doc)?;

    let mut out = String::new();
    out.push_str("Low-latency webhook config applied (local config only).\n");
    out.push_str(&format!("Backup: {}\n\n", backup_path.display()));
    out.push_str("Cloudflare tunnel/DNS commands are prepared below.\n\n");
    append_plan_lines(&mut out, &plans);
    out.push('\n');
    out.push_str("Next steps:\n");
    append_low_latency_next_steps(&mut out, probe_runtime_environment().environment(), &plans);
    Ok(out)
}

pub fn low_latency_cloudflared_commands_from_base_domain_with_dynamic(
    config_path: &Path,
    base_domain: &str,
    dynamic_bots: &[crate::traits::DynamicBot],
) -> anyhow::Result<Vec<SetupCommandSpec>> {
    let doc = load_config_doc(config_path)?;
    let plans =
        build_low_latency_plans_for_all_bots(&doc, base_domain, 40, false, true, dynamic_bots)?;
    Ok(low_latency_cloudflared_commands_for_plans(&plans))
}

pub fn low_latency_cloudflared_ingress_routes_from_base_domain_with_dynamic(
    config_path: &Path,
    base_domain: &str,
    dynamic_bots: &[crate::traits::DynamicBot],
) -> anyhow::Result<Vec<CloudflaredIngressRoute>> {
    let doc = load_config_doc(config_path)?;
    let plans =
        build_low_latency_plans_for_all_bots(&doc, base_domain, 40, false, true, dynamic_bots)?;
    Ok(low_latency_cloudflared_ingress_routes_for_plans(&plans))
}

fn load_config_doc(config_path: &Path) -> anyhow::Result<toml::Table> {
    let raw = std::fs::read_to_string(config_path)?;
    let doc: toml::Table = raw.parse()?;
    Ok(doc)
}

fn build_low_latency_plans_for_all_bots(
    doc: &toml::Table,
    base_domain: &str,
    max_connections: u8,
    drop_pending_updates: bool,
    bind_local_only: bool,
    dynamic_bots: &[crate::traits::DynamicBot],
) -> anyhow::Result<Vec<TelegramWebhookPlan>> {
    let mut bots = collect_telegram_bot_entries(doc);
    bots.extend(collect_dynamic_telegram_bot_entries(dynamic_bots));
    if bots.is_empty() {
        anyhow::bail!(
            "No Telegram bots found (config or dynamic). Configure [telegram], [[telegram_bots]], or /connect a bot first."
        );
    }

    let normalized_domain = normalize_domain_input(base_domain)?;
    let usage = collect_existing_listen_port_usage(&bots);
    let assigned = assign_listen_ports(&bots, &usage)?;
    let listen_host = if bind_local_only {
        "127.0.0.1"
    } else {
        "0.0.0.0"
    };

    let mut used_labels = HashSet::new();
    let mut plans = Vec::with_capacity(bots.len());
    for bot in bots {
        let label = unique_dns_label(&bot.slug, &mut used_labels);
        let port = assigned
            .get(&bot.target)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing listen port assignment"))?;
        let path = bot
            .existing
            .path
            .as_deref()
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| format!("/telegram/{}", bot.slug));
        let base_url = format!("https://{}.{}", label, normalized_domain);
        // The public_url must include the path: it goes verbatim to
        // Telegram's setWebhook and the listener serves that path. Without
        // it, Telegram delivers updates to "/" instead of "/telegram/<slug>".
        let public_url = format!("{}{}", base_url, path);
        plans.push(TelegramWebhookPlan {
            target: bot.target,
            display_name: bot.display_name,
            public_url,
            listen_addr: format!("{}:{}", listen_host, port),
            path,
            max_connections,
            drop_pending_updates,
        });
    }
    validate_webhook_plans(&plans)?;
    Ok(plans)
}

fn low_latency_cloudflared_commands_for_plans(
    plans: &[TelegramWebhookPlan],
) -> Vec<SetupCommandSpec> {
    let mut out = vec![SetupCommandSpec {
        program: "cloudflared".to_string(),
        args: vec![
            "tunnel".to_string(),
            "create".to_string(),
            "aidaemon-telegram".to_string(),
        ],
    }];
    let mut seen_hosts = HashSet::new();
    for host in plans
        .iter()
        .filter_map(|plan| extract_hostname_from_url(&plan.public_url))
    {
        if !seen_hosts.insert(host.clone()) {
            continue;
        }
        out.push(SetupCommandSpec {
            program: "cloudflared".to_string(),
            args: vec![
                "tunnel".to_string(),
                "route".to_string(),
                "dns".to_string(),
                "aidaemon-telegram".to_string(),
                host,
            ],
        });
    }
    out
}

fn low_latency_cloudflared_ingress_routes_for_plans(
    plans: &[TelegramWebhookPlan],
) -> Vec<CloudflaredIngressRoute> {
    let mut out = Vec::new();
    let mut seen_hosts = HashSet::new();
    for plan in plans {
        let Some(hostname) = extract_hostname_from_url(&plan.public_url) else {
            continue;
        };
        if !seen_hosts.insert(hostname.clone()) {
            continue;
        }
        out.push(CloudflaredIngressRoute {
            hostname,
            service: format!("http://{}", plan.listen_addr),
        });
    }
    out
}

fn append_plan_lines(out: &mut String, plans: &[TelegramWebhookPlan]) {
    out.push_str("Planned webhook settings:\n");
    for plan in plans {
        out.push_str(&format!(
            "- {}\n  public_url: {}\n  listen_addr: {}\n  path: {}\n",
            plan.display_name, plan.public_url, plan.listen_addr, plan.path
        ));
    }
}

fn runtime_mode_label(environment: LowLatencyEnvironment) -> &'static str {
    match environment {
        LowLatencyEnvironment::WranglerAndCloudflared => "wrangler + cloudflared (near one-click)",
        LowLatencyEnvironment::WranglerOnly => "wrangler only",
        LowLatencyEnvironment::CloudflaredOnly => "cloudflared only",
        LowLatencyEnvironment::Neither => "no cloud tooling detected",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn probe_runtime_environment() -> RuntimeProbe {
    let wrangler_installed = command_exists_sync("wrangler");
    let cloudflared_installed = command_exists_sync("cloudflared");
    let wrangler_authenticated =
        wrangler_installed && command_succeeds_sync("wrangler", &["whoami", "--json"]);
    let cloudflared_authenticated =
        cloudflared_installed && command_succeeds_sync("cloudflared", &["tunnel", "list"]);

    RuntimeProbe {
        wrangler_installed,
        wrangler_authenticated,
        cloudflared_installed,
        cloudflared_authenticated,
        cloudflared_tunnel_exists: cloudflared_authenticated
            && command_succeeds_sync("cloudflared", &["tunnel", "info", "aidaemon-telegram"]),
        cloudflared_local_tunnel_running: probe_local_cloudflared_tunnel_process(),
    }
}

fn command_exists_sync(binary: &str) -> bool {
    Command::new(binary).arg("--help").output().is_ok()
}

fn command_succeeds_sync(binary: &str, args: &[&str]) -> bool {
    Command::new(binary)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn probe_local_cloudflared_tunnel_process() -> Option<bool> {
    if !command_exists_sync("pgrep") {
        return None;
    }

    let output = Command::new("pgrep")
        .args(["-f", "cloudflared tunnel run aidaemon-telegram"])
        .output()
        .ok()?;
    Some(output.status.success())
}

fn local_tunnel_process_status(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "running",
        Some(false) => "not running",
        None => "unknown (pgrep unavailable)",
    }
}

fn print_runtime_probe_summary(probe: RuntimeProbe) {
    println!("  Runtime probe:");
    println!(
        "  - wrangler installed: {}",
        if probe.wrangler_installed {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  - wrangler authenticated: {}",
        if probe.wrangler_authenticated {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  - cloudflared installed: {}",
        if probe.cloudflared_installed {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  - cloudflared authenticated: {}",
        if probe.cloudflared_authenticated {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  - tunnel `aidaemon-telegram` exists: {}",
        if probe.cloudflared_tunnel_exists {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  - local tunnel process: {}",
        local_tunnel_process_status(probe.cloudflared_local_tunnel_running)
    );
    match probe.environment() {
        LowLatencyEnvironment::WranglerAndCloudflared => {
            println!("  - mode: near one-click (tunnel + DNS can be configured quickly)");
        }
        LowLatencyEnvironment::WranglerOnly => {
            println!("  - mode: wrangler-only (add cloudflared login or API token automation)");
        }
        LowLatencyEnvironment::CloudflaredOnly => {
            println!(
                "  - mode: cloudflared-only (webhook can work; wrangler integration optional)"
            );
        }
        LowLatencyEnvironment::Neither => {
            println!("  - mode: no cloud tooling detected (polling remains the safe default)");
        }
    }
    println!();
}

fn collect_telegram_bot_entries(doc: &toml::Table) -> Vec<TelegramBotEntry> {
    let mut out = Vec::new();

    if let Some(legacy) = doc.get("telegram").and_then(|v| v.as_table()) {
        let has_token = legacy
            .get("bot_token")
            .and_then(|v| v.as_str())
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        if has_token {
            out.push(TelegramBotEntry {
                target: TelegramBotTarget::Legacy,
                display_name: format!(
                    "telegram (legacy, token: {})",
                    token_source_hint(legacy.get("bot_token").and_then(|v| v.as_str()))
                ),
                slug: "legacy".to_string(),
                existing: extract_existing_webhook(legacy),
            });
        }
    }

    if let Some(bots) = doc.get("telegram_bots").and_then(|v| v.as_array()) {
        for (idx, bot) in bots.iter().enumerate() {
            let Some(table) = bot.as_table() else {
                continue;
            };
            let has_token = table
                .get("bot_token")
                .and_then(|v| v.as_str())
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false);
            if !has_token {
                continue;
            }
            out.push(TelegramBotEntry {
                target: TelegramBotTarget::ArrayIndex(idx),
                display_name: format!(
                    "telegram_bots[{}] (token: {})",
                    idx,
                    token_source_hint(table.get("bot_token").and_then(|v| v.as_str()))
                ),
                slug: format!("bot-{}", idx + 1),
                existing: extract_existing_webhook(table),
            });
        }
    }

    out
}

/// Build `TelegramBotEntry` items for dynamic bots (added via `/connect`).
/// These use `TelegramBotTarget::Dynamic` and are never written to config TOML.
fn collect_dynamic_telegram_bot_entries(
    dynamic_bots: &[crate::traits::DynamicBot],
) -> Vec<TelegramBotEntry> {
    dynamic_bots
        .iter()
        .filter(|b| b.channel_type == "telegram")
        .map(|bot| {
            let extra: serde_json::Value =
                serde_json::from_str(&bot.extra_config).unwrap_or_default();
            let username = extra["username"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("unknown");
            let slug = username.to_string();
            TelegramBotEntry {
                target: TelegramBotTarget::Dynamic(bot.id),
                display_name: format!("dynamic @{} (id: {})", username, bot.id),
                slug,
                existing: ExistingWebhookSettings::default(),
            }
        })
        .collect()
}

fn extract_existing_webhook(bot_table: &toml::Table) -> ExistingWebhookSettings {
    let Some(webhook) = bot_table.get("webhook").and_then(|v| v.as_table()) else {
        return ExistingWebhookSettings::default();
    };

    ExistingWebhookSettings {
        enabled: webhook
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        public_url: webhook
            .get("public_url")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string()),
        listen_addr: webhook
            .get("listen_addr")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string()),
        path: webhook
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string()),
    }
}

fn token_source_hint(token: Option<&str>) -> &'static str {
    let Some(value) = token.map(str::trim) else {
        return "missing";
    };
    if value.is_empty() {
        "missing"
    } else if value == "keychain" {
        "keychain"
    } else if value.starts_with("${") && value.ends_with('}') {
        "env"
    } else {
        "set"
    }
}

fn normalize_domain_input(raw: &str) -> anyhow::Result<String> {
    let trimmed = raw.trim().trim_end_matches('.').to_ascii_lowercase();
    if trimmed.is_empty() {
        anyhow::bail!("domain cannot be empty");
    }
    if trimmed.contains("://") {
        anyhow::bail!("domain must not include URL scheme");
    }
    if trimmed.contains('/') {
        anyhow::bail!("domain must not include path segments");
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '.')
    {
        anyhow::bail!("domain contains invalid characters");
    }
    if !trimmed.contains('.') {
        anyhow::bail!("domain should include at least one dot");
    }
    Ok(trimmed)
}

fn normalize_hostname_input(raw: &str) -> anyhow::Result<String> {
    let trimmed = raw.trim().trim_end_matches('.').to_ascii_lowercase();
    if trimmed.is_empty() {
        anyhow::bail!("hostname cannot be empty");
    }
    if trimmed.contains("://") || trimmed.contains('/') {
        anyhow::bail!("hostname must be host-only, without scheme/path");
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '.')
    {
        anyhow::bail!("hostname contains invalid characters");
    }
    if trimmed.starts_with('.') || trimmed.ends_with('.') || trimmed.contains("..") {
        anyhow::bail!("hostname is malformed");
    }
    Ok(trimmed)
}

fn unique_dns_label(base: &str, used: &mut HashSet<String>) -> String {
    let mut sanitized = base
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized = sanitized.trim_matches('-').to_string();
    if sanitized.is_empty() {
        sanitized = "bot".to_string();
    }
    if sanitized.len() > 40 {
        sanitized.truncate(40);
        sanitized = sanitized.trim_matches('-').to_string();
    }
    if sanitized.is_empty() {
        sanitized = "bot".to_string();
    }

    let mut candidate = sanitized.clone();
    let mut suffix = 2usize;
    while used.contains(&candidate) {
        candidate = format!("{}-{}", sanitized, suffix);
        suffix = suffix.saturating_add(1);
    }
    used.insert(candidate.clone());
    candidate
}

fn parse_max_connections(raw: &str) -> anyhow::Result<u8> {
    let value = raw.trim().parse::<u8>()?;
    if !(1..=100).contains(&value) {
        anyhow::bail!("max_connections must be in the range 1..=100");
    }
    Ok(value)
}

fn parse_listen_port(listen_addr: &str) -> Option<u16> {
    listen_addr
        .trim()
        .parse::<SocketAddr>()
        .ok()
        .map(|addr| addr.port())
}

fn collect_existing_listen_port_usage(bots: &[TelegramBotEntry]) -> HashMap<u16, usize> {
    let mut usage = HashMap::new();
    for bot in bots {
        let Some(addr) = bot.existing.listen_addr.as_deref() else {
            continue;
        };
        let Some(port) = parse_listen_port(addr) else {
            continue;
        };
        *usage.entry(port).or_insert(0) += 1;
    }
    usage
}

fn assign_listen_ports(
    selected_bots: &[TelegramBotEntry],
    all_port_usage: &HashMap<u16, usize>,
) -> anyhow::Result<HashMap<TelegramBotTarget, u16>> {
    let mut occupied: HashSet<u16> = all_port_usage.keys().copied().collect();
    let mut assigned: HashMap<TelegramBotTarget, u16> = HashMap::new();

    for bot in selected_bots {
        if let Some(existing_port) = bot
            .existing
            .listen_addr
            .as_deref()
            .and_then(parse_listen_port)
            && all_port_usage.get(&existing_port).copied().unwrap_or(0) <= 1
            && !assigned.values().any(|&port| port == existing_port)
        {
            assigned.insert(bot.target, existing_port);
            continue;
        }

        let mut candidate = 8443u16;
        while occupied.contains(&candidate) {
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("unable to find a free listen port"))?;
        }
        occupied.insert(candidate);
        assigned.insert(bot.target, candidate);
    }
    Ok(assigned)
}

fn validate_webhook_plans(plans: &[TelegramWebhookPlan]) -> anyhow::Result<()> {
    let mut listen_addrs = HashSet::new();
    for plan in plans {
        let parsed_url = reqwest::Url::parse(&plan.public_url)?;
        if parsed_url.scheme() != "https" {
            anyhow::bail!(
                "public_url for {} must use HTTPS (got {})",
                plan.display_name,
                plan.public_url
            );
        }
        if parsed_url.host_str().is_none() {
            anyhow::bail!("public_url for {} has no hostname", plan.display_name);
        }
        let _ = plan.listen_addr.parse::<SocketAddr>()?;
        if !listen_addrs.insert(plan.listen_addr.clone()) {
            anyhow::bail!("duplicate listen_addr detected: {}", plan.listen_addr);
        }
        if !plan.path.starts_with('/') {
            anyhow::bail!("webhook path for {} must start with '/'", plan.display_name);
        }
        if plan.max_connections == 0 || plan.max_connections > 100 {
            anyhow::bail!("max_connections for {} must be 1..=100", plan.display_name);
        }
    }
    Ok(())
}

fn extract_hostname_from_url(raw: &str) -> Option<String> {
    reqwest::Url::parse(raw)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_string()))
}

fn apply_webhook_plans(doc: &mut toml::Table, plans: &[TelegramWebhookPlan]) -> anyhow::Result<()> {
    for plan in plans {
        match plan.target {
            TelegramBotTarget::Dynamic(_) => {
                // Dynamic bots use global defaults at startup; skip TOML mutation.
                continue;
            }
            TelegramBotTarget::Legacy => {
                let webhook_value = toml::Value::Table(webhook_table_from_plan(plan));
                let Some(telegram) = doc.get_mut("telegram").and_then(|v| v.as_table_mut()) else {
                    anyhow::bail!("Legacy [telegram] section is missing");
                };
                telegram.insert("webhook".to_string(), webhook_value);
            }
            TelegramBotTarget::ArrayIndex(idx) => {
                let webhook_value = toml::Value::Table(webhook_table_from_plan(plan));
                let Some(bots) = doc.get_mut("telegram_bots").and_then(|v| v.as_array_mut()) else {
                    anyhow::bail!("[[telegram_bots]] section is missing");
                };
                let Some(bot_table) = bots.get_mut(idx).and_then(|v| v.as_table_mut()) else {
                    anyhow::bail!("telegram_bots[{}] is missing", idx);
                };
                bot_table.insert("webhook".to_string(), webhook_value);
            }
        }
    }
    Ok(())
}

fn webhook_table_from_plan(plan: &TelegramWebhookPlan) -> toml::Table {
    let mut webhook = toml::Table::new();
    webhook.insert("enabled".to_string(), toml::Value::Boolean(true));
    webhook.insert(
        "public_url".to_string(),
        toml::Value::String(plan.public_url.clone()),
    );
    webhook.insert(
        "listen_addr".to_string(),
        toml::Value::String(plan.listen_addr.clone()),
    );
    webhook.insert("path".to_string(), toml::Value::String(plan.path.clone()));
    webhook.insert(
        "max_connections".to_string(),
        toml::Value::Integer(i64::from(plan.max_connections)),
    );
    webhook.insert(
        "drop_pending_updates".to_string(),
        toml::Value::Boolean(plan.drop_pending_updates),
    );
    webhook
}

/// Write `[telegram_webhook_defaults]` into the TOML doc so that dynamic bots
/// auto-derive webhook config on restart (without per-bot TOML entries).
fn apply_telegram_webhook_defaults(doc: &mut toml::Table, base_domain: &str) {
    let mut section = toml::Table::new();
    section.insert("enabled".to_string(), toml::Value::Boolean(true));
    section.insert(
        "base_domain".to_string(),
        toml::Value::String(base_domain.to_string()),
    );
    section.insert("port_start".to_string(), toml::Value::Integer(8443));
    section.insert("max_connections".to_string(), toml::Value::Integer(40));
    section.insert(
        "drop_pending_updates".to_string(),
        toml::Value::Boolean(false),
    );
    section.insert("bind_local_only".to_string(), toml::Value::Boolean(true));
    doc.insert(
        "telegram_webhook_defaults".to_string(),
        toml::Value::Table(section),
    );
}

fn write_config_with_backup(
    config_path: &Path,
    doc: &toml::Table,
) -> anyhow::Result<std::path::PathBuf> {
    let backup_path = config_path.with_extension("toml.lowlatency.bak");
    std::fs::copy(config_path, &backup_path)?;

    let rendered = toml::to_string_pretty(&toml::Value::Table(doc.clone()))?;
    rendered.parse::<toml::Table>()?;

    if let Err(write_err) = aidaemon_tools::config_file::write_owner_only(config_path, &rendered) {
        let _ = std::fs::copy(&backup_path, config_path);
        return Err(write_err.into());
    }

    match std::fs::read_to_string(config_path) {
        Ok(written) => {
            if let Err(parse_err) = written.parse::<toml::Table>() {
                let _ = std::fs::copy(&backup_path, config_path);
                return Err(anyhow::anyhow!(
                    "post-write config validation failed; restored backup: {}",
                    parse_err
                ));
            }
        }
        Err(read_err) => {
            let _ = std::fs::copy(&backup_path, config_path);
            return Err(anyhow::anyhow!(
                "failed to read written config; restored backup: {}",
                read_err
            ));
        }
    }

    Ok(backup_path)
}

fn print_low_latency_next_steps(environment: LowLatencyEnvironment, plans: &[TelegramWebhookPlan]) {
    let mut out = String::new();
    out.push_str("  Next steps:\n");
    append_low_latency_next_steps(&mut out, environment, plans);
    print!("{}", out);
}

fn append_low_latency_next_steps(
    out: &mut String,
    environment: LowLatencyEnvironment,
    plans: &[TelegramWebhookPlan],
) {
    match environment {
        LowLatencyEnvironment::WranglerAndCloudflared | LowLatencyEnvironment::CloudflaredOnly => {
            out.push_str("1. Create or reuse a named tunnel:\n");
            out.push_str("   cloudflared tunnel create aidaemon-telegram\n");
            out.push_str("2. Add DNS routes for your webhook hosts:\n");
            for host in plans
                .iter()
                .filter_map(|plan| extract_hostname_from_url(&plan.public_url))
            {
                out.push_str(&format!(
                    "   cloudflared tunnel route dns aidaemon-telegram {}\n",
                    host
                ));
            }
            out.push_str(
                "3. Run your tunnel and route traffic to the configured local listen_addr values.\n",
            );
        }
        LowLatencyEnvironment::WranglerOnly => {
            out.push_str("1. Install/login cloudflared (`cloudflared tunnel login`) or\n");
            out.push_str("2. Use Cloudflare API token automation when available.\n");
            out.push_str("3. Until then, polling fallback keeps bots working.\n");
        }
        LowLatencyEnvironment::Neither => {
            out.push_str("1. Keep polling as default, or\n");
            out.push_str("2. Install cloudflared + authenticate to enable webhook delivery.\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Cloudflare / cloudflared infrastructure helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) enum CloudflaredZoneValidation {
    Match {
        zone_name: String,
    },
    Mismatch {
        zone_name: String,
        mismatched_hosts: Vec<String>,
    },
    Unknown {
        reason: String,
    },
}

pub(crate) fn hostname_matches_zone(host: &str, zone: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    let zone = zone.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() || zone.is_empty() {
        return false;
    }
    host == zone || host.ends_with(&format!(".{}", zone))
}

pub(crate) fn cloudflared_origin_cert_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TUNNEL_ORIGIN_CERT") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    std::env::var("HOME")
        .ok()
        .map(|home| PathBuf::from(home).join(".cloudflared").join("cert.pem"))
}

/// The tunnel name when `spec` is `cloudflared tunnel create <name>`.
pub(crate) fn cloudflared_tunnel_created_by(spec: &SetupCommandSpec) -> Option<&str> {
    match spec.args.as_slice() {
        [tunnel, create, name]
            if spec.program == "cloudflared" && tunnel == "tunnel" && create == "create" =>
        {
            Some(name.as_str())
        }
        _ => None,
    }
}

/// Whether `cloudflared tunnel list --output json` lists a live (not
/// deleted) tunnel called `name`.
pub(crate) fn cloudflared_tunnel_listed(list_json: &str, name: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(list_json)
        .ok()
        .and_then(|list| list.as_array().cloned())
        .is_some_and(|tunnels| {
            tunnels.iter().any(|tunnel| {
                tunnel["name"].as_str() == Some(name)
                    && tunnel["deleted_at"]
                        .as_str()
                        .is_none_or(|at| at.is_empty() || at.starts_with("0001-01-01"))
            })
        })
}

/// Whether the owner's Cloudflare account already has the tunnel `name`,
/// read from `cloudflared tunnel list --output json`.
pub(crate) async fn cloudflared_tunnel_exists(name: &str, timeout_secs: u64) -> bool {
    let run = tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        tokio::process::Command::new("cloudflared")
            .args(["tunnel", "list", "--output", "json", "--name", name])
            .stdin(std::process::Stdio::null())
            .output(),
    )
    .await;
    match run {
        Ok(Ok(output)) if output.status.success() => {
            cloudflared_tunnel_listed(&String::from_utf8_lossy(&output.stdout), name)
        }
        _ => false,
    }
}

pub(crate) fn format_setup_command(spec: &SetupCommandSpec) -> String {
    format!("{} {}", spec.program, spec.args.join(" "))
}

pub(crate) fn installer_for_missing_tool(
    tool: &str,
    has_brew: bool,
    has_npm: bool,
) -> Option<SetupCommandSpec> {
    match tool {
        "cloudflared" if has_brew => Some(SetupCommandSpec {
            program: "brew".to_string(),
            args: vec!["install".to_string(), "cloudflared".to_string()],
        }),
        "wrangler" if has_npm => Some(SetupCommandSpec {
            program: "npm".to_string(),
            args: vec![
                "install".to_string(),
                "-g".to_string(),
                "wrangler".to_string(),
            ],
        }),
        "wrangler" if has_brew => Some(SetupCommandSpec {
            program: "brew".to_string(),
            args: vec!["install".to_string(), "wrangler".to_string()],
        }),
        _ => None,
    }
}

pub(crate) fn summarize_setup_command_output(stdout: &[u8], stderr: &[u8]) -> String {
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    );
    let mut pieces = Vec::new();
    for line in combined.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        pieces.push(trimmed.to_string());
        if pieces.len() >= 2 {
            break;
        }
    }
    pieces.join(" | ")
}

pub(crate) fn setup_route_dns_hosts(specs: &[SetupCommandSpec]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for spec in specs {
        if !spec.program.eq_ignore_ascii_case("cloudflared") {
            continue;
        }
        if spec.args.len() < 5 {
            continue;
        }
        if spec.args[0] != "tunnel" || spec.args[1] != "route" || spec.args[2] != "dns" {
            continue;
        }
        let host = spec.args[4]
            .trim()
            .trim_end_matches('.')
            .to_ascii_lowercase();
        if host.is_empty() {
            continue;
        }
        if seen.insert(host.clone()) {
            out.push(host);
        }
    }
    out
}

pub(crate) fn extract_urls(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in text.split_whitespace() {
        let candidate = raw
            .trim_matches(|c: char| {
                matches!(
                    c,
                    '"' | '\'' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';'
                )
            })
            .trim();
        if (candidate.starts_with("https://") || candidate.starts_with("http://"))
            && seen.insert(candidate.to_string())
        {
            out.push(candidate.to_string());
        }
    }
    out
}

pub(crate) fn summarize_setup_log_lines(lines: &[String]) -> String {
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.to_string());
        if out.len() >= 2 {
            break;
        }
    }
    out.join(" | ")
}

pub(crate) fn render_cloudflared_ingress_config(routes: &[CloudflaredIngressRoute]) -> String {
    let mut out = String::from("# Managed by aidaemon /setup lowlatency apply\ningress:\n");
    for route in routes {
        out.push_str(&format!(
            "  - hostname: {}\n    service: {}\n",
            route.hostname, route.service
        ));
    }
    out.push_str("  - service: http_status:404\n");
    out
}

pub(crate) async fn command_exists(binary: &str) -> bool {
    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::process::Command::new(binary)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .map(|res| res.is_ok())
    .unwrap_or(false)
}

pub(crate) async fn command_succeeds(binary: &str, args: &[&str], timeout_secs: u64) -> bool {
    tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        tokio::process::Command::new(binary)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .map(|res| res.map(|o| o.status.success()).unwrap_or(false))
    .unwrap_or(false)
}

pub(crate) async fn cloudflared_authenticated_zone_name() -> Result<Option<String>, String> {
    let Some(cert_path) = cloudflared_origin_cert_path() else {
        return Ok(None);
    };
    let cert_content = match tokio::fs::read_to_string(&cert_path).await {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(format!(
                "failed to read cloudflared origin cert at `{}`: {}",
                cert_path.display(),
                err
            ));
        }
    };
    let payload = cert_content
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("-----"))
        .collect::<String>();
    if payload.is_empty() {
        return Ok(None);
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .map_err(|err| format!("failed to decode cloudflared cert payload: {}", err))?;
    let parsed: serde_json::Value = serde_json::from_slice(&decoded)
        .map_err(|err| format!("failed to parse cloudflared cert payload: {}", err))?;
    let zone_id = parsed
        .get("zoneID")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or_default();
    let api_token = parsed
        .get("apiToken")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or_default();
    if zone_id.is_empty() || api_token.is_empty() {
        return Ok(None);
    }
    let url = format!("https://api.cloudflare.com/client/v4/zones/{}", zone_id);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|err| format!("failed to build Cloudflare HTTP client: {}", err))?;
    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_token))
        .header("Content-Type", "application/json")
        .send()
        .await
        .map_err(|err| format!("Cloudflare zone lookup request failed: {}", err))?;
    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|err| format!("failed to parse Cloudflare zone lookup response: {}", err))?;
    if !status.is_success() {
        let message = body
            .get("errors")
            .and_then(|errors| errors.as_array())
            .and_then(|errors| errors.first())
            .and_then(|err| err.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(format!(
            "Cloudflare zone lookup failed (HTTP {}): {}",
            status, message
        ));
    }
    let zone_name = body
        .get("result")
        .and_then(|result| result.get("name"))
        .and_then(|v| v.as_str())
        .map(|name| name.trim().trim_end_matches('.').to_ascii_lowercase())
        .filter(|name| !name.is_empty());
    Ok(zone_name)
}

pub(crate) async fn validate_cloudflared_zone_for_hosts(
    hosts: &[String],
) -> CloudflaredZoneValidation {
    if hosts.is_empty() {
        return CloudflaredZoneValidation::Unknown {
            reason: "no DNS hosts were generated for this setup".to_string(),
        };
    }
    match cloudflared_authenticated_zone_name().await {
        Ok(Some(zone_name)) => {
            let mismatched_hosts = hosts
                .iter()
                .filter(|host| !hostname_matches_zone(host, &zone_name))
                .cloned()
                .collect::<Vec<_>>();
            if mismatched_hosts.is_empty() {
                CloudflaredZoneValidation::Match { zone_name }
            } else {
                CloudflaredZoneValidation::Mismatch {
                    zone_name,
                    mismatched_hosts,
                }
            }
        }
        Ok(None) => CloudflaredZoneValidation::Unknown {
            reason: "could not determine cloudflared zone from local credentials".to_string(),
        },
        Err(reason) => CloudflaredZoneValidation::Unknown { reason },
    }
}

fn managed_cloudflared_config_path(tunnel_name: &str) -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed)
                .join(".cloudflared")
                .join(format!("{}.aidaemon-ingress.yml", tunnel_name));
        }
    }
    PathBuf::from(format!("{}.aidaemon-ingress.yml", tunnel_name))
}

pub(crate) async fn write_cloudflared_ingress_config(
    tunnel_name: &str,
    routes: &[CloudflaredIngressRoute],
) -> Result<PathBuf, String> {
    if routes.is_empty() {
        return Err("no ingress routes were generated for cloudflared".to_string());
    }
    let config_path = managed_cloudflared_config_path(tunnel_name);
    if let Some(parent) = config_path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|err| {
            format!(
                "failed to create cloudflared config directory `{}`: {}",
                parent.display(),
                err
            )
        })?;
    }
    let rendered = render_cloudflared_ingress_config(routes);
    tokio::fs::write(&config_path, rendered)
        .await
        .map_err(|err| {
            format!(
                "failed to write cloudflared ingress config `{}`: {}",
                config_path.display(),
                err
            )
        })?;
    Ok(config_path)
}

async fn running_cloudflared_tunnel_pids(tunnel_name: &str) -> Result<Vec<u32>, String> {
    let pattern = format!("cloudflared.*tunnel.*run.*{}", tunnel_name);
    let output = match tokio::process::Command::new("pgrep")
        .args(["-f", &pattern])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(format!(
                "failed to inspect existing cloudflared process: {}",
                err
            ));
        }
    };
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<u32>() {
            Ok(pid) if seen.insert(pid) => out.push(pid),
            _ => continue,
        }
    }
    Ok(out)
}

#[cfg(unix)]
async fn is_pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    tokio::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
async fn is_pid_alive(_pid: u32) -> bool {
    false
}

async fn stop_cloudflared_tunnel_processes(pids: &[u32]) -> Vec<u32> {
    if pids.is_empty() {
        return Vec::new();
    }
    #[cfg(unix)]
    {
        for pid in pids {
            let _ = tokio::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        for pid in pids {
            if is_pid_alive(*pid).await {
                let _ = tokio::process::Command::new("kill")
                    .args(["-KILL", &pid.to_string()])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut stopped = Vec::new();
        for pid in pids {
            if !is_pid_alive(*pid).await {
                stopped.push(*pid);
            }
        }
        stopped
    }
    #[cfg(not(unix))]
    {
        let _ = pids;
        Vec::new()
    }
}

pub(crate) async fn stop_existing_cloudflared_tunnel_processes(
    tunnel_name: &str,
) -> Result<Vec<u32>, String> {
    let running = running_cloudflared_tunnel_pids(tunnel_name).await?;
    if running.is_empty() {
        return Ok(Vec::new());
    }
    let stopped = stop_cloudflared_tunnel_processes(&running).await;
    if stopped.len() < running.len() {
        let stopped_set: HashSet<u32> = stopped.iter().copied().collect();
        let stuck: Vec<u32> = running
            .iter()
            .copied()
            .filter(|pid| !stopped_set.contains(pid))
            .collect();
        return Err(format!(
            "failed to stop existing tunnel process(es): {}",
            stuck
                .iter()
                .map(|pid| pid.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(stopped)
}

pub(crate) async fn start_cloudflared_tunnel_background(
    tunnel_name: &str,
    routes: &[CloudflaredIngressRoute],
) -> Result<String, String> {
    let config_path = write_cloudflared_ingress_config(tunnel_name, routes).await?;
    let stopped = stop_existing_cloudflared_tunnel_processes(tunnel_name).await?;
    let mut child = tokio::process::Command::new("cloudflared")
        .arg("tunnel")
        .arg("--config")
        .arg(&config_path)
        .arg("run")
        .arg(tunnel_name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            format!(
                "failed to spawn `cloudflared tunnel --config {} run {}`: {}",
                config_path.display(),
                tunnel_name,
                err
            )
        })?;

    let pid = child.id();
    tokio::time::sleep(Duration::from_secs(2)).await;
    match child.try_wait() {
        Ok(Some(status)) => Err(format!(
            "`cloudflared tunnel --config {} run {}` exited immediately with status {}",
            config_path.display(),
            tunnel_name,
            status
        )),
        Ok(None) => {
            let tunnel = tunnel_name.to_string();
            tokio::spawn(async move {
                match child.wait().await {
                    Ok(status) if status.success() => {
                        info!(tunnel = %tunnel, "cloudflared tunnel process exited");
                    }
                    Ok(status) => {
                        warn!(
                            tunnel = %tunnel,
                            status = %status,
                            "cloudflared tunnel process exited unexpectedly"
                        );
                    }
                    Err(err) => {
                        warn!(
                            tunnel = %tunnel,
                            error = %err,
                            "failed while waiting for cloudflared tunnel process"
                        );
                    }
                }
            });

            let mut pieces = Vec::new();
            if !stopped.is_empty() {
                pieces.push(format!(
                    "Stopped existing tunnel process(es): {}.",
                    stopped
                        .iter()
                        .map(|pid| pid.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if let Some(pid) = pid {
                pieces.push(format!(
                    "Wrote ingress config to `{}`.",
                    config_path.display()
                ));
                pieces.push(format!(
                    "Started `cloudflared tunnel --config {} run {}` in background (pid {}).",
                    config_path.display(),
                    tunnel_name,
                    pid
                ));
            } else {
                pieces.push(format!(
                    "Wrote ingress config to `{}`.",
                    config_path.display()
                ));
                pieces.push(format!(
                    "Started `cloudflared tunnel --config {} run {}` in background.",
                    config_path.display(),
                    tunnel_name
                ));
            }
            Ok(pieces.join(" "))
        }
        Err(err) => Err(format!(
            "failed to check cloudflared process status: {}",
            err
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CloudflaredIngressRoute, SetupCommandSpec, TelegramBotTarget, TelegramWebhookPlan,
        apply_webhook_plans, assign_listen_ports, build_low_latency_plans_for_all_bots,
        collect_telegram_bot_entries, hostname_matches_zone,
        low_latency_cloudflared_ingress_routes_for_plans, normalize_domain_input,
        normalize_hostname_input, render_cloudflared_ingress_config, setup_route_dns_hosts,
    };

    #[test]
    fn an_existing_tunnel_is_read_from_cloudflareds_json_list() {
        use super::{cloudflared_tunnel_created_by, cloudflared_tunnel_listed};
        let create = SetupCommandSpec {
            program: "cloudflared".to_string(),
            args: vec!["tunnel".into(), "create".into(), "aidaemon-telegram".into()],
        };
        assert_eq!(
            cloudflared_tunnel_created_by(&create),
            Some("aidaemon-telegram")
        );
        let route = SetupCommandSpec {
            program: "cloudflared".to_string(),
            args: vec![
                "tunnel".into(),
                "route".into(),
                "dns".into(),
                "aidaemon-telegram".into(),
                "bot.example.com".into(),
            ],
        };
        assert_eq!(cloudflared_tunnel_created_by(&route), None);
        let listed = r#"[{"id":"00000000-0000-0000-0000-000000000000","name":"aidaemon-telegram","created_at":"2026-11-01T00:00:00Z","deleted_at":"0001-01-01T00:00:00Z","connections":[]}]"#;
        assert!(cloudflared_tunnel_listed(listed, "aidaemon-telegram"));
        assert!(!cloudflared_tunnel_listed(listed, "another-tunnel"));
        assert!(!cloudflared_tunnel_listed("[]", "aidaemon-telegram"));
        // Prose saying it exists is not the list.
        assert!(!cloudflared_tunnel_listed(
            "tunnel with name already exists",
            "aidaemon-telegram"
        ));
    }

    #[test]
    fn normalize_domain_input_rejects_url_scheme() {
        assert!(normalize_domain_input("https://example.com").is_err());
        assert!(normalize_domain_input("example.com").is_ok());
    }

    #[test]
    fn normalize_hostname_input_rejects_path() {
        assert!(normalize_hostname_input("bot.example.com/path").is_err());
        assert!(normalize_hostname_input("bot.example.com").is_ok());
    }

    #[test]
    fn assign_listen_ports_preserves_unique_existing_and_fixes_conflicts() {
        let doc: toml::Table = r#"
[telegram]
bot_token = "legacy-token"
[telegram.webhook]
listen_addr = "127.0.0.1:8443"

[[telegram_bots]]
bot_token = "bot-1"
[telegram_bots.webhook]
listen_addr = "127.0.0.1:8443"

[[telegram_bots]]
bot_token = "bot-2"
[telegram_bots.webhook]
listen_addr = "127.0.0.1:8445"
"#
        .parse()
        .expect("parse toml");

        let bots = collect_telegram_bot_entries(&doc);
        let selected = vec![bots[0].clone(), bots[1].clone(), bots[2].clone()];
        let usage = super::collect_existing_listen_port_usage(&bots);
        let assigned = assign_listen_ports(&selected, &usage).expect("assign ports");

        // Legacy + first array bot had a conflict on 8443; one of them must move.
        let legacy = assigned[&TelegramBotTarget::Legacy];
        let bot0 = assigned[&TelegramBotTarget::ArrayIndex(0)];
        let bot1 = assigned[&TelegramBotTarget::ArrayIndex(1)];
        assert_ne!(legacy, bot0);
        assert_eq!(bot1, 8445);
    }

    #[test]
    fn apply_webhook_plans_updates_legacy_and_array_entries() {
        let mut doc: toml::Table = r#"
[telegram]
bot_token = "legacy-token"

[[telegram_bots]]
bot_token = "bot-1"
"#
        .parse()
        .expect("parse toml");

        let plans = vec![
            TelegramWebhookPlan {
                target: TelegramBotTarget::Legacy,
                display_name: "legacy".to_string(),
                public_url: "https://legacy.example.com".to_string(),
                listen_addr: "127.0.0.1:8443".to_string(),
                path: "/telegram/legacy".to_string(),
                max_connections: 40,
                drop_pending_updates: false,
            },
            TelegramWebhookPlan {
                target: TelegramBotTarget::ArrayIndex(0),
                display_name: "bot-1".to_string(),
                public_url: "https://bot1.example.com".to_string(),
                listen_addr: "127.0.0.1:8444".to_string(),
                path: "/telegram/bot-1".to_string(),
                max_connections: 40,
                drop_pending_updates: true,
            },
        ];

        apply_webhook_plans(&mut doc, &plans).expect("apply plans");

        let legacy_webhook = doc
            .get("telegram")
            .and_then(|v| v.as_table())
            .and_then(|v| v.get("webhook"))
            .and_then(|v| v.as_table())
            .expect("legacy webhook");
        assert_eq!(
            legacy_webhook
                .get("public_url")
                .and_then(|v| v.as_str())
                .unwrap_or_default(),
            "https://legacy.example.com"
        );

        let array_webhook = doc
            .get("telegram_bots")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_table())
            .and_then(|v| v.get("webhook"))
            .and_then(|v| v.as_table())
            .expect("array webhook");
        assert_eq!(
            array_webhook
                .get("listen_addr")
                .and_then(|v| v.as_str())
                .unwrap_or_default(),
            "127.0.0.1:8444"
        );
    }

    #[test]
    fn build_low_latency_plans_generates_https_hosts_and_unique_ports() {
        let doc: toml::Table = r#"
[telegram]
bot_token = "legacy-token"
[telegram.webhook]
listen_addr = "127.0.0.1:8443"
path = "/legacy"

[[telegram_bots]]
bot_token = "bot-1"
[telegram_bots.webhook]
listen_addr = "127.0.0.1:8443"

[[telegram_bots]]
bot_token = "bot-2"
"#
        .parse()
        .expect("parse toml");

        let plans =
            build_low_latency_plans_for_all_bots(&doc, "bots.example.com", 40, false, true, &[])
                .expect("build plans");
        assert_eq!(plans.len(), 3);

        let mut listen_addrs = std::collections::HashSet::new();
        for plan in &plans {
            assert!(plan.public_url.starts_with("https://"));
            assert!(listen_addrs.insert(plan.listen_addr.clone()));
        }
    }

    #[test]
    fn ingress_routes_derive_hostname_and_service_from_plans() {
        let plans = vec![
            TelegramWebhookPlan {
                target: TelegramBotTarget::Legacy,
                display_name: "legacy".to_string(),
                public_url: "https://legacy.example.com".to_string(),
                listen_addr: "127.0.0.1:8443".to_string(),
                path: "/telegram/legacy".to_string(),
                max_connections: 40,
                drop_pending_updates: false,
            },
            TelegramWebhookPlan {
                target: TelegramBotTarget::ArrayIndex(0),
                display_name: "bot-1".to_string(),
                public_url: "https://bot1.example.com".to_string(),
                listen_addr: "127.0.0.1:8444".to_string(),
                path: "/telegram/bot-1".to_string(),
                max_connections: 40,
                drop_pending_updates: false,
            },
            TelegramWebhookPlan {
                target: TelegramBotTarget::ArrayIndex(1),
                display_name: "duplicate-host".to_string(),
                public_url: "https://legacy.example.com".to_string(),
                listen_addr: "127.0.0.1:8555".to_string(),
                path: "/telegram/dup".to_string(),
                max_connections: 40,
                drop_pending_updates: false,
            },
        ];

        let routes = low_latency_cloudflared_ingress_routes_for_plans(&plans);
        assert_eq!(
            routes,
            vec![
                CloudflaredIngressRoute {
                    hostname: "legacy.example.com".to_string(),
                    service: "http://127.0.0.1:8443".to_string(),
                },
                CloudflaredIngressRoute {
                    hostname: "bot1.example.com".to_string(),
                    service: "http://127.0.0.1:8444".to_string(),
                },
            ]
        );
    }

    #[test]
    fn hostname_matches_zone_checks_suffix_boundaries() {
        assert!(hostname_matches_zone("legacy.aidaemon.ai", "aidaemon.ai"));
        assert!(hostname_matches_zone("aidaemon.ai", "aidaemon.ai"));
        assert!(!hostname_matches_zone("legacy.aidaemon.ai", "example.com"));
    }

    #[test]
    fn setup_route_dns_hosts_extracts_unique_cloudflared_hosts() {
        let specs = vec![
            SetupCommandSpec {
                program: "cloudflared".to_string(),
                args: vec![
                    "tunnel".to_string(),
                    "create".to_string(),
                    "aidaemon-telegram".to_string(),
                ],
            },
            SetupCommandSpec {
                program: "cloudflared".to_string(),
                args: vec![
                    "tunnel".to_string(),
                    "route".to_string(),
                    "dns".to_string(),
                    "aidaemon-telegram".to_string(),
                    "legacy.aidaemon.ai".to_string(),
                ],
            },
            SetupCommandSpec {
                program: "cloudflared".to_string(),
                args: vec![
                    "tunnel".to_string(),
                    "route".to_string(),
                    "dns".to_string(),
                    "aidaemon-telegram".to_string(),
                    "legacy.aidaemon.ai".to_string(),
                ],
            },
            SetupCommandSpec {
                program: "echo".to_string(),
                args: vec!["ignore".to_string()],
            },
        ];
        let hosts = setup_route_dns_hosts(&specs);
        assert_eq!(hosts, vec!["legacy.aidaemon.ai"]);
    }

    #[test]
    fn render_cloudflared_ingress_config_contains_routes_and_fallback() {
        let routes = vec![
            CloudflaredIngressRoute {
                hostname: "legacy.aidaemon.ai".to_string(),
                service: "http://127.0.0.1:8443".to_string(),
            },
            CloudflaredIngressRoute {
                hostname: "bot1.aidaemon.ai".to_string(),
                service: "http://127.0.0.1:8444".to_string(),
            },
        ];

        let rendered = render_cloudflared_ingress_config(&routes);
        assert!(rendered.contains("ingress:\n"));
        assert!(rendered.contains("hostname: legacy.aidaemon.ai"));
        assert!(rendered.contains("service: http://127.0.0.1:8443"));
        assert!(rendered.contains("hostname: bot1.aidaemon.ai"));
        assert!(rendered.contains("service: http://127.0.0.1:8444"));
        assert!(rendered.contains("service: http_status:404"));
    }
}
