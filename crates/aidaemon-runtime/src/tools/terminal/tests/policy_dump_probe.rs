use super::*;

/// Operator probe: write the exact seatbelt policy the daemon would use
/// for a build in a given project so it can be replayed with sandbox-exec.
/// Run with: AIDAEMON_POLICY_PROBE_DIR=<dir> AIDAEMON_POLICY_PROBE_CMD='npm run build'
/// Optional: AIDAEMON_POLICY_PROBE_READS=<a:b> AIDAEMON_POLICY_PROBE_WRITES=<a:b>
///           AIDAEMON_POLICY_PROBE_NETWORK=1
///   cargo test --lib policy_dump_probe -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn dump_policy_for_project_build() {
    let Ok(dir) = std::env::var("AIDAEMON_POLICY_PROBE_DIR") else {
        return;
    };
    let command =
        std::env::var("AIDAEMON_POLICY_PROBE_CMD").unwrap_or_else(|_| "npm run build".into());
    let backend = aidaemon_tools::testing::active_execution_backend();
    let writes = std::env::var("AIDAEMON_POLICY_PROBE_WRITES")
        .ok()
        .map(|w| w.split(':').map(str::to_string).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![format!("{dir}/dist"), dir.clone()]);
    let mut reads = vec![dir.clone()];
    if let Ok(extra) = std::env::var("AIDAEMON_POLICY_PROBE_READS") {
        reads.extend(
            extra
                .split(':')
                .filter(|p| !p.is_empty())
                .map(str::to_string),
        );
    }
    let capabilities = ConfinedCapabilities {
        network: std::env::var("AIDAEMON_POLICY_PROBE_NETWORK").is_ok_and(|v| v == "1"),
        ..ConfinedCapabilities::default()
    };
    let request = confined_terminal_execution_request_inner(
        &backend,
        &command,
        false,
        Some(&dir),
        &reads,
        &writes,
        &[],
        capabilities,
    )
    .await
    .expect("confined request");
    match &request.command {
        aidaemon_tools::CommandSpec::Argv { args, .. } => {
            let policy = args.get(1).cloned().unwrap_or_default();
            std::fs::write("/tmp/aidaemon-policy-probe.sb", &policy).unwrap();
            println!("ARGS={}", serde_json::to_string(args).unwrap());
            println!(
                "ENV={}",
                serde_json::to_string(&request.env.iter().collect::<Vec<_>>()).unwrap()
            );
        }
        other => println!("NON-ARGV: {other:?}"),
    }
}
