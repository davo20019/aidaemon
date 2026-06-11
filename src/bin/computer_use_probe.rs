//! Manual macOS probe for computer_use native stack.
//!
//! Usage:
//!   computer_use_probe --app Calculator --inspect
//!   computer_use_probe --app Calculator --click-index 3

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("computer_use_probe requires macOS");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    use std::env;

    let args: Vec<String> = env::args().collect();
    let mut app = "Calculator".to_string();
    let mut inspect_only = true;
    let mut click_index: Option<u32> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--app" => {
                i += 1;
                app = args.get(i).cloned().unwrap_or(app);
            }
            "--inspect" => inspect_only = true,
            "--click-index" => {
                inspect_only = false;
                i += 1;
                click_index = Some(
                    args.get(i)
                        .and_then(|s| s.parse().ok())
                        .expect("--click-index requires a number"),
                );
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(index) = click_index {
        run_click(&app, index);
    } else if inspect_only {
        run_inspect(&app);
    }
}

#[cfg(target_os = "macos")]
fn run_inspect(app: &str) {
    use std::path::PathBuf;

    use aidaemon::computer_use::cache::SnapshotCache;
    use aidaemon::computer_use::harness::{ComputerHarness, HarnessRequestContext};
    use aidaemon::computer_use::macos::MacOsHarness;
    use aidaemon::computer_use::types::format_full_tree;
    use aidaemon::ComputerUseConfig;

    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .try_init();

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let harness = MacOsHarness::new(ComputerUseConfig {
            enabled: true,
            ..Default::default()
        });
        harness.check_permissions().expect("permissions");
        let ctx = HarnessRequestContext {
            task_id: "probe".into(),
            session_id: "probe".into(),
        };
        let mut cache = SnapshotCache::default();
        let snapshot = harness
            .get_app_state(app, &ctx, &mut cache)
            .await
            .expect("get_app_state");
        println!("{}", format_full_tree(&snapshot));
        let out_path = PathBuf::from("computer_use_probe.png");
        std::fs::write(&out_path, &snapshot.png).expect("write png");
        println!("Wrote screenshot to {}", out_path.display());
    });
}

#[cfg(target_os = "macos")]
fn run_click(app: &str, index: u32) {
    use aidaemon::computer_use::cache::SnapshotCache;
    use aidaemon::computer_use::harness::{ComputerHarness, HarnessRequestContext};
    use aidaemon::computer_use::macos::MacOsHarness;
    use aidaemon::computer_use::types::format_condensed_refresh;
    use aidaemon::ComputerUseConfig;

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async {
        let harness = MacOsHarness::new(ComputerUseConfig {
            enabled: true,
            ..Default::default()
        });
        harness.check_permissions().expect("permissions");
        let ctx = HarnessRequestContext {
            task_id: "probe".into(),
            session_id: "probe".into(),
        };
        let mut cache = SnapshotCache::default();
        let snapshot = harness
            .get_app_state(app, &ctx, &mut cache)
            .await
            .expect("get_app_state");
        let generation = snapshot.generation;
        let (after, _, _) = harness
            .click(app, generation, Some(index), None, None, &ctx, &mut cache)
            .await
            .expect("click");
        println!("{}", format_condensed_refresh(&after, Some(index)));
    });
}
