//! `aidaemon share-screen`: share this computer's main display with a
//! daemon elsewhere, so its owner can watch and drive it from the app like
//! the daemon's own screen. Pair once with `--pair <link>` (the link
//! `aidaemon pair` prints on the daemon's computer); it shares only while
//! this runs, and says so here and in a notification when it is driven.

#[cfg(all(target_os = "macos", feature = "computer_use-macos"))]
mod mac {
    use aidaemon_client_core::screen_share::{Captured, ScreenSource};
    use aidaemon_client_protocol::media::SurfaceInput;
    use aidaemon_runtime::tools::computer_use::live_desktop::LocalScreen;

    pub struct Shared(pub LocalScreen);

    impl ScreenSource for Shared {
        fn size(&self) -> (u32, u32) {
            self.0.size()
        }

        fn capture(&mut self, max_w: u32, max_h: u32, quality: u8) -> Result<Captured, String> {
            let (codec, w, h, bytes) = self.0.capture(max_w, max_h, quality)?;
            Ok(Captured { codec, w, h, bytes })
        }

        fn input(&mut self, event: SurfaceInput) -> Result<(), String> {
            let event =
                aidaemon_client_gateway::surfaces::input_of(event).map_err(str::to_string)?;
            self.0.input(event)
        }

        fn controlled(&mut self, on: bool) {
            self.0.controlled(on);
        }
    }
}

fn usage() {
    println!("Usage: aidaemon share-screen [--pair <link>] [--name <name>]");
    println!();
    println!("Share this computer's main display with your AIdaemon, so you can watch and");
    println!("control it from the app. Pair once with the link `aidaemon pair` prints on the");
    println!("daemon's computer. It shares only while this runs; press Ctrl-C to stop.");
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return Ok(());
    }
    let mut link = None;
    let mut name = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pair" => {
                link = args.get(i + 1).cloned();
                i += 2;
            }
            "--name" => {
                name = args.get(i + 1).cloned();
                i += 2;
            }
            other => anyhow::bail!("Unknown share-screen option: {other}"),
        }
    }
    let name = name.unwrap_or_else(|| hostname_or("This computer"));
    share(link, name)
}

fn hostname_or(fallback: &str) -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(all(target_os = "macos", feature = "computer_use-macos"))]
fn share(link: Option<String>, name: String) -> anyhow::Result<()> {
    use aidaemon_client_core::local;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        if let Some(link) = link {
            println!("Pairing… approve “{name}” on your daemon's chat or app.");
            local::pair_from_link(&link, &name, aidaemon_client_protocol::ClientKind::Desktop)
                .await
                .map_err(anyhow::Error::msg)?;
            println!("Paired.");
        }
        let (url, auth) = local::sign_in().ok_or_else(|| {
            anyhow::anyhow!("Not paired yet: run `aidaemon share-screen --pair <link>`")
        })?;
        let base = aidaemon_client_core::machine::http_base(&url);
        let screen = aidaemon_runtime::tools::computer_use::live_desktop::LocalScreen::primary()
            .map_err(anyhow::Error::msg)?;
        println!("Sharing “{name}” with {base}. Press Ctrl-C to stop.");
        tokio::select! {
            ended = aidaemon_client_core::screen_share::share(
                &base,
                &auth,
                &name,
                Box::new(mac::Shared(screen)),
                |state| println!("{state}"),
            ) => ended.map_err(anyhow::Error::msg),
            _ = tokio::signal::ctrl_c() => {
                println!("Stopped sharing.");
                Ok(())
            }
        }
    })
}

#[cfg(not(all(target_os = "macos", feature = "computer_use-macos")))]
fn share(_link: Option<String>, _name: String) -> anyhow::Result<()> {
    anyhow::bail!(
        "Sharing this screen needs the macOS build with desktop control (feature \
         computer_use-macos)."
    )
}
