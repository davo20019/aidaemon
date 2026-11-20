//! The version the app shows as its own: `AIDAEMON_APP_VERSION` and
//! `AIDAEMON_APP_BUILD` when the packaging sets them (releases pass the
//! same values to `nsh package --app-version/--build`), else `nsh.json`'s
//! `version` and `build`.

fn main() {
    println!("cargo:rerun-if-changed=nsh.json");
    println!("cargo:rerun-if-env-changed=AIDAEMON_APP_VERSION");
    println!("cargo:rerun-if-env-changed=AIDAEMON_APP_BUILD");
    let manifest: serde_json::Value = std::fs::read_to_string("nsh.json")
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    let from_env = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    let version = from_env("AIDAEMON_APP_VERSION")
        .or_else(|| manifest["version"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "0.0.0".into());
    let build = from_env("AIDAEMON_APP_BUILD")
        .or_else(|| manifest["build"].as_u64().map(|b| b.to_string()))
        .unwrap_or_default();
    println!("cargo:rustc-env=AIDAEMON_APP_VERSION={version}");
    println!("cargo:rustc-env=AIDAEMON_APP_BUILD={build}");
}
