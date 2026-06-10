fn main() {
    // The `axuielement` crate (feature `computer_use-macos`) statically links a
    // Swift bridge whose objects force-load Swift compatibility archives
    // (libswiftCompatibility56.a, libswiftCompatibilityConcurrency.a). On machines
    // with only Command Line Tools (no full Xcode), the auto-link search paths
    // baked into those objects point at a Toolchains/ layout that CLT does not
    // have, so linking fails with undefined __swift_FORCE_LOAD_$_swiftCompatibility*
    // symbols. Emit the CLT location explicitly; a build-script link-search path
    // propagates to bins, tests, and doctests alike. On machines with full Xcode
    // the directory is absent and nothing is emitted.
    println!("cargo:rerun-if-changed=build.rs");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let computer_use_macos = std::env::var("CARGO_FEATURE_COMPUTER_USE_MACOS").is_ok();
    if target_os == "macos" && computer_use_macos {
        let clt_swift = "/Library/Developer/CommandLineTools/usr/lib/swift/macosx";
        if std::path::Path::new(clt_swift).exists() {
            println!("cargo:rustc-link-search=native={clt_swift}");
        }
    }
}
