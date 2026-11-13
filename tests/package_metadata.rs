use std::fs;
use std::path::Path;

#[test]
fn db_probe_is_feature_gated_for_binary_installers() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest: toml::Value =
        toml::from_str(&fs::read_to_string(&manifest_path).expect("read Cargo.toml"))
            .expect("parse Cargo.toml");

    let db_probe = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .and_then(|bins| {
            bins.iter()
                .find(|bin| bin.get("name").and_then(toml::Value::as_str) == Some("db_probe"))
        })
        .expect("Cargo.toml must explicitly declare the db_probe binary");

    assert_eq!(
        db_probe
            .get("required-features")
            .and_then(toml::Value::as_array)
            .expect("db_probe must be feature-gated")
            .iter()
            .filter_map(toml::Value::as_str)
            .collect::<Vec<_>>(),
        ["encryption"]
    );
}

/// A cargo feature only adds test helpers; it never picks a production
/// behaviour. The root crate's dev-dependencies turn `testing` on, and any
/// `cargo test` rebuilds `target/debug/aidaemon` with it, so a behaviour
/// chosen by it (a journal mode, an embedder, a desktop harness) would ship.
/// Hosts choose such behaviour through constructor inputs instead.
#[test]
fn no_production_behaviour_depends_on_the_testing_feature() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();
    let mut pending = vec![root.join("src"), root.join("crates")];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("read source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = fs::read_to_string(&path).expect("read source");
                for (number, line) in source.lines().enumerate() {
                    let compact: String = line.split_whitespace().collect();
                    // A value chosen by the feature, or code compiled only
                    // when the feature is off (the production branch of a
                    // choice the feature makes).
                    let chooses = (compact.contains("cfg!(") && compact.contains("\"testing\""))
                        || compact.contains("not(any(test,feature=\"testing\"))")
                        || compact.contains("not(feature=\"testing\")");
                    if chooses {
                        offenders.push(format!(
                            "{}:{}: {}",
                            path.strip_prefix(root).unwrap().display(),
                            number + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "behaviour picked by the `testing` feature:\n{}",
        offenders.join("\n")
    );
}
