use std::fs;
use std::path::Path;

#[test]
fn db_probe_is_feature_gated_for_binary_installers() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest: toml::Value = fs::read_to_string(&manifest_path)
        .expect("read Cargo.toml")
        .parse()
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
