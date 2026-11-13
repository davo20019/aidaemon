//! Write the Client Protocol's JSON Schema to
//! `schemas/client-protocol/v1.schema.json` (or the path given), the file
//! non-Rust clients generate their types from.
//!
//! `cargo run -p aidaemon-client-protocol --bin export-client-schema`

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../schemas/client-protocol/v1.schema.json")
        });
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, aidaemon_client_protocol::schema_file())?;
    println!("wrote {}", path.display());
    Ok(())
}
