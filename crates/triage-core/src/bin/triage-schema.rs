//! Writes the JSON schemas of the request and response types to a directory
//! (default `schema/`), for the Python harness. A triage-core test fails when
//! the committed files are out of date.
//!
//!   cargo run -p triage-core --bin triage-schema [DIR]

use std::path::PathBuf;

use triage_core::schema::{render, schemas};

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "schema".into()));
    std::fs::create_dir_all(&dir)?;
    for (name, schema) in schemas() {
        let path = dir.join(format!("{name}.schema.json"));
        std::fs::write(&path, render(&schema))?;
        println!("{}", path.display());
    }
    Ok(())
}
