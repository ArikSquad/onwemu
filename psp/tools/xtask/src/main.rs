//! Repository validation commands used by CI and local development.

use anyhow::{Context, Result, bail};
use std::process::Command;
fn run(program: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("failed to run {program}"))?;
    if !status.success() {
        bail!("{program} {} failed", args.join(" "))
    }
    Ok(())
}
fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() != Some("validate") {
        bail!("usage: cargo xtask validate")
    }
    run("cargo", &["fmt", "--all", "--", "--check"])?;
    run(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run("cargo", &["test", "--workspace", "--all-targets"])?;
    run("cargo", &["doc", "--workspace", "--no-deps"])
}
