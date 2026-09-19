mod assets;
mod cli;
mod docs;
mod manifest;
mod ops;
mod preview;
mod project;
mod scaffold;
mod state;

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    cli::run().await
}
