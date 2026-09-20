mod assets;
mod cli;
mod docs;
mod importer;
mod manifest;
mod ops;
mod preview;
mod project;
mod replay;
mod scaffold;
mod source;
mod state;

use std::process::ExitCode;

fn main() -> ExitCode {
    cli::run()
}
