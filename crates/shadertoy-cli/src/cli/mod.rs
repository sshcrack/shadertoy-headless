mod conversions;
use crate::docs;
use crate::ops;
use crate::ops::{ChannelSetOptions, InspectMode, Output, RenderOptions};
use crate::preview;
use crate::preview::PreviewConfig;
use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "shadertoy",
    version,
    about = "Agent-friendly ShaderToy project, rendering, inspection, and live-preview CLI"
)]
struct Cli {
    /// Emit machine-readable JSON results and errors on stdout.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create a new ShaderToy project directory.
    New(NewArgs),
    /// Initialize the current/existing directory as a ShaderToy project.
    Init(InitArgs),
    /// Import a ShaderToy URL or shader id through Camoufox into a local project.
    Import(ImportArgs),
    /// Validate the manifest, graph, assets, and GLSL compilation.
    Check(ProjectPathArgs),
    /// Build the project into an STTF artifact.
    Build(BuildArgs),
    /// Deterministically render the final image or a named 2D buffer pass.
    Render(RenderArgs),
    /// Run a native-rendered live preview web server with hot reload.
    Preview(PreviewArgs),
    /// Inspect project structure progressively, from summary to pass/channel detail.
    Inspect(InspectArgs),
    /// Capture, inspect, or modify resumable feedback-buffer state.
    State(StateArgs),
    /// Safely mutate shader passes in ShaderToy.toml.
    Pass(PassArgs),
    /// Safely mutate iChannel bindings in ShaderToy.toml.
    Channel(ChannelArgs),
    /// Read concise built-in project/agent documentation.
    Docs(DocsArgs),
}

#[derive(Debug, Args)]
struct NewArgs {
    /// Destination directory. Its basename becomes project.name.
    path: PathBuf,
    #[arg(long, value_enum, default_value_t = TemplateArg::Minimal)]
    template: TemplateArg,
}

#[derive(Debug, Args)]
struct InitArgs {
    #[arg(default_value = ".")]
    path: PathBuf,
    #[arg(long, value_enum, default_value_t = TemplateArg::Minimal)]
    template: TemplateArg,
}

#[derive(Debug, Args)]
struct ImportArgs {
    /// ShaderToy view URL or shader id.
    source: String,
    /// Destination directory. Defaults to a filesystem-safe form of the shader name.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ProjectPathArgs {
    /// Project directory (or any path inside it).
    #[arg(value_name = "PATH", conflicts_with = "project")]
    path: Option<PathBuf>,
    /// Project directory (or any path inside it).
    #[arg(long, value_name = "PATH")]
    project: Option<PathBuf>,
}

impl ProjectPathArgs {
    fn resolved(&self) -> PathBuf {
        self.project
            .clone()
            .or_else(|| self.path.clone())
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Project directory (or any path inside it).
    #[arg(value_name = "PATH", conflicts_with = "project")]
    path: Option<PathBuf>,
    /// Project directory (or any path inside it).
    #[arg(long, value_name = "PATH")]
    project: Option<PathBuf>,
    /// Output STTF path. Defaults to target/PROJECT.sttf.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct RenderArgs {
    /// Project directory (or any path inside it).
    #[arg(long, default_value = ".")]
    project: PathBuf,
    /// Output PNG path. Defaults to target/render.png.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Render/snapshot a named pass instead of the final Image pass.
    #[arg(long)]
    pass: Option<String>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    fps: Option<f32>,
    /// Deterministic target iFrame.
    #[arg(long, conflicts_with = "time")]
    frame: Option<i32>,
    /// Deterministic target iTime in seconds.
    #[arg(long, conflicts_with = "frame")]
    time: Option<f32>,
    /// Resume from a previously captured .ststate artifact.
    #[arg(long)]
    state: Option<PathBuf>,
    /// Override a persistent buffer immediately before the target frame, e.g. buffer-a=fixture.png.
    #[arg(long = "set-buffer")]
    set_buffers: Vec<String>,
}

#[derive(Debug, Args)]
struct PreviewArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 4321)]
    port: u16,
    /// Force opening the browser.
    #[arg(long)]
    open: bool,
    /// Never open the browser automatically.
    #[arg(long)]
    no_open: bool,
    /// Required when binding preview to a non-loopback address.
    #[arg(long)]
    token: Option<String>,
    /// Reset time/buffer state on every successful hot reload.
    #[arg(long)]
    reset_on_reload: bool,
}

#[derive(Debug, Args)]
struct InspectArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[command(subcommand)]
    command: Option<InspectCommand>,
}

#[derive(Debug, Subcommand)]
enum InspectCommand {
    /// Show the complete pass/channel dependency graph.
    Graph,
    /// Inspect one pass.
    Pass { name: String },
    /// Inspect one pass's iChannel bindings.
    Channels { name: String },
    /// Inspect a .ststate artifact.
    State { path: PathBuf },
}

#[derive(Debug, Args)]
struct StateArgs {
    #[command(subcommand)]
    command: StateCommand,
}

#[derive(Debug, Subcommand)]
enum StateCommand {
    /// Render deterministically to a point and capture lossless persistent buffer state.
    Capture(StateCaptureArgs),
    /// Inspect .ststate metadata without loading OpenGL.
    Inspect { path: PathBuf },
    /// Replace one or more buffers inside an existing state using exact-size images.
    Set(StateSetArgs),
}

#[derive(Debug, Args)]
struct StateCaptureArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    fps: Option<f32>,
    #[arg(long, conflicts_with = "time")]
    frame: Option<i32>,
    #[arg(long, conflicts_with = "frame")]
    time: Option<f32>,
}

#[derive(Debug, Args)]
struct StateSetArgs {
    input: PathBuf,
    /// BUFFER=IMAGE assignments.
    #[arg(required = true)]
    assignments: Vec<String>,
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct PassArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[command(subcommand)]
    command: PassCommand,
}

#[derive(Debug, Subcommand)]
enum PassCommand {
    /// Add an offscreen buffer/cubemap pass and create a source stub if needed.
    Add {
        name: String,
        #[arg(long, value_enum, default_value_t = PassKindArg::Buffer)]
        kind: PassKindArg,
        /// Source path relative to the project root.
        #[arg(long)]
        source: Option<PathBuf>,
    },
    /// Remove a non-final pass. Source files are intentionally left on disk.
    Remove {
        name: String,
        /// Also remove channel bindings that consume this pass.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Debug, Args)]
struct ChannelArgs {
    #[arg(long, default_value = ".")]
    project: PathBuf,
    #[command(subcommand)]
    command: ChannelCommand,
}

#[derive(Debug, Subcommand)]
enum ChannelCommand {
    /// Set/replace one iChannel binding.
    Set {
        pass: String,
        channel: u8,
        source: String,
        #[arg(long, value_enum)]
        kind: Option<InputKindArg>,
        #[arg(long)]
        previous: bool,
        #[arg(long, value_enum, default_value_t = FilterArg::Linear)]
        filter: FilterArg,
        #[arg(long, value_enum, default_value_t = WrapArg::Repeat)]
        wrap: WrapArg,
    },
    /// Remove one iChannel binding.
    Remove { pass: String, channel: u8 },
}

#[derive(Debug, Args)]
struct DocsArgs {
    /// agent, project, manifest, passes, buffers, channels, state, or preview.
    #[arg(default_value = "agent")]
    topic: String,
    /// Print the exact JSON Schema for ShaderToy.toml (manifest topic only).
    #[arg(long)]
    schema: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum TemplateArg {
    #[default]
    Minimal,
    Multipass,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum PassKindArg {
    #[default]
    Buffer,
    Cubemap,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputKindArg {
    Pass,
    Texture,
    Keyboard,
    Music,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum FilterArg {
    Mipmap,
    #[default]
    Linear,
    Nearest,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum WrapArg {
    Clamp,
    #[default]
    Repeat,
}

pub fn run() -> ExitCode {
    let args = std::env::args_os().collect::<Vec<_>>();
    let json_requested = args.iter().skip(1).any(|arg| arg == "--json");
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = if error.use_stderr() { 2 } else { 0 };
            if json_requested && error.use_stderr() {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "ok": false,
                        "error": error.to_string(),
                        "kind": format!("{:?}", error.kind()),
                    }))
                    .expect("clap error JSON serialization cannot fail")
                );
            } else {
                let _ = error.print();
            }
            return ExitCode::from(exit_code);
        }
    };
    let json_mode = cli.json;

    let result = dispatch(cli.command, json_mode);
    match result {
        Ok(Some(output)) => {
            emit(output, json_mode);
            ExitCode::SUCCESS
        }
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            if json_mode {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "ok": false,
                        "error": format!("{error:#}"),
                    }))
                    .expect("error JSON serialization cannot fail")
                );
            } else {
                eprintln!("error: {error:#}");
            }
            ExitCode::from(1)
        }
    }
}

fn dispatch(command: Command, json_mode: bool) -> Result<Option<Output>> {
    let output = match command {
        Command::New(args) => ops::new_project(&args.path, args.template.into())?,
        Command::Init(args) => ops::init_project(&args.path, args.template.into())?,
        Command::Import(args) => ops::import_project(&args.source, args.output.as_deref())?,
        Command::Check(args) => ops::check_project(&args.resolved())?,
        Command::Build(args) => {
            let project = args
                .project
                .clone()
                .or_else(|| args.path.clone())
                .unwrap_or_else(|| PathBuf::from("."));
            ops::build_project(&project, args.output.as_deref())?
        }
        Command::Render(args) => ops::render_project(&RenderOptions {
            project: args.project,
            output: args.output,
            pass: args.pass,
            width: args.width,
            height: args.height,
            fps: args.fps,
            frame: args.frame,
            time: args.time,
            state: args.state,
            set_buffers: args.set_buffers,
        })?,
        Command::Preview(args) => {
            preview::run(
                PreviewConfig {
                    project: args.project,
                    host: args.host,
                    port: args.port,
                    open: args.open,
                    no_open: args.no_open,
                    token: args.token,
                    preserve_reload_state: !args.reset_on_reload,
                },
                json_mode,
            )?;
            return Ok(None);
        }
        Command::Inspect(args) => match args.command {
            None => ops::inspect_project(&args.project, InspectMode::Summary)?,
            Some(InspectCommand::Graph) => ops::inspect_project(&args.project, InspectMode::Graph)?,
            Some(InspectCommand::Pass { name }) => {
                ops::inspect_project(&args.project, InspectMode::Pass(name))?
            }
            Some(InspectCommand::Channels { name }) => {
                ops::inspect_project(&args.project, InspectMode::Channels(name))?
            }
            Some(InspectCommand::State { path }) => ops::inspect_state(&path)?,
        },
        Command::State(args) => match args.command {
            StateCommand::Capture(args) => ops::capture_state(
                &args.project,
                &args.output,
                args.width,
                args.height,
                args.fps,
                args.frame,
                args.time,
            )?,
            StateCommand::Inspect { path } => ops::inspect_state(&path)?,
            StateCommand::Set(args) => {
                ops::set_state_buffers(&args.input, &args.output, &args.assignments)?
            }
        },
        Command::Pass(args) => match args.command {
            PassCommand::Add { name, kind, source } => {
                ops::add_pass(&args.project, &name, kind.into(), source.as_deref())?
            }
            PassCommand::Remove { name, force } => ops::remove_pass(&args.project, &name, force)?,
        },
        Command::Channel(args) => match args.command {
            ChannelCommand::Set {
                pass,
                channel,
                source,
                kind,
                previous,
                filter,
                wrap,
            } => ops::set_channel(
                &args.project,
                &ChannelSetOptions {
                    pass,
                    channel,
                    source,
                    kind: kind.map(Into::into),
                    previous,
                    filter: filter.into(),
                    wrap: wrap.into(),
                },
            )?,
            ChannelCommand::Remove { pass, channel } => {
                ops::remove_channel(&args.project, &pass, channel)?
            }
        },
        Command::Docs(args) => {
            let text = docs::topic(&args.topic, args.schema)?;
            Output {
                human: text.clone(),
                json: json!({
                    "ok": true,
                    "topic": args.topic,
                    "schema": args.schema,
                    "text": text,
                }),
            }
        }
    };
    Ok(Some(output))
}

fn emit(output: Output, json_mode: bool) {
    if json_mode {
        println!(
            "{}",
            serde_json::to_string(&output.json)
                .expect("command output JSON serialization cannot fail")
        );
    } else {
        println!("{}", output.human);
    }
}
