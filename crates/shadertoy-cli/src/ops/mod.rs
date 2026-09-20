mod images;
mod importer;
mod inspect;
mod mutate;
mod profile;
mod project;
mod regression;
mod render;
mod replay;
mod state;

use crate::manifest::{LoadedManifest, PassKind};
use crate::project::{build_native_project, ensure_source_files_exist};
use crate::scaffold::{Template, create_project};
use crate::state::StateFile;
use ::image::ImageReader;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use shadertoy::{HeadlessContext, RgbImage, Runtime};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub use images::rgb_png_bytes;
pub use importer::import_project;
pub use inspect::{inspect_buffer, inspect_project};
pub use mutate::{ChannelSetOptions, add_pass, remove_channel, remove_pass, set_channel};
pub use profile::profile_project;
pub use project::{build_project, check_project, init_project, new_project};
pub use regression::test_project;
pub use render::{render_frames_project, render_project};
pub use replay::replay_project;
pub use state::{capture_state, inspect_state, set_state_buffers};

use images::{BufferOverride, flip_rgba_rows, load_overrides, save_rgb_png, split_assignment};
use render::{render_from_zero, resolve_target_frame, validate_dimensions, validate_fps};

#[derive(Debug)]
pub struct Output {
    pub human: String,
    pub json: Value,
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub project: PathBuf,
    pub output: Option<PathBuf>,
    pub pass: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub frame: Option<i32>,
    pub time: Option<f32>,
    pub state: Option<PathBuf>,
    pub set_buffers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReplayOptions {
    pub project: PathBuf,
    pub recording: PathBuf,
    pub output: Option<PathBuf>,
    pub pass: Option<String>,
    pub frame: Option<u64>,
    pub allow_project_changes: bool,
}

#[derive(Debug, Clone)]
pub struct TestOptions {
    pub project: PathBuf,
    pub update: bool,
    pub filter: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProfileOptions {
    pub project: PathBuf,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub frame: Option<i32>,
    pub time: Option<f32>,
    pub warmup: u32,
    pub samples: u32,
}

#[derive(Debug, Clone)]
pub struct RenderFramesOptions {
    pub project: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub contact_sheet: Option<PathBuf>,
    pub columns: Option<u32>,
    pub pass: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub frames: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
pub enum InspectVisualization {
    Auto,
    Rgb,
    Signed,
    Magnitude,
}

#[derive(Debug, Clone)]
pub struct InspectBufferOptions {
    pub project: PathBuf,
    pub pass: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub frame: Option<i32>,
    pub time: Option<f32>,
    pub pixel: Option<(u32, u32)>,
    pub output: Option<PathBuf>,
    pub visualization: InspectVisualization,
}

#[derive(Debug, Clone)]
pub enum InspectMode {
    Summary,
    Graph,
    Pass(String),
    Channels(String),
}
