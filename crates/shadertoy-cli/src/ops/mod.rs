mod images;
mod importer;
mod inspect;
mod mutate;
mod project;
mod render;
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
pub use inspect::inspect_project;
pub use mutate::{ChannelSetOptions, add_pass, remove_channel, remove_pass, set_channel};
pub use project::{build_project, check_project, init_project, new_project};
pub use render::{render_frames_project, render_project};
pub use state::{capture_state, inspect_state, set_state_buffers};

use images::{flip_rgba_rows, load_overrides, save_rgb_png, split_assignment};
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

#[derive(Debug, Clone)]
pub enum InspectMode {
    Summary,
    Graph,
    Pass(String),
    Channels(String),
}
