use anyhow::{Context, Result, bail};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const MANIFEST_NAME: &str = "ShaderToy.toml";
pub const FORMAT_VERSION: u32 = 1;
pub const MAX_RENDER_DIMENSION: u32 = 16384;
pub const MAX_RENDER_FPS: f32 = 1000.0;
const RESERVED_INPUT_NAMES: [&str; 2] = ["keyboard", "music"];

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Project manifest format version. Version 1 is the only supported value.
    pub format: u32,
    pub project: ProjectSection,
    #[serde(default)]
    pub render: RenderSection,
    #[serde(default)]
    pub shader: ShaderSection,
    #[serde(default, rename = "asset", skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<Asset>,
    #[serde(rename = "pass")]
    pub passes: Vec<Pass>,
    #[serde(default, rename = "test", skip_serializing_if = "Vec::is_empty")]
    pub tests: Vec<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSection {
    /// Human-readable project name.
    pub name: String,
    /// Original ShaderToy author, when this project was imported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Original ShaderToy description, when this project was imported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Canonical source URL for imported projects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// ShaderToy shader id for imported projects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct RenderSection {
    /// Default render/preview width in pixels.
    pub width: u32,
    /// Default render/preview height in pixels.
    pub height: u32,
    /// Fixed timestep frame rate used by deterministic rendering.
    pub fps: f32,
    /// Representative time used by `shadertoy render` when no frame/time is supplied.
    pub preview_time: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ShaderSection {
    /// Project-relative directories searched after the including file's directory for quoted #include paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include_dirs: Vec<String>,
}

impl Default for RenderSection {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 60.0,
            preview_time: 1.0,
        }
    }
}

fn default_test_tolerance() -> f32 {
    0.002
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TestCase {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default = "default_test_tolerance")]
    pub tolerance: f32,
    #[serde(default)]
    pub assert_no_nan: bool,
    #[serde(default)]
    pub assert_no_inf: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_range: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PassKind {
    Image,
    Buffer,
    Cubemap,
    Compute,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RenderFormat {
    R32f,
    Rg32f,
    Rgba16f,
    #[default]
    Rgba32f,
}

fn default_iterations() -> u32 {
    1
}

fn is_default_iterations(value: &u32) -> bool {
    *value == 1
}

fn is_default_render_format(value: &RenderFormat) -> bool {
    *value == RenderFormat::Rgba32f
}

fn is_zero_u8(value: &u8) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StorageBuffer {
    /// Shader storage binding index used by an explicit std430 binding.
    pub binding: u32,
    /// Shared resource name. Bindings with the same name share persistent GPU storage across passes.
    pub name: String,
    /// Persistent storage size in bytes.
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pass {
    /// Unique pass name. Names are used by inspect/render/debug commands.
    pub name: String,
    pub kind: PassKind,
    /// GLSL source path relative to the project root.
    pub source: String,
    /// Optional fixed width for a buffer/compute pass. Must be paired with height.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Optional fixed height for a buffer/compute pass. Must be paired with width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Floating-point render-target format for buffer/compute outputs.
    #[serde(default, skip_serializing_if = "is_default_render_format")]
    pub format: RenderFormat,
    /// Additional render targets beyond output 0. Indices are 1-based after the primary format.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_outputs: Vec<RenderFormat>,
    /// Number of compute dispatches per rendered frame. iIteration is 0-based.
    #[serde(
        default = "default_iterations",
        skip_serializing_if = "is_default_iterations"
    )]
    pub iterations: u32,
    /// Compute local workgroup dimensions. Defaults to [8, 8, 1]; Z must be 1 for the 2D mainCompute entrypoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_size: Option<[u32; 3]>,
    /// Persistent shader-storage buffers bound for this pass.
    #[serde(default, rename = "storage", skip_serializing_if = "Vec::is_empty")]
    pub storage: Vec<StorageBuffer>,
    #[serde(default, rename = "input", skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<Input>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InputKind {
    Pass,
    Texture,
    Cubemap,
    Volume,
    Keyboard,
    Music,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FrameRef {
    #[default]
    Current,
    Previous,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Filter {
    Mipmap,
    #[default]
    Linear,
    Nearest,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Wrap {
    Clamp,
    #[default]
    Repeat,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// ShaderToy iChannel index. Must be between 0 and 3.
    pub channel: u8,
    /// Pass/asset name, or the reserved names "keyboard" and "music".
    pub source: String,
    /// Optional explicit source kind. Normally inferred from the source name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<InputKind>,
    /// Render-target index when consuming another pass. Output 0 is the primary target.
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub output: u8,
    #[serde(default)]
    pub frame: FrameRef,
    /// Per-input texture interpolation/minification mode.
    /// nearest = exact texel sampling, linear = bilinear interpolation,
    /// mipmap = trilinear minification with linear magnification.
    #[serde(default)]
    pub filter: Filter,
    /// Per-input texture coordinate addressing mode.
    #[serde(default)]
    pub wrap: Wrap,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AssetKind {
    Texture,
    Cubemap,
    Volume,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    /// Unique asset name used by pass inputs.
    pub name: String,
    pub kind: AssetKind,
    /// Asset file path relative to the project root.
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct LoadedManifest {
    pub root: PathBuf,
    pub manifest: Manifest,
}

impl Manifest {
    pub fn minimal(name: impl Into<String>) -> Self {
        Self {
            format: FORMAT_VERSION,
            project: ProjectSection {
                name: name.into(),
                author: None,
                description: None,
                source_url: None,
                source_id: None,
            },
            render: RenderSection::default(),
            shader: ShaderSection::default(),
            assets: Vec::new(),
            passes: vec![Pass {
                name: "image".into(),
                kind: PassKind::Image,
                source: "shaders/image.frag".into(),
                width: None,
                height: None,
                format: RenderFormat::default(),
                extra_outputs: Vec::new(),
                iterations: 1,
                local_size: None,
                storage: Vec::new(),
                inputs: Vec::new(),
            }],
            tests: Vec::new(),
        }
    }

    pub fn multipass(name: impl Into<String>) -> Self {
        Self {
            format: FORMAT_VERSION,
            project: ProjectSection {
                name: name.into(),
                author: None,
                description: None,
                source_url: None,
                source_id: None,
            },
            render: RenderSection::default(),
            shader: ShaderSection::default(),
            assets: Vec::new(),
            passes: vec![
                Pass {
                    name: "buffer-a".into(),
                    kind: PassKind::Buffer,
                    source: "shaders/buffer-a.frag".into(),
                    width: None,
                    height: None,
                    format: RenderFormat::default(),
                    extra_outputs: Vec::new(),
                    iterations: 1,
                    local_size: None,
                    storage: Vec::new(),
                    inputs: vec![Input {
                        channel: 0,
                        source: "buffer-a".into(),
                        kind: Some(InputKind::Pass),
                        output: 0,
                        frame: FrameRef::Previous,
                        filter: Filter::Linear,
                        wrap: Wrap::Clamp,
                    }],
                },
                Pass {
                    name: "image".into(),
                    kind: PassKind::Image,
                    source: "shaders/image.frag".into(),
                    width: None,
                    height: None,
                    format: RenderFormat::default(),
                    extra_outputs: Vec::new(),
                    iterations: 1,
                    local_size: None,
                    storage: Vec::new(),
                    inputs: vec![Input {
                        channel: 0,
                        source: "buffer-a".into(),
                        kind: Some(InputKind::Pass),
                        output: 0,
                        frame: FrameRef::Current,
                        filter: Filter::Linear,
                        wrap: Wrap::Clamp,
                    }],
                },
            ],
            tests: Vec::new(),
        }
    }

    pub fn validate_structure(&self) -> Result<()> {
        if self.format != FORMAT_VERSION {
            bail!(
                "unsupported ShaderToy.toml format {}; supported format is {}",
                self.format,
                FORMAT_VERSION
            );
        }
        if self.project.name.trim().is_empty() {
            bail!("project.name must not be empty");
        }
        if self.render.width == 0 || self.render.height == 0 {
            bail!("render width and height must be positive");
        }
        if self.render.width > MAX_RENDER_DIMENSION || self.render.height > MAX_RENDER_DIMENSION {
            bail!("render dimensions exceed the {MAX_RENDER_DIMENSION} pixel safety limit");
        }
        if !self.render.fps.is_finite()
            || self.render.fps <= 0.0
            || self.render.fps > MAX_RENDER_FPS
        {
            bail!("render.fps must be finite and in the range (0, {MAX_RENDER_FPS}]");
        }
        if !self.render.preview_time.is_finite() || self.render.preview_time < 0.0 {
            bail!("render.preview_time must be a finite non-negative number");
        }
        for include_dir in &self.shader.include_dirs {
            validate_project_relative_path(include_dir, "shader.include_dirs entry")?;
        }
        if self.passes.is_empty() {
            bail!("project must contain at least one [[pass]]");
        }

        let mut names = HashSet::new();
        let mut pass_kinds = HashMap::new();
        let mut image_count = 0usize;
        for pass in &self.passes {
            if pass.name.trim().is_empty() {
                bail!("pass name must not be empty");
            }
            if is_reserved_input_name(&pass.name) {
                bail!("pass name '{}' is reserved for a built-in input", pass.name);
            }
            validate_project_relative_path(
                &pass.source,
                &format!("source for pass '{}'", pass.name),
            )?;
            if !names.insert(pass.name.as_str()) {
                bail!("duplicate pass/asset name '{}'", pass.name);
            }
            match (pass.width, pass.height) {
                (None, None) => {
                    if pass.kind == PassKind::Compute {
                        bail!("compute pass '{}' must specify width and height", pass.name);
                    }
                }
                (Some(width), Some(height)) => {
                    if pass.kind != PassKind::Buffer && pass.kind != PassKind::Compute {
                        bail!(
                            "fixed width/height are only supported for buffer/compute passes ('{}')",
                            pass.name
                        );
                    }
                    if width == 0 || height == 0 {
                        bail!("fixed pass dimensions must be positive ('{}')", pass.name);
                    }
                    if width > MAX_RENDER_DIMENSION || height > MAX_RENDER_DIMENSION {
                        bail!(
                            "fixed pass dimensions for '{}' exceed the {} pixel safety limit",
                            pass.name,
                            MAX_RENDER_DIMENSION
                        );
                    }
                }
                _ => bail!(
                    "pass '{}' must specify both width and height or neither",
                    pass.name
                ),
            }
            if !matches!(pass.kind, PassKind::Buffer | PassKind::Compute)
                && pass.format != RenderFormat::Rgba32f
            {
                bail!(
                    "pass '{}' can only set format on buffer/compute passes",
                    pass.name
                );
            }
            if !pass.extra_outputs.is_empty()
                && !matches!(pass.kind, PassKind::Buffer | PassKind::Compute)
            {
                bail!(
                    "pass '{}' extra_outputs are only valid for buffer/compute passes",
                    pass.name
                );
            }
            if pass.extra_outputs.len() > 7 {
                bail!("pass '{}' may expose at most 8 render targets", pass.name);
            }
            if pass.iterations == 0 || pass.iterations > 4096 {
                bail!("pass '{}' iterations must be in 1..=4096", pass.name);
            }
            if pass.kind != PassKind::Compute && pass.iterations != 1 {
                bail!(
                    "pass '{}' iterations are currently supported only for compute passes",
                    pass.name
                );
            }
            if let Some([x, y, z]) = pass.local_size {
                if pass.kind != PassKind::Compute {
                    bail!(
                        "pass '{}' local_size is only valid for compute passes",
                        pass.name
                    );
                }
                if x == 0 || y == 0 || z == 0 || u64::from(x) * u64::from(y) * u64::from(z) > 1024 {
                    bail!(
                        "pass '{}' local_size must be positive and at most 1024 total invocations",
                        pass.name
                    );
                }
                if z != 1 {
                    bail!(
                        "pass '{}' local_size z must be 1 because mainCompute receives 2D coordinates",
                        pass.name
                    );
                }
            }
            let mut storage_bindings = HashSet::new();
            for storage in &pass.storage {
                if storage.name.trim().is_empty() {
                    bail!(
                        "pass '{}' has a storage buffer with an empty name",
                        pass.name
                    );
                }
                if storage.size == 0 {
                    bail!(
                        "pass '{}' storage buffer '{}' must have a positive size",
                        pass.name,
                        storage.name
                    );
                }
                if !storage_bindings.insert(storage.binding) {
                    bail!(
                        "pass '{}' binds storage slot {} more than once",
                        pass.name,
                        storage.binding
                    );
                }
            }
            if pass.kind == PassKind::Image {
                image_count += 1;
            }
            pass_kinds.insert(pass.name.as_str(), pass.kind);
        }

        let mut asset_names = HashSet::new();
        let mut asset_kinds = HashMap::new();
        for asset in &self.assets {
            if asset.name.trim().is_empty() {
                bail!("asset name must not be empty");
            }
            if is_reserved_input_name(&asset.name) {
                bail!(
                    "asset name '{}' is reserved for a built-in input",
                    asset.name
                );
            }
            validate_project_relative_path(
                &asset.path,
                &format!("path for asset '{}'", asset.name),
            )?;
            if !names.insert(asset.name.as_str()) || !asset_names.insert(asset.name.as_str()) {
                bail!("duplicate pass/asset name '{}'", asset.name);
            }
            asset_kinds.insert(asset.name.as_str(), asset.kind);
        }

        if image_count != 1 {
            bail!("project must contain exactly one image pass (found {image_count})");
        }

        let mut storage_sizes = HashMap::new();
        for pass in &self.passes {
            for storage in &pass.storage {
                if let Some(existing) = storage_sizes.insert(storage.name.as_str(), storage.size)
                    && existing != storage.size
                {
                    bail!(
                        "storage buffer '{}' uses inconsistent sizes ({} vs {})",
                        storage.name,
                        existing,
                        storage.size
                    );
                }
            }
        }

        for pass in &self.passes {
            let mut channels = HashSet::new();
            for input in &pass.inputs {
                if input.channel > 3 {
                    bail!(
                        "pass '{}' uses invalid channel {}; valid channels are 0..3",
                        pass.name,
                        input.channel
                    );
                }
                if !channels.insert(input.channel) {
                    bail!(
                        "pass '{}' assigns iChannel{} more than once",
                        pass.name,
                        input.channel
                    );
                }
                let kind = self.infer_input_kind(input)?;
                if kind != InputKind::Pass && input.output != 0 {
                    bail!(
                        "pass '{}' channel {} selects output {} on a non-pass input",
                        pass.name,
                        input.channel,
                        input.output
                    );
                }
                if input.frame == FrameRef::Previous && kind != InputKind::Pass {
                    bail!(
                        "pass '{}' channel {} uses previous frame on a non-pass input",
                        pass.name,
                        input.channel
                    );
                }
                if input.frame == FrameRef::Previous && input.output != 0 {
                    bail!(
                        "pass '{}' channel {} uses previous-frame feedback from output {}; only output 0 is resumable",
                        pass.name,
                        input.channel,
                        input.output
                    );
                }
                if kind == InputKind::Pass {
                    let Some(source_kind) = pass_kinds.get(input.source.as_str()) else {
                        bail!(
                            "pass '{}' channel {} references unknown pass '{}'",
                            pass.name,
                            input.channel,
                            input.source
                        );
                    };
                    if input.frame == FrameRef::Previous && *source_kind == PassKind::Image {
                        bail!("the final image pass cannot be a previous-frame source");
                    }
                    let source_pass = self
                        .passes
                        .iter()
                        .find(|candidate| candidate.name == input.source)
                        .expect("pass kind map and pass list stay in sync");
                    if usize::from(input.output) > source_pass.extra_outputs.len() {
                        bail!(
                            "pass '{}' channel {} selects output {} from '{}', which exposes outputs 0..{}",
                            pass.name,
                            input.channel,
                            input.output,
                            input.source,
                            source_pass.extra_outputs.len()
                        );
                    }
                }
                if matches!(
                    kind,
                    InputKind::Texture | InputKind::Cubemap | InputKind::Volume
                ) {
                    let Some(asset_kind) = asset_kinds.get(input.source.as_str()) else {
                        bail!(
                            "pass '{}' channel {} references unknown asset '{}'",
                            pass.name,
                            input.channel,
                            input.source
                        );
                    };
                    let expected = match kind {
                        InputKind::Texture => AssetKind::Texture,
                        InputKind::Cubemap => AssetKind::Cubemap,
                        InputKind::Volume => AssetKind::Volume,
                        _ => unreachable!(),
                    };
                    if *asset_kind != expected {
                        bail!(
                            "pass '{}' channel {} expects {:?} asset '{}' but it is {:?}",
                            pass.name,
                            input.channel,
                            expected,
                            input.source,
                            asset_kind
                        );
                    }
                }
                if kind == InputKind::Keyboard && input.source != "keyboard" {
                    bail!(
                        "pass '{}' channel {} uses keyboard input with source '{}'; use source 'keyboard'",
                        pass.name,
                        input.channel,
                        input.source
                    );
                }
                if kind == InputKind::Music && input.source != "music" {
                    bail!(
                        "pass '{}' channel {} uses music input with source '{}'; use source 'music'",
                        pass.name,
                        input.channel,
                        input.source
                    );
                }
            }
        }
        let mut test_names = HashSet::new();
        for test in &self.tests {
            if test.name.trim().is_empty() {
                bail!("test name must not be empty");
            }
            if !test_names.insert(test.name.as_str()) {
                bail!("duplicate test name '{}'", test.name);
            }
            if test.frame.is_some() && test.time.is_some() {
                bail!(
                    "test '{}' must use either frame or time, not both",
                    test.name
                );
            }
            if test.frame.is_some_and(|frame| frame < 0) {
                bail!("test '{}' frame must be non-negative", test.name);
            }
            if test
                .time
                .is_some_and(|time| !time.is_finite() || time < 0.0)
            {
                bail!("test '{}' time must be finite and non-negative", test.name);
            }
            match (test.width, test.height) {
                (None, None) => {}
                (Some(width), Some(height))
                    if width > 0
                        && height > 0
                        && width <= MAX_RENDER_DIMENSION
                        && height <= MAX_RENDER_DIMENSION => {}
                (Some(_), Some(_)) => bail!(
                    "test '{}' dimensions must be positive and at most {}",
                    test.name,
                    MAX_RENDER_DIMENSION
                ),
                _ => bail!(
                    "test '{}' must specify both width and height or neither",
                    test.name
                ),
            }
            if !test.tolerance.is_finite() || !(0.0..=1.0).contains(&test.tolerance) {
                bail!(
                    "test '{}' tolerance must be finite and in [0, 1]",
                    test.name
                );
            }
            if let Some(reference) = &test.reference {
                validate_project_relative_path(
                    reference,
                    &format!("reference for test '{}'", test.name),
                )?;
            }
            if let Some(pass) = &test.pass
                && !self.passes.iter().any(|candidate| candidate.name == *pass)
            {
                bail!("test '{}' references unknown pass '{}'", test.name, pass);
            }
            if let Some([min, max]) = test.mean_range
                && (!min.is_finite() || !max.is_finite() || min > max)
            {
                bail!(
                    "test '{}' mean_range must contain finite [min, max]",
                    test.name
                );
            }
            if test.reference.is_none()
                && !test.assert_no_nan
                && !test.assert_no_inf
                && test.mean_range.is_none()
            {
                bail!(
                    "test '{}' must define a reference or at least one numeric assertion",
                    test.name
                );
            }
        }

        Ok(())
    }

    pub fn infer_input_kind(&self, input: &Input) -> Result<InputKind> {
        if let Some(kind) = input.kind {
            return Ok(kind);
        }
        if input.source == "keyboard" {
            return Ok(InputKind::Keyboard);
        }
        if input.source == "music" {
            return Ok(InputKind::Music);
        }
        if let Some(asset) = self.assets.iter().find(|asset| asset.name == input.source) {
            return Ok(match asset.kind {
                AssetKind::Texture => InputKind::Texture,
                AssetKind::Cubemap => InputKind::Cubemap,
                AssetKind::Volume => InputKind::Volume,
            });
        }
        if self.passes.iter().any(|pass| pass.name == input.source) {
            return Ok(InputKind::Pass);
        }
        bail!(
            "cannot infer input kind for unknown source '{}'",
            input.source
        )
    }

    pub fn pass_dimensions(
        &self,
        pass: &Pass,
        output_width: u32,
        output_height: u32,
    ) -> (u32, u32) {
        match (pass.width, pass.height) {
            (Some(width), Some(height)) => (width, height),
            _ => (output_width, output_height),
        }
    }

    pub fn final_pass(&self) -> &Pass {
        self.passes
            .iter()
            .find(|pass| pass.kind == PassKind::Image)
            .expect("validated manifest always has one image pass")
    }

    pub fn to_pretty_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("failed to serialize ShaderToy.toml")
    }
}

impl LoadedManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let root = find_project_root(path.as_ref())?;
        let manifest_path = root.join(MANIFEST_NAME);
        let source = fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        let manifest: Manifest = toml::from_str(&source)
            .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
        manifest.validate_structure()?;
        crate::project_schema::refresh_existing(&root)?;
        Ok(Self { root, manifest })
    }
}

pub fn find_project_root(start: &Path) -> Result<PathBuf> {
    let start = if start.is_file() {
        start.parent().unwrap_or(start)
    } else {
        start
    };
    let absolute = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir()?.join(start)
    };

    for candidate in absolute.ancestors() {
        if candidate.join(MANIFEST_NAME).is_file() {
            return Ok(candidate.to_path_buf());
        }
    }
    bail!(
        "could not find {MANIFEST_NAME} from {}; run 'shadertoy new' or 'shadertoy init'",
        absolute.display()
    )
}

pub fn schema_json() -> Result<String> {
    serde_json::to_string_pretty(&schema_for!(Manifest))
        .context("failed to serialize manifest schema")
}

pub fn validate_project_relative_path(value: &str, label: &str) -> Result<()> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} must not be empty");
    }

    // Manifests use portable project-relative paths. Normalize separators before
    // checking so a Windows traversal is rejected even when inspected on Unix.
    let normalized = value.replace('\\', "/");
    let bytes = normalized.as_bytes();
    let windows_drive = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    let path = Path::new(&normalized);
    if windows_drive
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("{label} must stay relative to the project root");
    }
    Ok(())
}

fn is_reserved_input_name(value: &str) -> bool {
    RESERVED_INPUT_NAMES.contains(&value)
}

#[cfg(test)]
mod tests;
