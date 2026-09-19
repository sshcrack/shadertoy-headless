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
    #[serde(default, rename = "asset", skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<Asset>,
    #[serde(rename = "pass")]
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSection {
    /// Human-readable project name.
    pub name: String,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PassKind {
    Image,
    Buffer,
    Cubemap,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pass {
    /// Unique pass name. Names are used by inspect/render/debug commands.
    pub name: String,
    pub kind: PassKind,
    /// GLSL source path relative to the project root.
    pub source: String,
    #[serde(default, rename = "input", skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<Input>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InputKind {
    Pass,
    Texture,
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
    #[serde(default)]
    pub frame: FrameRef,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default)]
    pub wrap: Wrap,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AssetKind {
    Texture,
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
            project: ProjectSection { name: name.into() },
            render: RenderSection::default(),
            assets: Vec::new(),
            passes: vec![Pass {
                name: "image".into(),
                kind: PassKind::Image,
                source: "shaders/image.frag".into(),
                inputs: Vec::new(),
            }],
        }
    }

    pub fn multipass(name: impl Into<String>) -> Self {
        Self {
            format: FORMAT_VERSION,
            project: ProjectSection { name: name.into() },
            render: RenderSection::default(),
            assets: Vec::new(),
            passes: vec![
                Pass {
                    name: "buffer-a".into(),
                    kind: PassKind::Buffer,
                    source: "shaders/buffer-a.frag".into(),
                    inputs: vec![Input {
                        channel: 0,
                        source: "buffer-a".into(),
                        kind: Some(InputKind::Pass),
                        frame: FrameRef::Previous,
                        filter: Filter::Linear,
                        wrap: Wrap::Clamp,
                    }],
                },
                Pass {
                    name: "image".into(),
                    kind: PassKind::Image,
                    source: "shaders/image.frag".into(),
                    inputs: vec![Input {
                        channel: 0,
                        source: "buffer-a".into(),
                        kind: Some(InputKind::Pass),
                        frame: FrameRef::Current,
                        filter: Filter::Linear,
                        wrap: Wrap::Clamp,
                    }],
                },
            ],
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
            if pass.kind == PassKind::Image {
                image_count += 1;
            }
            pass_kinds.insert(pass.name.as_str(), pass.kind);
        }

        let mut asset_names = HashSet::new();
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
        }

        if image_count != 1 {
            bail!("project must contain exactly one image pass (found {image_count})");
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
                if input.frame == FrameRef::Previous && kind != InputKind::Pass {
                    bail!(
                        "pass '{}' channel {} uses previous frame on a non-pass input",
                        pass.name,
                        input.channel
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
                }
                if kind == InputKind::Texture && !asset_names.contains(input.source.as_str()) {
                    bail!(
                        "pass '{}' channel {} references unknown texture '{}'",
                        pass.name,
                        input.channel,
                        input.source
                    );
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
        if self.assets.iter().any(|asset| asset.name == input.source) {
            return Ok(InputKind::Texture);
        }
        if self.passes.iter().any(|pass| pass.name == input.source) {
            return Ok(InputKind::Pass);
        }
        bail!(
            "cannot infer input kind for unknown source '{}'",
            input.source
        )
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
