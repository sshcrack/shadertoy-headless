use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ShaderToyEntry {
    #[serde(default)]
    pub info: ShaderInfo,
    #[serde(default)]
    pub renderpass: Vec<RenderPass>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ShaderInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenderPass {
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub inputs: Vec<ShaderInput>,
    #[serde(default)]
    pub outputs: Vec<ShaderOutput>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShaderInput {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub channel: u8,
    #[serde(default)]
    pub filepath: String,
    #[serde(default)]
    pub sampler: Sampler,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Sampler {
    #[serde(default = "default_filter")]
    pub filter: String,
    #[serde(default = "default_wrap")]
    pub wrap: String,
    #[serde(default)]
    pub vflip: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShaderOutput {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub channel: Option<u8>,
}

fn default_filter() -> String {
    "linear".into()
}

fn default_wrap() -> String {
    "repeat".into()
}
