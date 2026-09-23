use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

const RESERVED_UNIFORMS: &[&str] = &[
    "iResolution",
    "iTime",
    "iTimeDelta",
    "iFrameRate",
    "iFrame",
    "iIteration",
    "iMouse",
    "iDate",
    "iChannelResolution",
    "iChannelTime",
    "iChannel0",
    "iChannel1",
    "iChannel2",
    "iChannel3",
    "iChannel4",
    "iChannel5",
    "iChannel6",
    "iChannel7",
    "iChannel8",
    "iChannel9",
    "iChannel10",
    "iChannel11",
    "iChannel12",
    "iChannel13",
    "iChannel14",
    "iChannel15",
    "iMusicBands",
    "iMusicHits",
    "iMusicBeat",
    "iMusicStereo",
    "iMusicStructure",
    "iMusicMeta",
    "iAudioLoudness",
    "iAudioBass",
    "iAudioMid",
    "iAudioTreble",
    "iAudioOnset",
    "iAudioKick",
    "iAudioSnare",
    "iAudioHihat",
    "iAudioBpm",
    "iAudioBeatPhase",
    "iAudioBeatConfidence",
    "iAudioBeatStrength",
    "iAudioStereoWidth",
    "iAudioStereoBalance",
    "iAudioStereoCorrelation",
    "iAudioEnergyTrend",
    "iAudioDrop",
    "iAudioSectionChange",
    "iAudioSpectralCentroid",
    "iAudioSpectralFlux",
    "iAudioAvailable",
    "iAudioSilence",
    "iAudioSampleRate",
    "iOutput",
    "iOutput1",
    "iOutput2",
    "iOutput3",
    "iOutput4",
    "iOutput5",
    "iOutput6",
    "iOutput7",
];

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum UniformDefinition {
    Float {
        name: String,
        default: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<f32>,
    },
    Int {
        name: String,
        default: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<i32>,
    },
    Bool {
        name: String,
        default: bool,
    },
    Vec2 {
        name: String,
        default: [f32; 2],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<[f32; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<[f32; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<f32>,
    },
    Vec3 {
        name: String,
        default: [f32; 3],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<[f32; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<[f32; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<f32>,
    },
    Vec4 {
        name: String,
        default: [f32; 4],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<f32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub enum UniformValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

impl UniformDefinition {
    pub fn name(&self) -> &str {
        match self {
            Self::Float { name, .. }
            | Self::Int { name, .. }
            | Self::Bool { name, .. }
            | Self::Vec2 { name, .. }
            | Self::Vec3 { name, .. }
            | Self::Vec4 { name, .. } => name,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Float { .. } => "float",
            Self::Int { .. } => "int",
            Self::Bool { .. } => "bool",
            Self::Vec2 { .. } => "vec2",
            Self::Vec3 { .. } => "vec3",
            Self::Vec4 { .. } => "vec4",
        }
    }

    pub fn glsl_type(&self) -> &'static str {
        match self {
            Self::Float { .. } => "float",
            Self::Int { .. } => "int",
            Self::Bool { .. } => "bool",
            Self::Vec2 { .. } => "vec2",
            Self::Vec3 { .. } => "vec3",
            Self::Vec4 { .. } => "vec4",
        }
    }

    pub fn default_value(&self) -> UniformValue {
        match self {
            Self::Float { default, .. } => UniformValue::Float(*default),
            Self::Int { default, .. } => UniformValue::Int(*default),
            Self::Bool { default, .. } => UniformValue::Bool(*default),
            Self::Vec2 { default, .. } => UniformValue::Vec2(*default),
            Self::Vec3 { default, .. } => UniformValue::Vec3(*default),
            Self::Vec4 { default, .. } => UniformValue::Vec4(*default),
        }
    }

    pub fn parse_value(&self, raw: &str) -> Result<UniformValue> {
        let raw = raw.trim();
        let value = match self {
            Self::Float { .. } => UniformValue::Float(
                raw.parse::<f32>()
                    .map_err(|_| anyhow::anyhow!("expected a float"))?,
            ),
            Self::Int { .. } => UniformValue::Int(
                raw.parse::<i32>()
                    .map_err(|_| anyhow::anyhow!("expected an integer"))?,
            ),
            Self::Bool { .. } => UniformValue::Bool(match raw {
                "true" | "1" | "on" => true,
                "false" | "0" | "off" => false,
                _ => bail!("expected true/false"),
            }),
            Self::Vec2 { .. } => UniformValue::Vec2(parse_vec::<2>(raw)?),
            Self::Vec3 { .. } => UniformValue::Vec3(parse_vec::<3>(raw)?),
            Self::Vec4 { .. } => UniformValue::Vec4(parse_vec::<4>(raw)?),
        };
        self.validate_value(&value)?;
        Ok(value)
    }

    pub fn validate_definition(&self) -> Result<()> {
        let name = self.name();
        validate_identifier(name)?;
        if RESERVED_UNIFORMS.contains(&name) {
            bail!("custom uniform '{name}' conflicts with a built-in ShaderToy uniform");
        }
        self.validate_value(&self.default_value())?;
        match self {
            Self::Float { min, max, step, .. } => {
                validate_float_range(name, *min, *max, *step)?;
            }
            Self::Int { min, max, step, .. } => {
                if let (Some(min), Some(max)) = (min, max)
                    && min > max
                {
                    bail!("uniform '{name}' min must not exceed max");
                }
                if step.is_some_and(|step| step <= 0) {
                    bail!("uniform '{name}' step must be positive");
                }
            }
            Self::Vec2 { min, max, step, .. } => {
                validate_vec_range(
                    name,
                    min.as_ref().map(|v| v.as_slice()),
                    max.as_ref().map(|v| v.as_slice()),
                    *step,
                )?;
            }
            Self::Vec3 { min, max, step, .. } => {
                validate_vec_range(
                    name,
                    min.as_ref().map(|v| v.as_slice()),
                    max.as_ref().map(|v| v.as_slice()),
                    *step,
                )?;
            }
            Self::Vec4 { min, max, step, .. } => {
                validate_vec_range(
                    name,
                    min.as_ref().map(|v| v.as_slice()),
                    max.as_ref().map(|v| v.as_slice()),
                    *step,
                )?;
            }
            Self::Bool { .. } => {}
        }
        Ok(())
    }

    pub fn validate_value(&self, value: &UniformValue) -> Result<()> {
        let name = self.name();
        let in_float_range = |value: f32, min: Option<f32>, max: Option<f32>| -> Result<()> {
            if !value.is_finite() {
                bail!("uniform '{name}' value must be finite");
            }
            if min.is_some_and(|min| value < min) || max.is_some_and(|max| value > max) {
                bail!("uniform '{name}' value is outside its configured range");
            }
            Ok(())
        };
        match (self, value) {
            (Self::Float { min, max, .. }, UniformValue::Float(value)) => {
                in_float_range(*value, *min, *max)
            }
            (Self::Int { min, max, .. }, UniformValue::Int(value)) => {
                if min.is_some_and(|min| *value < min) || max.is_some_and(|max| *value > max) {
                    bail!("uniform '{name}' value is outside its configured range");
                }
                Ok(())
            }
            (Self::Bool { .. }, UniformValue::Bool(_)) => Ok(()),
            (Self::Vec2 { min, max, .. }, UniformValue::Vec2(value)) => {
                validate_vec_value(name, value, min.as_ref(), max.as_ref())
            }
            (Self::Vec3 { min, max, .. }, UniformValue::Vec3(value)) => {
                validate_vec_value(name, value, min.as_ref(), max.as_ref())
            }
            (Self::Vec4 { min, max, .. }, UniformValue::Vec4(value)) => {
                validate_vec_value(name, value, min.as_ref(), max.as_ref())
            }
            _ => bail!("uniform '{name}' value has the wrong type"),
        }
    }

    pub fn preview_min(&self) -> Option<UniformValue> {
        match self {
            Self::Float { min: Some(v), .. } => Some(UniformValue::Float(*v)),
            Self::Int { min: Some(v), .. } => Some(UniformValue::Int(*v)),
            Self::Vec2 { min: Some(v), .. } => Some(UniformValue::Vec2(*v)),
            Self::Vec3 { min: Some(v), .. } => Some(UniformValue::Vec3(*v)),
            Self::Vec4 { min: Some(v), .. } => Some(UniformValue::Vec4(*v)),
            _ => None,
        }
    }

    pub fn preview_max(&self) -> Option<UniformValue> {
        match self {
            Self::Float { max: Some(v), .. } => Some(UniformValue::Float(*v)),
            Self::Int { max: Some(v), .. } => Some(UniformValue::Int(*v)),
            Self::Vec2 { max: Some(v), .. } => Some(UniformValue::Vec2(*v)),
            Self::Vec3 { max: Some(v), .. } => Some(UniformValue::Vec3(*v)),
            Self::Vec4 { max: Some(v), .. } => Some(UniformValue::Vec4(*v)),
            _ => None,
        }
    }

    pub fn preview_step(&self) -> Option<f32> {
        match self {
            Self::Float { step, .. } => *step,
            Self::Int { step, .. } => step.map(|v| v as f32),
            Self::Vec2 { step, .. } | Self::Vec3 { step, .. } | Self::Vec4 { step, .. } => *step,
            Self::Bool { .. } => None,
        }
    }
}

pub fn validate_definitions(definitions: &[UniformDefinition]) -> Result<()> {
    let mut names = HashSet::new();
    for definition in definitions {
        definition.validate_definition()?;
        if !names.insert(definition.name()) {
            bail!("duplicate custom uniform '{}'", definition.name());
        }
    }
    Ok(())
}

pub fn declarations(definitions: &[UniformDefinition]) -> String {
    let mut result = String::new();
    for definition in definitions {
        result.push_str("uniform ");
        result.push_str(definition.glsl_type());
        result.push(' ');
        result.push_str(definition.name());
        result.push_str(";\n");
    }
    result
}

pub fn defaults(definitions: &[UniformDefinition]) -> BTreeMap<String, UniformValue> {
    definitions
        .iter()
        .map(|definition| (definition.name().to_string(), definition.default_value()))
        .collect()
}

pub fn parse_assignments(
    definitions: &[UniformDefinition],
    assignments: &[String],
) -> Result<BTreeMap<String, UniformValue>> {
    let mut result = defaults(definitions);
    for assignment in assignments {
        let Some((name, raw)) = assignment.split_once('=') else {
            bail!("uniform override '{assignment}' must use NAME=VALUE");
        };
        let definition = definitions
            .iter()
            .find(|definition| definition.name() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown custom uniform '{name}'"))?;
        let value = definition
            .parse_value(raw)
            .map_err(|error| anyhow::anyhow!("invalid value for uniform '{name}': {error:#}"))?;
        result.insert(name.to_string(), value);
    }
    Ok(result)
}

pub fn merge_values(
    definitions: &[UniformDefinition],
    overrides: &BTreeMap<String, UniformValue>,
) -> Result<BTreeMap<String, UniformValue>> {
    let mut values = defaults(definitions);
    for (name, value) in overrides {
        let definition = definitions
            .iter()
            .find(|definition| definition.name() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown custom uniform '{name}'"))?;
        definition.validate_value(value)?;
        values.insert(name.clone(), value.clone());
    }
    Ok(values)
}

fn parse_vec<const N: usize>(raw: &str) -> Result<[f32; N]> {
    let values = raw
        .split(',')
        .map(str::trim)
        .map(|value| value.parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow::anyhow!("expected {N} comma-separated floats"))?;
    if values.len() != N {
        bail!("expected {N} comma-separated floats");
    }
    values
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected {N} comma-separated floats"))
}

fn validate_identifier(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        bail!("custom uniform name must not be empty");
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        bail!("custom uniform '{name}' is not a valid GLSL identifier");
    }
    Ok(())
}

fn validate_float_range(
    name: &str,
    min: Option<f32>,
    max: Option<f32>,
    step: Option<f32>,
) -> Result<()> {
    for (label, value) in [("min", min), ("max", max), ("step", step)] {
        if value.is_some_and(|value| !value.is_finite()) {
            bail!("uniform '{name}' {label} must be finite");
        }
    }
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        bail!("uniform '{name}' min must not exceed max");
    }
    if step.is_some_and(|step| step <= 0.0) {
        bail!("uniform '{name}' step must be positive");
    }
    Ok(())
}

fn validate_vec_range(
    name: &str,
    min: Option<&[f32]>,
    max: Option<&[f32]>,
    step: Option<f32>,
) -> Result<()> {
    if step.is_some_and(|step| !step.is_finite() || step <= 0.0) {
        bail!("uniform '{name}' step must be finite and positive");
    }
    if let Some(min) = min
        && min.iter().any(|value| !value.is_finite())
    {
        bail!("uniform '{name}' min must contain only finite values");
    }
    if let Some(max) = max
        && max.iter().any(|value| !value.is_finite())
    {
        bail!("uniform '{name}' max must contain only finite values");
    }
    if let (Some(min), Some(max)) = (min, max)
        && min.iter().zip(max).any(|(min, max)| min > max)
    {
        bail!("uniform '{name}' min must not exceed max");
    }
    Ok(())
}

fn validate_vec_value<const N: usize>(
    name: &str,
    value: &[f32; N],
    min: Option<&[f32; N]>,
    max: Option<&[f32; N]>,
) -> Result<()> {
    for index in 0..N {
        let component = value[index];
        if !component.is_finite() {
            bail!("uniform '{name}' value must contain only finite components");
        }
        if min.is_some_and(|min| component < min[index])
            || max.is_some_and(|max| component > max[index])
        {
            bail!("uniform '{name}' value is outside its configured range");
        }
    }
    Ok(())
}

pub fn apply_to_runtime(
    runtime: &mut shadertoy::Runtime<'_>,
    values: &BTreeMap<String, UniformValue>,
) -> Result<()> {
    for (name, value) in values {
        match value {
            UniformValue::Bool(value) => runtime.set_uniform_i32(name, i32::from(*value))?,
            UniformValue::Int(value) => runtime.set_uniform_i32(name, *value)?,
            UniformValue::Float(value) => runtime.set_uniform_f32(name, &[*value])?,
            UniformValue::Vec2(value) => runtime.set_uniform_f32(name, value)?,
            UniformValue::Vec3(value) => runtime.set_uniform_f32(name, value)?,
            UniformValue::Vec4(value) => runtime.set_uniform_f32(name, value)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typed_overrides_and_rejects_out_of_range_values() {
        let definitions = vec![
            UniformDefinition::Float {
                name: "wave_height".into(),
                default: 1.0,
                min: Some(0.0),
                max: Some(2.0),
                step: Some(0.1),
            },
            UniformDefinition::Vec2 {
                name: "wind".into(),
                default: [1.0, 0.0],
                min: None,
                max: None,
                step: None,
            },
        ];
        let values = parse_assignments(
            &definitions,
            &["wave_height=1.5".into(), "wind=0.25,-0.75".into()],
        )
        .unwrap();
        assert_eq!(values["wave_height"], UniformValue::Float(1.5));
        assert_eq!(values["wind"], UniformValue::Vec2([0.25, -0.75]));
        assert!(parse_assignments(&definitions, &["wave_height=3".into()]).is_err());
    }

    #[test]
    fn rejects_builtin_and_duplicate_uniform_names() {
        for name in ["iTime", "iChannelTime", "iAudioBass", "iOutput"] {
            let builtin = UniformDefinition::Float {
                name: name.into(),
                default: 0.0,
                min: None,
                max: None,
                step: None,
            };
            assert!(validate_definitions(&[builtin]).is_err(), "{name}");
        }

        let first = UniformDefinition::Bool {
            name: "enabled".into(),
            default: true,
        };
        let second = UniformDefinition::Bool {
            name: "enabled".into(),
            default: false,
        };
        assert!(validate_definitions(&[first, second]).is_err());
    }
}
