use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"STSTATE1";
const STATE_FORMAT: u32 = 1;
const MAX_HEADER_BYTES: usize = 1024 * 1024;
const MAX_BUFFERS: usize = 64;
const MAX_DIMENSION: u32 = 16384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateHeader {
    pub format: u32,
    pub project: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub time: f32,
    pub frame: i32,
    pub buffers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StateFile {
    pub header: StateHeader,
    pub buffers: BTreeMap<String, Vec<f32>>,
}

impl StateFile {
    pub fn new(
        project: String,
        width: u32,
        height: u32,
        fps: f32,
        time: f32,
        frame: i32,
        buffers: BTreeMap<String, Vec<f32>>,
    ) -> Result<Self> {
        validate_dimensions(width, height)?;
        if !fps.is_finite() || fps <= 0.0 {
            bail!("state fps must be a positive finite number");
        }
        if !time.is_finite() || time < 0.0 {
            bail!("state time must be a finite non-negative number");
        }
        if frame < 0 {
            bail!("state frame must be non-negative");
        }
        if buffers.len() > MAX_BUFFERS {
            bail!("state contains too many buffers");
        }
        let expected = pixel_value_count(width, height)?;
        for (name, data) in &buffers {
            if name.is_empty() {
                bail!("state buffer name must not be empty");
            }
            if data.len() != expected {
                bail!(
                    "state buffer '{}' has {} values; expected {}",
                    name,
                    data.len(),
                    expected
                );
            }
        }
        let names = buffers.keys().cloned().collect();
        Ok(Self {
            header: StateHeader {
                format: STATE_FORMAT,
                project,
                width,
                height,
                fps,
                time,
                frame,
                buffers: names,
            },
            buffers,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let header =
            serde_json::to_vec(&self.header).context("failed to serialize state header")?;
        if header.len() > MAX_HEADER_BYTES {
            bail!("state header is unexpectedly large");
        }

        let mut file = fs::File::create(path)
            .with_context(|| format!("failed to create state file {}", path.display()))?;
        file.write_all(MAGIC)?;
        file.write_all(&(header.len() as u32).to_le_bytes())?;
        file.write_all(&header)?;
        for name in &self.header.buffers {
            let values = self
                .buffers
                .get(name)
                .with_context(|| format!("missing state buffer '{name}'"))?;
            for value in values {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read state file {}", path.display()))?;
        let mut cursor = Cursor::new(bytes.as_slice());

        let mut magic = [0u8; 8];
        cursor
            .read_exact(&mut magic)
            .context("state file is truncated before its header")?;
        if &magic != MAGIC {
            bail!("{} is not a ShaderToy .ststate file", path.display());
        }

        let mut header_len_bytes = [0u8; 4];
        cursor.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            bail!("invalid state header length {header_len}");
        }

        let mut header_bytes = vec![0u8; header_len];
        cursor
            .read_exact(&mut header_bytes)
            .context("state file is truncated in its JSON header")?;
        let header: StateHeader =
            serde_json::from_slice(&header_bytes).context("invalid state JSON header")?;
        validate_header(&header)?;

        let expected_values = pixel_value_count(header.width, header.height)?;
        let expected_bytes_per_buffer = expected_values
            .checked_mul(std::mem::size_of::<f32>())
            .context("state buffer size overflow")?;
        let remaining = bytes.len() - cursor.position() as usize;
        let expected_remaining = expected_bytes_per_buffer
            .checked_mul(header.buffers.len())
            .context("state file size overflow")?;
        if remaining != expected_remaining {
            bail!(
                "state payload has {remaining} bytes; expected {expected_remaining} for {} buffers",
                header.buffers.len()
            );
        }

        let mut buffers = BTreeMap::new();
        for name in &header.buffers {
            let mut raw = vec![0u8; expected_bytes_per_buffer];
            cursor.read_exact(&mut raw)?;
            let mut values = Vec::with_capacity(expected_values);
            for chunk in raw.chunks_exact(4) {
                values.push(f32::from_le_bytes(
                    chunk.try_into().expect("four-byte chunk"),
                ));
            }
            if buffers.insert(name.clone(), values).is_some() {
                bail!("state contains duplicate buffer '{name}'");
            }
        }

        let state = Self { header, buffers };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<()> {
        validate_header(&self.header)?;
        let expected = pixel_value_count(self.header.width, self.header.height)?;
        if self.buffers.len() != self.header.buffers.len() {
            bail!("state buffer table does not match its header");
        }
        for name in &self.header.buffers {
            let values = self
                .buffers
                .get(name)
                .with_context(|| format!("state header references missing buffer '{name}'"))?;
            if values.len() != expected {
                bail!(
                    "state buffer '{}' has {} values; expected {}",
                    name,
                    values.len(),
                    expected
                );
            }
        }
        Ok(())
    }

    pub fn replace_buffer_rgba8(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<()> {
        if width != self.header.width || height != self.header.height {
            bail!(
                "replacement image is {}x{} but state is {}x{}",
                width,
                height,
                self.header.width,
                self.header.height
            );
        }
        if !self.buffers.contains_key(name) {
            bail!("state has no buffer named '{name}'");
        }
        let expected_bytes = width as usize * height as usize * 4;
        if rgba.len() != expected_bytes {
            bail!("replacement RGBA8 payload has the wrong size");
        }

        let values = rgba
            .iter()
            .map(|value| f32::from(*value) / 255.0)
            .collect::<Vec<_>>();
        self.buffers.insert(name.to_string(), values);
        Ok(())
    }
}

fn validate_header(header: &StateHeader) -> Result<()> {
    if header.format != STATE_FORMAT {
        bail!(
            "unsupported .ststate format {}; supported format is {}",
            header.format,
            STATE_FORMAT
        );
    }
    if header.project.trim().is_empty() {
        bail!("state project name must not be empty");
    }
    validate_dimensions(header.width, header.height)?;
    if !header.fps.is_finite() || header.fps <= 0.0 {
        bail!("state fps must be a positive finite number");
    }
    if !header.time.is_finite() || header.time < 0.0 {
        bail!("state time must be a finite non-negative number");
    }
    if header.frame < 0 {
        bail!("state frame must be non-negative");
    }
    if header.buffers.len() > MAX_BUFFERS {
        bail!("state contains too many buffers");
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("state dimensions must be positive");
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        bail!(
            "state dimensions {}x{} exceed the maximum {}",
            width,
            height,
            MAX_DIMENSION
        );
    }
    Ok(())
}

fn pixel_value_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .context("state dimensions overflow")
}
