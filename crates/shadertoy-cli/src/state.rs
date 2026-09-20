use crate::manifest::RenderFormat;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"STSTATE1";
const LEGACY_STATE_FORMAT: u32 = 1;
const DIMENSION_STATE_FORMAT: u32 = 2;
const RENDER_FORMAT_STATE_FORMAT: u32 = 3;
const STATE_FORMAT: u32 = 4;
const MAX_HEADER_BYTES: usize = 1024 * 1024;
const MAX_BUFFERS: usize = 64;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferDimensions {
    pub width: u32,
    pub height: u32,
}

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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub buffer_dimensions: BTreeMap<String, BufferDimensions>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub buffer_formats: BTreeMap<String, RenderFormat>,
    /// Named SSBO sizes in bytes. Payloads follow image buffers in sorted name order.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub storage_buffers: BTreeMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct StateFile {
    pub header: StateHeader,
    pub buffers: BTreeMap<String, Vec<f32>>,
    pub storage_buffers: BTreeMap<String, Vec<u8>>,
}

impl StateFile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        project: String,
        width: u32,
        height: u32,
        fps: f32,
        time: f32,
        frame: i32,
        buffers: BTreeMap<String, Vec<f32>>,
        buffer_dimensions: BTreeMap<String, BufferDimensions>,
        buffer_formats: BTreeMap<String, RenderFormat>,
        storage_buffers: BTreeMap<String, Vec<u8>>,
    ) -> Result<Self> {
        validate_dimensions(width, height)?;
        if !fps.is_finite() || fps <= 0.0 || fps > crate::manifest::MAX_RENDER_FPS {
            bail!(
                "state fps must be finite and in the range (0, {}]",
                crate::manifest::MAX_RENDER_FPS
            );
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
        for (name, data) in &buffers {
            if name.is_empty() {
                bail!("state buffer name must not be empty");
            }
            let dimensions = buffer_dimensions
                .get(name)
                .copied()
                .unwrap_or(BufferDimensions { width, height });
            validate_dimensions(dimensions.width, dimensions.height)?;
            let expected = pixel_value_count(dimensions.width, dimensions.height)?;
            if data.len() != expected {
                bail!(
                    "state buffer '{}' has {} values; expected {} for {}x{}",
                    name,
                    data.len(),
                    expected,
                    dimensions.width,
                    dimensions.height
                );
            }
        }
        for name in buffer_dimensions.keys() {
            if !buffers.contains_key(name) {
                bail!("state buffer dimensions reference unknown buffer '{name}'");
            }
        }
        for name in buffer_formats.keys() {
            if !buffers.contains_key(name) {
                bail!("state buffer formats reference unknown buffer '{name}'");
            }
        }
        for name in buffers.keys() {
            if !buffer_formats.contains_key(name) {
                bail!("state buffer '{name}' is missing its render format");
            }
        }
        if storage_buffers.len() > MAX_BUFFERS {
            bail!("state contains too many storage buffers");
        }
        for (name, data) in &storage_buffers {
            if name.trim().is_empty() {
                bail!("state storage buffer name must not be empty");
            }
            if data.is_empty() {
                bail!("state storage buffer '{name}' must not be empty");
            }
        }
        let names = buffers.keys().cloned().collect();
        let storage_sizes = storage_buffers
            .iter()
            .map(|(name, data)| {
                Ok((
                    name.clone(),
                    u64::try_from(data.len()).context("storage buffer size exceeds u64")?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
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
                buffer_dimensions,
                buffer_formats,
                storage_buffers: storage_sizes,
            },
            buffers,
            storage_buffers,
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
        for name in self.header.storage_buffers.keys() {
            let data = self
                .storage_buffers
                .get(name)
                .with_context(|| format!("missing state storage buffer '{name}'"))?;
            file.write_all(data)?;
        }
        Ok(())
    }

    pub fn inspect_header(path: impl AsRef<Path>) -> Result<StateHeader> {
        let path = path.as_ref();
        let file = fs::File::open(path)
            .with_context(|| format!("failed to open state file {}", path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("failed to inspect state file {}", path.display()))?
            .len();
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .context("state file is truncated before its header")?;
        if &magic != MAGIC {
            bail!("{} is not a ShaderToy .ststate file", path.display());
        }

        let mut header_len_bytes = [0u8; 4];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            bail!("invalid state header length {header_len}");
        }

        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .context("state file is truncated in its JSON header")?;
        let header: StateHeader =
            serde_json::from_slice(&header_bytes).context("invalid state JSON header")?;
        validate_header(&header)?;
        validate_file_length(file_len, header_len, &header)?;
        Ok(header)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = fs::File::open(path)
            .with_context(|| format!("failed to open state file {}", path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("failed to inspect state file {}", path.display()))?
            .len();
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .context("state file is truncated before its header")?;
        if &magic != MAGIC {
            bail!("{} is not a ShaderToy .ststate file", path.display());
        }

        let mut header_len_bytes = [0u8; 4];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            bail!("invalid state header length {header_len}");
        }

        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .context("state file is truncated in its JSON header")?;
        let header: StateHeader =
            serde_json::from_slice(&header_bytes).context("invalid state JSON header")?;
        validate_header(&header)?;

        validate_file_length(file_len, header_len, &header)?;

        let mut buffers = BTreeMap::new();
        for name in &header.buffers {
            let dimensions =
                header
                    .buffer_dimensions
                    .get(name)
                    .copied()
                    .unwrap_or(BufferDimensions {
                        width: header.width,
                        height: header.height,
                    });
            let expected_values = pixel_value_count(dimensions.width, dimensions.height)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(expected_values)
                .with_context(|| format!("failed to allocate state buffer '{name}'"))?;

            let mut remaining = expected_values;
            let mut raw = [0u8; 64 * 1024];
            while remaining > 0 {
                let values_in_chunk = remaining.min(raw.len() / std::mem::size_of::<f32>());
                let bytes_in_chunk = values_in_chunk * std::mem::size_of::<f32>();
                reader.read_exact(&mut raw[..bytes_in_chunk])?;
                for chunk in raw[..bytes_in_chunk].as_chunks::<4>().0 {
                    values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                remaining -= values_in_chunk;
            }
            buffers.insert(name.clone(), values);
        }

        let mut storage_buffers = BTreeMap::new();
        for (name, size) in &header.storage_buffers {
            let size = usize::try_from(*size).context("state storage buffer size exceeds usize")?;
            let mut data = vec![0u8; size];
            reader
                .read_exact(&mut data)
                .with_context(|| format!("state file is truncated in storage buffer '{name}'"))?;
            storage_buffers.insert(name.clone(), data);
        }

        let state = Self {
            header,
            buffers,
            storage_buffers,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<()> {
        validate_header(&self.header)?;
        if self.buffers.len() != self.header.buffers.len() {
            bail!("state buffer table does not match its header");
        }
        for name in &self.header.buffers {
            let values = self
                .buffers
                .get(name)
                .with_context(|| format!("state header references missing buffer '{name}'"))?;
            let dimensions =
                self.header
                    .buffer_dimensions
                    .get(name)
                    .copied()
                    .unwrap_or(BufferDimensions {
                        width: self.header.width,
                        height: self.header.height,
                    });
            let expected = pixel_value_count(dimensions.width, dimensions.height)?;
            if values.len() != expected {
                bail!(
                    "state buffer '{}' has {} values; expected {} for {}x{}",
                    name,
                    values.len(),
                    expected,
                    dimensions.width,
                    dimensions.height
                );
            }
        }
        if self.storage_buffers.len() != self.header.storage_buffers.len() {
            bail!("state storage table does not match its header");
        }
        for (name, expected_size) in &self.header.storage_buffers {
            let data = self.storage_buffers.get(name).with_context(|| {
                format!("state header references missing storage buffer '{name}'")
            })?;
            if u64::try_from(data.len()).ok() != Some(*expected_size) {
                bail!(
                    "state storage buffer '{}' has {} bytes; expected {}",
                    name,
                    data.len(),
                    expected_size
                );
            }
        }
        Ok(())
    }

    pub fn replace_storage_buffer(&mut self, name: &str, data: Vec<u8>) -> Result<()> {
        let expected = self
            .header
            .storage_buffers
            .get(name)
            .copied()
            .with_context(|| format!("state has no storage buffer named '{name}'"))?;
        if u64::try_from(data.len()).ok() != Some(expected) {
            bail!(
                "replacement storage buffer '{}' has {} bytes; expected {}",
                name,
                data.len(),
                expected
            );
        }
        self.storage_buffers.insert(name.to_string(), data);
        Ok(())
    }

    pub fn buffer_dimensions(&self, name: &str) -> Result<BufferDimensions> {
        if !self.buffers.contains_key(name) {
            bail!("state has no buffer named '{name}'");
        }
        Ok(self
            .header
            .buffer_dimensions
            .get(name)
            .copied()
            .unwrap_or(BufferDimensions {
                width: self.header.width,
                height: self.header.height,
            }))
    }

    pub fn replace_buffer_rgba8(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<()> {
        if !self.buffers.contains_key(name) {
            bail!("state has no buffer named '{name}'");
        }
        let dimensions =
            self.header
                .buffer_dimensions
                .get(name)
                .copied()
                .unwrap_or(BufferDimensions {
                    width: self.header.width,
                    height: self.header.height,
                });
        if width != dimensions.width || height != dimensions.height {
            bail!(
                "replacement image is {}x{} but state buffer '{}' is {}x{}",
                width,
                height,
                name,
                dimensions.width,
                dimensions.height
            );
        }
        let expected_bytes = pixel_value_count(width, height)?;
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

fn validate_file_length(file_len: u64, header_len: usize, header: &StateHeader) -> Result<()> {
    let mut payload_bytes = 0usize;
    for name in &header.buffers {
        let dimensions = header
            .buffer_dimensions
            .get(name)
            .copied()
            .unwrap_or(BufferDimensions {
                width: header.width,
                height: header.height,
            });
        let values = pixel_value_count(dimensions.width, dimensions.height)?;
        let bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .context("state buffer size overflow")?;
        payload_bytes = payload_bytes
            .checked_add(bytes)
            .context("state file size overflow")?;
    }
    for size in header.storage_buffers.values() {
        let bytes = usize::try_from(*size).context("state storage size exceeds usize")?;
        payload_bytes = payload_bytes
            .checked_add(bytes)
            .context("state file size overflow")?;
    }
    let prefix_bytes = MAGIC
        .len()
        .checked_add(std::mem::size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(header_len))
        .context("state file size overflow")?;
    let expected_file_len = prefix_bytes
        .checked_add(payload_bytes)
        .context("state file size overflow")?;
    let expected_file_len =
        u64::try_from(expected_file_len).context("state file size exceeds u64")?;
    if file_len != expected_file_len {
        let actual_payload = file_len.saturating_sub(prefix_bytes as u64);
        bail!(
            "state payload has {actual_payload} bytes; expected {payload_bytes} for {} buffers",
            header.buffers.len()
        );
    }
    Ok(())
}

fn validate_header(header: &StateHeader) -> Result<()> {
    if !matches!(
        header.format,
        LEGACY_STATE_FORMAT | DIMENSION_STATE_FORMAT | RENDER_FORMAT_STATE_FORMAT | STATE_FORMAT
    ) {
        bail!(
            "unsupported .ststate format {}; supported formats are {}, {}, {}, and {}",
            header.format,
            LEGACY_STATE_FORMAT,
            DIMENSION_STATE_FORMAT,
            RENDER_FORMAT_STATE_FORMAT,
            STATE_FORMAT
        );
    }
    if header.format == LEGACY_STATE_FORMAT && !header.buffer_dimensions.is_empty() {
        bail!("legacy .ststate format cannot contain per-buffer dimensions");
    }
    if header.format < RENDER_FORMAT_STATE_FORMAT && !header.buffer_formats.is_empty() {
        bail!("legacy .ststate formats cannot contain per-buffer render formats");
    }
    if header.format < STATE_FORMAT && !header.storage_buffers.is_empty() {
        bail!("legacy .ststate formats cannot contain storage buffers");
    }
    if header.project.trim().is_empty() {
        bail!("state project name must not be empty");
    }
    validate_dimensions(header.width, header.height)?;
    if !header.fps.is_finite() || header.fps <= 0.0 || header.fps > crate::manifest::MAX_RENDER_FPS
    {
        bail!(
            "state fps must be finite and in the range (0, {}]",
            crate::manifest::MAX_RENDER_FPS
        );
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
    let mut names = HashSet::with_capacity(header.buffers.len());
    for name in &header.buffers {
        if name.trim().is_empty() {
            bail!("state buffer name must not be empty");
        }
        if !names.insert(name.as_str()) {
            bail!("state contains duplicate buffer '{name}'");
        }
    }
    for (name, dimensions) in &header.buffer_dimensions {
        if !names.contains(name.as_str()) {
            bail!("state has dimensions for unknown buffer '{name}'");
        }
        validate_dimensions(dimensions.width, dimensions.height)?;
    }
    for name in header.buffer_formats.keys() {
        if !names.contains(name.as_str()) {
            bail!("state has a render format for unknown buffer '{name}'");
        }
    }
    if header.format >= RENDER_FORMAT_STATE_FORMAT {
        for name in &header.buffers {
            if !header.buffer_formats.contains_key(name) {
                bail!("state buffer '{name}' is missing its render format");
            }
        }
    }
    if header.storage_buffers.len() > MAX_BUFFERS {
        bail!("state contains too many storage buffers");
    }
    for (name, size) in &header.storage_buffers {
        if name.trim().is_empty() {
            bail!("state storage buffer name must not be empty");
        }
        if *size == 0 {
            bail!("state storage buffer '{name}' must have a positive size");
        }
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("state dimensions must be positive");
    }
    if width > crate::manifest::MAX_RENDER_DIMENSION
        || height > crate::manifest::MAX_RENDER_DIMENSION
    {
        bail!(
            "state dimensions {}x{} exceed the maximum {}",
            width,
            height,
            crate::manifest::MAX_RENDER_DIMENSION
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

#[cfg(test)]
mod tests {
    use super::*;

    fn header(buffers: Vec<String>) -> StateHeader {
        StateHeader {
            format: STATE_FORMAT,
            project: "test".into(),
            width: 1,
            height: 1,
            fps: 60.0,
            time: 0.0,
            frame: 0,
            buffers,
            buffer_dimensions: BTreeMap::new(),
            buffer_formats: BTreeMap::new(),
            storage_buffers: BTreeMap::new(),
        }
    }

    #[test]
    fn header_rejects_empty_and_duplicate_buffer_names() {
        assert!(validate_header(&header(vec!["".into()])).is_err());
        assert!(validate_header(&header(vec!["a".into(), "a".into()])).is_err());
    }

    #[test]
    fn header_uses_render_fps_limit() {
        let mut value = header(Vec::new());
        value.fps = crate::manifest::MAX_RENDER_FPS + 1.0;
        assert!(validate_header(&value).is_err());
    }

    #[test]
    fn legacy_v1_header_without_per_buffer_dimensions_remains_valid() {
        let mut value = header(vec!["buffer-a".into()]);
        value.format = LEGACY_STATE_FORMAT;
        assert!(validate_header(&value).is_ok());

        value.buffer_dimensions.insert(
            "buffer-a".into(),
            BufferDimensions {
                width: 1,
                height: 1,
            },
        );
        assert!(validate_header(&value).is_err());
    }
    #[test]
    fn legacy_v2_header_without_render_formats_remains_valid() {
        let mut value = header(vec!["buffer-a".into()]);
        value.format = DIMENSION_STATE_FORMAT;
        value.buffer_dimensions.insert(
            "buffer-a".into(),
            BufferDimensions {
                width: 1,
                height: 1,
            },
        );
        assert!(validate_header(&value).is_ok());
    }

    #[test]
    fn current_header_requires_render_format_for_every_buffer() {
        let mut value = header(vec!["buffer-a".into()]);
        value.format = RENDER_FORMAT_STATE_FORMAT;
        assert!(validate_header(&value).is_err());
        value
            .buffer_formats
            .insert("buffer-a".into(), RenderFormat::Rg32f);
        assert!(validate_header(&value).is_ok());
    }
}
