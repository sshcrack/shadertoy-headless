use super::*;

pub(super) struct BufferOverride {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub(super) fn load_overrides(
    assignments: &[String],
    manifest: &crate::manifest::Manifest,
    output_width: u32,
    output_height: u32,
) -> Result<Vec<BufferOverride>> {
    let mut result = Vec::new();
    for assignment in assignments {
        let (name, path) = split_assignment(assignment)?;
        let pass = manifest
            .passes
            .iter()
            .find(|pass| pass.name == name)
            .with_context(|| format!("unknown buffer override pass '{name}'"))?;
        if pass.kind != PassKind::Buffer {
            bail!("buffer override '{}' is not a buffer pass", name);
        }
        let (width, height) = manifest.pass_dimensions(pass, output_width, output_height);
        let image = ImageReader::open(path)
            .with_context(|| format!("failed to open buffer override {}", path.display()))?
            .decode()
            .with_context(|| format!("failed to decode buffer override {}", path.display()))?
            .to_rgba8();
        let dimensions = image.dimensions();
        if dimensions != (width, height) {
            bail!(
                "buffer override '{}' is {}x{} but pass '{}' is {}x{}; use an exact-size image",
                name,
                dimensions.0,
                dimensions.1,
                name,
                width,
                height
            );
        }
        let mut rgba = image.into_raw();
        flip_rgba_rows(&mut rgba, width, height);
        result.push(BufferOverride {
            name: name.to_string(),
            width,
            height,
            rgba,
        });
    }
    Ok(result)
}

pub(super) fn split_assignment(value: &str) -> Result<(&str, &Path)> {
    let (name, path) = value
        .split_once('=')
        .with_context(|| format!("expected BUFFER=IMAGE, got '{value}'"))?;
    if name.trim().is_empty() || path.trim().is_empty() {
        bail!("expected non-empty BUFFER=IMAGE assignment, got '{value}'");
    }
    Ok((name.trim(), Path::new(path.trim())))
}

pub(super) fn save_rgb_png(image: &RgbImage, output: &Path) -> Result<()> {
    let mut pixels = image.pixels.clone();
    flip_rgb_rows(&mut pixels, image.width, image.height);
    ::image::save_buffer(
        output,
        &pixels,
        image.width,
        image.height,
        ::image::ColorType::Rgb8,
    )
    .with_context(|| format!("failed to write PNG {}", output.display()))
}

pub fn rgb_png_bytes(image: &RgbImage) -> Result<Vec<u8>> {
    let mut pixels = image.pixels.clone();
    flip_rgb_rows(&mut pixels, image.width, image.height);
    let mut cursor = std::io::Cursor::new(Vec::new());
    ::image::write_buffer_with_format(
        &mut cursor,
        &pixels,
        image.width,
        image.height,
        ::image::ColorType::Rgb8,
        ::image::ImageFormat::Png,
    )
    .context("failed to encode preview PNG")?;
    Ok(cursor.into_inner())
}

pub fn flip_rgb_rows(data: &mut [u8], width: u32, height: u32) {
    flip_rows(data, width, height, 3);
}

pub(super) fn flip_rgba_rows(data: &mut [u8], width: u32, height: u32) {
    flip_rows(data, width, height, 4);
}

fn flip_rows(data: &mut [u8], width: u32, height: u32, channels: usize) {
    let Some(row) = (width as usize).checked_mul(channels) else {
        debug_assert!(false, "row size overflow");
        return;
    };
    let Some(expected_len) = row.checked_mul(height as usize) else {
        debug_assert!(false, "image size overflow");
        return;
    };
    debug_assert_eq!(data.len(), expected_len);
    for y in 0..(height as usize / 2) {
        let opposite = height as usize - 1 - y;
        let (before_opposite, opposite_and_after) = data.split_at_mut(opposite * row);
        let top = &mut before_opposite[y * row..(y + 1) * row];
        let bottom = &mut opposite_and_after[..row];
        top.swap_with_slice(bottom);
    }
}
