use super::*;

pub fn render_project(options: &RenderOptions) -> Result<Output> {
    let loaded = LoadedManifest::load(&options.project)?;
    ensure_source_files_exist(&loaded)?;
    let state = options.state.as_ref().map(StateFile::load).transpose()?;

    if let Some(state) = &state
        && state.header.project != loaded.manifest.project.name
    {
        bail!(
            "state belongs to project '{}' but current project is '{}'",
            state.header.project,
            loaded.manifest.project.name
        );
    }

    let (width, height) =
        resolve_dimensions(&loaded, state.as_ref(), options.width, options.height)?;
    let fps = resolve_fps(&loaded, state.as_ref(), options.fps)?;
    let target_frame =
        resolve_target_frame(&loaded, state.as_ref(), options.frame, options.time, fps)?;

    let output = options
        .output
        .clone()
        .unwrap_or_else(|| loaded.root.join("target/render.png"));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let overrides = load_overrides(&options.set_buffers, width, height)?;

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for rendering")?;
    let mut runtime = Runtime::new(&context)?;
    let project = build_native_project(&loaded)?;
    runtime.load_project(&project)?;

    let final_image = if let Some(state) = &state {
        restore_state(&mut runtime, state)?;
        render_from_restored_state(
            &mut runtime,
            state.header.frame,
            target_frame,
            fps,
            width,
            height,
            &overrides,
        )?
    } else {
        render_from_zero(&mut runtime, target_frame, fps, width, height, &overrides)?
    };

    let requested_pass = options.pass.as_deref();
    let image = match requested_pass {
        None => final_image.context("render did not produce a final image")?,
        Some(name) if name == loaded.manifest.final_pass().name => {
            final_image.context("render did not produce a final image")?
        }
        Some(name) => runtime.snapshot_pass_rgb(name, width, height)?,
    };
    save_rgb_png(&image, &output)?;

    Ok(Output {
        human: format!(
            "Rendered {}x{} frame {} ({:.3}s){} -> {}",
            width,
            height,
            runtime.frame(),
            runtime.time(),
            requested_pass
                .map(|name| format!(" pass '{name}'"))
                .unwrap_or_default(),
            output.display()
        ),
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "output": output,
            "width": width,
            "height": height,
            "fps": fps,
            "frame": runtime.frame(),
            "time": runtime.time(),
            "pass": requested_pass.unwrap_or(&loaded.manifest.final_pass().name),
            "state": options.state,
        }),
    })
}

pub(super) fn restore_state(runtime: &mut Runtime<'_>, state: &StateFile) -> Result<()> {
    runtime.set_fixed_state(state.header.time, state.header.frame, state.header.fps)?;
    for name in &state.header.buffers {
        let values = state
            .buffers
            .get(name)
            .with_context(|| format!("state is missing buffer '{name}'"))?;
        runtime.restore_pass_rgba32f(name, state.header.width, state.header.height, values)?;
    }
    Ok(())
}

pub(super) fn render_from_zero(
    runtime: &mut Runtime<'_>,
    target_frame: i32,
    fps: f32,
    width: u32,
    height: u32,
    overrides: &[(String, Vec<u8>)],
) -> Result<Option<RgbImage>> {
    if target_frame < 0 {
        bail!("target frame must be non-negative");
    }

    if target_frame == 0 {
        apply_overrides(runtime, overrides, width, height)?;
        return Ok(Some(runtime.render(width, height)?));
    }

    // Establish deterministic frame-0 buffer contents before advancing.
    let _ = runtime.render(width, height)?;
    for _frame in 1..target_frame {
        runtime.tick_fixed(1.0 / fps, fps)?;
        let _ = runtime.render(width, height)?;
    }
    apply_overrides(runtime, overrides, width, height)?;
    runtime.tick_fixed(1.0 / fps, fps)?;
    Ok(Some(runtime.render(width, height)?))
}

fn render_from_restored_state(
    runtime: &mut Runtime<'_>,
    state_frame: i32,
    target_frame: i32,
    fps: f32,
    width: u32,
    height: u32,
    overrides: &[(String, Vec<u8>)],
) -> Result<Option<RgbImage>> {
    if target_frame < state_frame {
        bail!("requested frame {target_frame} precedes restored state frame {state_frame}");
    }
    if target_frame == state_frame {
        if !overrides.is_empty() {
            apply_overrides(runtime, overrides, width, height)?;
        }
        return Ok(None);
    }

    let mut image = None;
    for frame in (state_frame + 1)..=target_frame {
        if frame == target_frame {
            apply_overrides(runtime, overrides, width, height)?;
        }
        runtime.tick_fixed(1.0 / fps, fps)?;
        image = Some(runtime.render(width, height)?);
    }
    Ok(image)
}

fn apply_overrides(
    runtime: &mut Runtime<'_>,
    overrides: &[(String, Vec<u8>)],
    width: u32,
    height: u32,
) -> Result<()> {
    for (name, rgba) in overrides {
        runtime.override_pass_rgba8(name, width, height, rgba)?;
    }
    Ok(())
}

fn resolve_dimensions(
    loaded: &LoadedManifest,
    state: Option<&StateFile>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<(u32, u32)> {
    if let Some(state) = state {
        if let Some(width) = width
            && width != state.header.width
        {
            bail!(
                "resumable state is {} pixels wide; rendering it at {} would discard buffer state",
                state.header.width,
                width
            );
        }
        if let Some(height) = height
            && height != state.header.height
        {
            bail!(
                "resumable state is {} pixels high; rendering it at {} would discard buffer state",
                state.header.height,
                height
            );
        }
        return Ok((state.header.width, state.header.height));
    }

    let width = width.unwrap_or(loaded.manifest.render.width);
    let height = height.unwrap_or(loaded.manifest.render.height);
    validate_dimensions(width, height)?;
    Ok((width, height))
}

fn resolve_fps(
    loaded: &LoadedManifest,
    state: Option<&StateFile>,
    fps: Option<f32>,
) -> Result<f32> {
    let fps = fps.unwrap_or_else(|| {
        state
            .map(|state| state.header.fps)
            .unwrap_or(loaded.manifest.render.fps)
    });
    validate_fps(fps)?;
    Ok(fps)
}

pub(super) fn resolve_target_frame(
    loaded: &LoadedManifest,
    state: Option<&StateFile>,
    frame: Option<i32>,
    time: Option<f32>,
    fps: f32,
) -> Result<i32> {
    if frame.is_some() && time.is_some() {
        bail!("use either --frame or --time, not both");
    }
    if let Some(frame) = frame {
        if frame < 0 {
            bail!("--frame must be non-negative");
        }
        return Ok(frame);
    }
    if let Some(time) = time {
        if !time.is_finite() || time < 0.0 {
            bail!("--time must be a finite non-negative number");
        }
        return frame_for_time(time, fps, "--time");
    }
    if let Some(state) = state {
        return state
            .header
            .frame
            .checked_add(1)
            .context("state frame counter overflow");
    }
    frame_for_time(
        loaded.manifest.render.preview_time,
        fps,
        "render.preview_time",
    )
}

pub(super) fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("render dimensions must be positive");
    }
    if width > crate::manifest::MAX_RENDER_DIMENSION
        || height > crate::manifest::MAX_RENDER_DIMENSION
    {
        bail!(
            "render dimensions exceed the {} pixel safety limit",
            crate::manifest::MAX_RENDER_DIMENSION
        );
    }
    Ok(())
}

pub(super) fn validate_fps(fps: f32) -> Result<()> {
    if !fps.is_finite() || fps <= 0.0 || fps > crate::manifest::MAX_RENDER_FPS {
        bail!(
            "fps must be finite and in the range (0, {}]",
            crate::manifest::MAX_RENDER_FPS
        );
    }
    Ok(())
}

fn frame_for_time(time: f32, fps: f32, source: &str) -> Result<i32> {
    let frame = f64::from(time) * f64::from(fps);
    if !frame.is_finite() || frame.round() > f64::from(i32::MAX) {
        bail!("{source} resolves to a frame outside the supported i32 range");
    }
    Ok(frame.round() as i32)
}

const MAX_BATCH_FRAMES: usize = 1024;
const MAX_CONTACT_SHEET_BYTES: usize = 256 * 1024 * 1024;

pub fn render_frames_project(options: &RenderFramesOptions) -> Result<Output> {
    let loaded = LoadedManifest::load(&options.project)?;
    ensure_source_files_exist(&loaded)?;
    let frames = normalize_frames(&options.frames)?;
    let (width, height) = resolve_dimensions(&loaded, None, options.width, options.height)?;
    let fps = resolve_fps(&loaded, None, options.fps)?;

    let selected_pass = options
        .pass
        .as_deref()
        .unwrap_or(&loaded.manifest.final_pass().name);
    let pass = loaded
        .manifest
        .passes
        .iter()
        .find(|pass| pass.name == selected_pass)
        .with_context(|| format!("unknown render pass '{selected_pass}'"))?;
    if pass.kind == PassKind::Cubemap {
        bail!("render-frames only supports the final image and 2D buffer passes");
    }

    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| loaded.root.join("target/frames"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut contact_sheet = options
        .contact_sheet
        .as_ref()
        .map(|path| prepare_contact_sheet(path, frames.len(), options.columns, width, height))
        .transpose()?;

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for multi-frame rendering")?;
    let mut runtime = Runtime::new(&context)?;
    let project = build_native_project(&loaded)?;
    runtime.load_project(&project)?;

    let final_pass = loaded.manifest.final_pass().name.as_str();
    let mut outputs = Vec::with_capacity(frames.len());
    let mut next_requested = 0usize;
    let max_frame = *frames
        .last()
        .expect("normalize_frames guarantees non-empty");

    for frame in 0..=max_frame {
        if frame > 0 {
            runtime.tick_fixed(1.0 / fps, fps)?;
        }
        let final_image = runtime.render(width, height)?;

        if frame != frames[next_requested] {
            continue;
        }

        let image = if selected_pass == final_pass {
            final_image
        } else {
            runtime.snapshot_pass_rgb(selected_pass, width, height)?
        };
        let path = output_dir.join(format!("frame-{frame:06}.png"));
        save_rgb_png(&image, &path)?;
        if let Some(sheet) = &mut contact_sheet {
            sheet.blit(next_requested, &image)?;
        }
        outputs.push(path);
        next_requested += 1;
        if next_requested == frames.len() {
            break;
        }
    }

    let contact_sheet_output = if let Some(sheet) = contact_sheet {
        Some(sheet.save()?)
    } else {
        None
    };

    let frame_list = frames
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Ok(Output {
        human: format!(
            "Rendered {}x{} frames [{}]{} -> {}{}",
            width,
            height,
            frame_list,
            options
                .pass
                .as_ref()
                .map(|name| format!(" pass '{name}'"))
                .unwrap_or_default(),
            output_dir.display(),
            contact_sheet_output
                .as_ref()
                .map(|path| format!("; contact sheet {}", path.display()))
                .unwrap_or_default()
        ),
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "output_dir": output_dir,
            "outputs": outputs,
            "contact_sheet": contact_sheet_output,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
            "pass": selected_pass,
        }),
    })
}

fn normalize_frames(frames: &[i32]) -> Result<Vec<i32>> {
    if frames.is_empty() {
        bail!("--frames must contain at least one frame");
    }
    if frames.len() > MAX_BATCH_FRAMES {
        bail!("--frames accepts at most {MAX_BATCH_FRAMES} entries");
    }
    if let Some(frame) = frames.iter().find(|frame| **frame < 0) {
        bail!("frame {frame} is negative; deterministic frames must be non-negative");
    }
    let mut normalized = frames.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

struct ContactSheet {
    path: PathBuf,
    columns: u32,
    width: u32,
    height: u32,
    sheet_width: u32,
    sheet_height: u32,
    pixels: Vec<u8>,
}

impl ContactSheet {
    fn blit(&mut self, index: usize, image: &RgbImage) -> Result<()> {
        if image.width != self.width || image.height != self.height {
            bail!("contact-sheet frame dimensions changed during rendering");
        }
        let mut source = image.pixels.clone();
        super::images::flip_rgb_rows(&mut source, image.width, image.height);
        let column = (index as u32) % self.columns;
        let row = (index as u32) / self.columns;
        let x = column * self.width;
        let y = row * self.height;
        let source_row_bytes = self.width as usize * 3;
        let sheet_row_bytes = self.sheet_width as usize * 3;
        for source_y in 0..self.height as usize {
            let source_start = source_y * source_row_bytes;
            let destination_start = (y as usize + source_y) * sheet_row_bytes + x as usize * 3;
            self.pixels[destination_start..destination_start + source_row_bytes]
                .copy_from_slice(&source[source_start..source_start + source_row_bytes]);
        }
        Ok(())
    }

    fn save(self) -> Result<PathBuf> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        ::image::save_buffer_with_format(
            &self.path,
            &self.pixels,
            self.sheet_width,
            self.sheet_height,
            ::image::ColorType::Rgb8,
            ::image::ImageFormat::Png,
        )
        .with_context(|| format!("failed to write contact sheet {}", self.path.display()))?;
        Ok(self.path)
    }
}

fn prepare_contact_sheet(
    path: &Path,
    frame_count: usize,
    requested_columns: Option<u32>,
    width: u32,
    height: u32,
) -> Result<ContactSheet> {
    let columns = match requested_columns {
        Some(0) => bail!("--columns must be positive"),
        Some(columns) => columns.min(frame_count as u32),
        None => (frame_count as f64).sqrt().ceil() as u32,
    };
    let rows = (frame_count as u32).div_ceil(columns);
    let sheet_width = width
        .checked_mul(columns)
        .context("contact-sheet width overflow")?;
    let sheet_height = height
        .checked_mul(rows)
        .context("contact-sheet height overflow")?;
    let bytes = (sheet_width as usize)
        .checked_mul(sheet_height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("contact-sheet allocation size overflow")?;
    if bytes > MAX_CONTACT_SHEET_BYTES {
        bail!(
            "contact sheet would require {} MiB; reduce resolution/frame count or change --columns (limit {} MiB)",
            bytes / (1024 * 1024),
            MAX_CONTACT_SHEET_BYTES / (1024 * 1024)
        );
    }
    Ok(ContactSheet {
        path: path.to_path_buf(),
        columns,
        width,
        height,
        sheet_width,
        sheet_height,
        pixels: vec![0; bytes],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_for_time_rejects_i32_overflow() {
        assert!(frame_for_time(f32::MAX, 1000.0, "--time").is_err());
    }

    #[test]
    fn normalize_frames_sorts_and_deduplicates() {
        assert_eq!(
            normalize_frames(&[120, 0, 60, 60]).unwrap(),
            vec![0, 60, 120]
        );
        assert!(normalize_frames(&[-1]).is_err());
        assert!(normalize_frames(&[]).is_err());
    }

    #[test]
    fn contact_sheet_uses_near_square_default() {
        let sheet = prepare_contact_sheet(Path::new("sheet.png"), 4, None, 10, 5).unwrap();
        assert_eq!(sheet.columns, 2);
        assert_eq!((sheet.sheet_width, sheet.sheet_height), (20, 10));
    }
}
