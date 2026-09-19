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
        return Ok((time * fps).round() as i32);
    }
    if let Some(state) = state {
        return state
            .header
            .frame
            .checked_add(1)
            .context("state frame counter overflow");
    }
    Ok((loaded.manifest.render.preview_time * fps).round() as i32)
}

pub(super) fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("render dimensions must be positive");
    }
    if width > 16384 || height > 16384 {
        bail!("render dimensions exceed the 16384 pixel safety limit");
    }
    Ok(())
}

pub(super) fn validate_fps(fps: f32) -> Result<()> {
    if !fps.is_finite() || fps <= 0.0 || fps > 1000.0 {
        bail!("fps must be finite and in the range (0, 1000]");
    }
    Ok(())
}
