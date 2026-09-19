use super::state::{clear_error, reload, set_error, update_status};
use super::*;

pub(super) fn render_loop(
    root: PathBuf,
    shared: Shared,
    controls: mpsc::Receiver<Control>,
    preserve_reload_state: bool,
    runtime: &mut Runtime<'_>,
) {
    let mut loaded: Option<LoadedManifest> = None;
    let mut width = 1280u32;
    let mut height = 720u32;
    let mut fps = 60.0f32;
    let mut paused = false;
    let mut view = String::from("image");
    let mut fresh = true;
    let mut force_render = true;
    let mut reload_due: Option<Instant> = None;
    let mut next_frame = Instant::now();

    match reload(
        &root,
        runtime,
        &mut loaded,
        &mut width,
        &mut height,
        &mut fps,
        &mut view,
        preserve_reload_state,
    ) {
        Ok(()) => {
            update_status(
                &shared,
                loaded.as_ref(),
                runtime,
                width,
                height,
                fps,
                paused,
                &view,
                None,
                false,
            );
        }
        Err(error) => {
            update_status(
                &shared,
                loaded.as_ref(),
                runtime,
                width,
                height,
                fps,
                paused,
                &view,
                Some(error.to_string()),
                false,
            );
        }
    }
    'main: loop {
        let mut pending = Vec::new();
        while let Ok(control) = controls.try_recv() {
            pending.push(control);
        }
        for control in pending {
            if !handle_control(
                control,
                &root,
                &shared,
                runtime,
                &mut loaded,
                &mut width,
                &mut height,
                &mut fps,
                &mut paused,
                &mut view,
                &mut fresh,
                &mut force_render,
                &mut reload_due,
                &mut next_frame,
            ) {
                break 'main;
            }
        }

        if let Some(due) = reload_due
            && Instant::now() >= due
        {
            reload_due = None;
            match reload(
                &root,
                runtime,
                &mut loaded,
                &mut width,
                &mut height,
                &mut fps,
                &mut view,
                preserve_reload_state,
            ) {
                Ok(()) => {
                    fresh = true;
                    force_render = true;
                    clear_error(&shared);
                }
                Err(error) => {
                    // Runtime::load_project compiles before replacing the active pipeline.
                    // Keep rendering the last good project and frame.
                    set_error(&shared, error.to_string());
                }
            }
        }

        let now = Instant::now();
        let clients = shared.clients.load(Ordering::Relaxed);
        let scheduled = !paused && clients > 0 && now >= next_frame;
        if loaded.is_some() && (force_render || scheduled) {
            let should_advance = !fresh && !paused && scheduled;
            if should_advance && let Err(error) = runtime.tick_fixed(1.0 / fps, fps) {
                set_error(&shared, error.to_string());
            }

            match runtime.render(width, height) {
                Ok(final_image) => {
                    let selected = if let Some(project) = &loaded {
                        if view == project.manifest.final_pass().name {
                            Some(final_image)
                        } else {
                            match runtime.snapshot_pass_rgb(&view, width, height) {
                                Ok(image) => Some(image),
                                Err(error) => {
                                    set_error(&shared, error.to_string());
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    };

                    if let Some(image) = selected {
                        match rgb_png_bytes(&image) {
                            Ok(png) => {
                                let png = Bytes::from(png);
                                *shared
                                    .frame_png
                                    .write()
                                    .expect("preview frame lock poisoned") = png.clone();
                                update_status(
                                    &shared,
                                    loaded.as_ref(),
                                    runtime,
                                    width,
                                    height,
                                    fps,
                                    paused,
                                    &view,
                                    None,
                                    true,
                                );
                                let _ = shared.frames.send(png);
                            }
                            Err(error) => set_error(&shared, error.to_string()),
                        }
                    }
                    let _ = runtime.clear_key_transients();
                }
                Err(error) => set_error(&shared, error.to_string()),
            }
            fresh = false;
            force_render = false;
            next_frame = Instant::now() + Duration::from_secs_f64(1.0 / f64::from(fps.max(1.0)));
        }

        let timeout = if !paused && shared.clients.load(Ordering::Relaxed) > 0 {
            next_frame
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(50))
        } else {
            Duration::from_millis(50)
        };
        match controls.recv_timeout(timeout) {
            Ok(control) => {
                if !handle_control(
                    control,
                    &root,
                    &shared,
                    runtime,
                    &mut loaded,
                    &mut width,
                    &mut height,
                    &mut fps,
                    &mut paused,
                    &mut view,
                    &mut fresh,
                    &mut force_render,
                    &mut reload_due,
                    &mut next_frame,
                ) {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_control(
    control: Control,
    root: &Path,
    shared: &Shared,
    runtime: &mut Runtime<'_>,
    loaded: &mut Option<LoadedManifest>,
    width: &mut u32,
    height: &mut u32,
    fps: &mut f32,
    paused: &mut bool,
    view: &mut String,
    fresh: &mut bool,
    force_render: &mut bool,
    reload_due: &mut Option<Instant>,
    next_frame: &mut Instant,
) -> bool {
    match control {
        Control::Shutdown => return false,
        Control::Reload => {
            *reload_due = Some(Instant::now() + Duration::from_millis(120));
        }
        Control::Pause => {
            if let Err(error) = runtime.pause() {
                set_error(shared, error.to_string());
            } else {
                *paused = true;
                *force_render = true;
            }
        }
        Control::Resume => {
            if let Err(error) = runtime.resume() {
                set_error(shared, error.to_string());
            } else {
                *paused = false;
                *next_frame = Instant::now();
            }
        }
        Control::Reset => match reload(root, runtime, loaded, width, height, fps, view, false) {
            Ok(()) => {
                *fresh = true;
                *force_render = true;
                clear_error(shared);
            }
            Err(error) => set_error(shared, error.to_string()),
        },
        Control::Step => {
            let was_paused = *paused;
            if was_paused {
                let _ = runtime.resume();
            }
            if let Err(error) = runtime.tick_fixed(1.0 / *fps, *fps) {
                set_error(shared, error.to_string());
            } else {
                *fresh = true;
                *force_render = true;
            }
            if was_paused {
                let _ = runtime.pause();
            }
        }
        Control::View(pass) => {
            if let Some(project) = loaded {
                if project
                    .manifest
                    .passes
                    .iter()
                    .any(|candidate| candidate.name == pass && candidate.kind != PassKind::Cubemap)
                {
                    *view = pass;
                    *force_render = true;
                } else {
                    set_error(shared, format!("unknown/non-2D preview pass '{pass}'"));
                }
            }
        }
        Control::Resolution(new_width, new_height) => {
            if new_width == 0
                || new_height == 0
                || new_width > MAX_PREVIEW_DIMENSION
                || new_height > MAX_PREVIEW_DIMENSION
            {
                set_error(
                    shared,
                    format!(
                        "preview resolution must be between 1x1 and {0}x{0}",
                        MAX_PREVIEW_DIMENSION
                    ),
                );
            } else {
                *width = new_width;
                *height = new_height;
                match reload(root, runtime, loaded, width, height, fps, view, false) {
                    Ok(()) => {
                        *fresh = true;
                        *force_render = true;
                        clear_error(shared);
                    }
                    Err(error) => set_error(shared, error.to_string()),
                }
            }
        }
        Control::TimeScale(value) => {
            if value.is_finite() && (-8.0..=8.0).contains(&value) {
                if let Err(error) = runtime.set_time_scale(value) {
                    set_error(shared, error.to_string());
                } else {
                    *force_render = true;
                }
            } else {
                set_error(
                    shared,
                    "time scale must be a finite log2 value in [-8, 8]".into(),
                );
            }
        }
        Control::Mouse {
            x,
            y,
            down,
            clicked,
        } => {
            if let Err(error) = runtime.set_mouse(x, y, down, clicked) {
                set_error(shared, error.to_string());
            } else if *paused {
                *force_render = true;
            }
        }
        Control::Key {
            code,
            down,
            pressed,
        } => {
            if let Err(error) = runtime.set_key(code, down, pressed) {
                set_error(shared, error.to_string());
            } else if *paused {
                *force_render = true;
            }
        }
    }
    true
}
