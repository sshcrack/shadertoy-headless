use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn reload(
    root: &Path,
    runtime: &mut Runtime<'_>,
    loaded: &mut Option<LoadedManifest>,
    sources: &mut Option<SourceGraph>,
    width: &mut u32,
    height: &mut u32,
    fps: &mut f32,
    view: &mut String,
    preserve_state: bool,
) -> Result<()> {
    let saved = if preserve_state {
        loaded
            .as_ref()
            .map(|current| save_runtime_state(runtime, current, *width, *height, *fps))
            .transpose()?
    } else {
        None
    };

    let candidate = LoadedManifest::load(root)?;
    ensure_source_files_exist(&candidate)?;
    let (project, candidate_sources) = build_native_project_with_sources(&candidate)?;
    runtime.load_project(&project)?;

    if loaded.is_none() {
        *width = candidate.manifest.render.width;
        *height = candidate.manifest.render.height;
        *fps = candidate.manifest.render.fps;
    }

    if let Some(saved) = saved {
        runtime.set_fixed_state(saved.time, saved.frame, saved.fps)?;
        for (name, (saved_dimensions, data)) in saved.buffers {
            if let Some(pass) = candidate
                .manifest
                .passes
                .iter()
                .find(|pass| pass.name == name && pass.kind == PassKind::Buffer)
            {
                let (pass_width, pass_height) =
                    candidate.manifest.pass_dimensions(pass, *width, *height);
                if (pass_width, pass_height) == (saved_dimensions.width, saved_dimensions.height) {
                    let _ = runtime.restore_pass_rgba32f(&name, pass_width, pass_height, &data);
                }
            }
        }
    }

    let final_pass = candidate.manifest.final_pass().name.clone();
    if !candidate
        .manifest
        .passes
        .iter()
        .any(|pass| pass.name == *view)
    {
        *view = final_pass;
    }
    *loaded = Some(candidate);
    *sources = Some(candidate_sources);
    Ok(())
}

pub(super) fn reload_changed_sources(
    root: &Path,
    runtime: &mut Runtime<'_>,
    loaded: &LoadedManifest,
    sources: &mut SourceGraph,
    changed: &BTreeSet<PathBuf>,
) -> Result<bool> {
    let manifest_path = root.join(crate::manifest::MANIFEST_NAME);
    if changed.iter().any(|path| same_path(path, &manifest_path)) {
        return Ok(false);
    }

    let mut impacted = Vec::new();
    for pass in &loaded.manifest.passes {
        let Some(current) = sources.get(&pass.name) else {
            return Ok(false);
        };
        if changed.iter().any(|path| {
            current
                .dependencies
                .iter()
                .any(|dependency| same_path(path, dependency))
        }) {
            impacted.push(pass);
        }
    }

    if impacted.is_empty() {
        let source_like = changed.iter().all(|path| {
            matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("glsl" | "frag" | "vert")
            )
        });
        return Ok(source_like);
    }

    let mut replacements = Vec::with_capacity(impacted.len());
    for pass in impacted {
        replacements.push((pass.name.clone(), expand_pass(loaded, pass)?));
    }

    let mut applied: Vec<String> = Vec::new();
    for (name, replacement) in &replacements {
        if let Err(error) = runtime.reload_pass_source(name, &replacement.text) {
            for previous_name in applied.iter().rev() {
                if let Some(previous) = sources.get(previous_name) {
                    let _ = runtime.reload_pass_source(previous_name, &previous.text);
                }
            }
            return Err(error.into());
        }
        applied.push(name.clone());
    }
    for (name, replacement) in replacements {
        sources.insert(name, replacement);
    }
    Ok(true)
}

fn same_path(left: &Path, right: &Path) -> bool {
    let normalize = |path: &Path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    normalize(left) == normalize(right)
}

fn save_runtime_state(
    runtime: &mut Runtime<'_>,
    loaded: &LoadedManifest,
    width: u32,
    height: u32,
    fps: f32,
) -> Result<SavedRuntimeState> {
    let mut buffers = BTreeMap::new();
    for pass in &loaded.manifest.passes {
        if pass.kind != PassKind::Buffer {
            continue;
        }
        let (pass_width, pass_height) = loaded.manifest.pass_dimensions(pass, width, height);
        if let Ok(data) = runtime.snapshot_pass_rgba32f(&pass.name, pass_width, pass_height) {
            buffers.insert(
                pass.name.clone(),
                (
                    crate::state::BufferDimensions {
                        width: pass_width,
                        height: pass_height,
                    },
                    data,
                ),
            );
        }
    }
    Ok(SavedRuntimeState {
        time: runtime.time(),
        frame: runtime.frame(),
        fps,
        buffers,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_status(
    shared: &Shared,
    loaded: Option<&LoadedManifest>,
    runtime: &Runtime<'_>,
    width: u32,
    height: u32,
    fps: f32,
    paused: bool,
    view: &str,
    error: Option<String>,
    increment_sequence: bool,
) {
    let mut status = shared.status.write().expect("preview status lock poisoned");
    if let Some(loaded) = loaded {
        status.project = loaded.manifest.project.name.clone();
        status.final_pass = loaded.manifest.final_pass().name.clone();
        status.passes = loaded
            .manifest
            .passes
            .iter()
            .filter(|pass| pass.kind != PassKind::Cubemap)
            .map(|pass| pass.name.clone())
            .collect();
    }
    status.frame = runtime.frame();
    status.time = runtime.time();
    status.paused = paused;
    status.time_scale = runtime.time_scale();
    status.width = width;
    status.height = height;
    status.fps = fps;
    status.view = view.to_string();
    if error.is_some() {
        status.error = error;
    } else if increment_sequence {
        status.error = None;
    }
    if increment_sequence {
        status.sequence = status.sequence.wrapping_add(1);
    }
    broadcast_status(shared, &status);
}

pub(super) fn set_error(shared: &Shared, message: String) {
    let mut status = shared.status.write().expect("preview status lock poisoned");
    status.error = Some(message);
    broadcast_status(shared, &status);
}

pub(super) fn clear_error(shared: &Shared) {
    let mut status = shared.status.write().expect("preview status lock poisoned");
    status.error = None;
    broadcast_status(shared, &status);
}

fn broadcast_status(shared: &Shared, status: &PreviewStatus) {
    if let Ok(message) = serde_json::to_string(status) {
        let _ = shared.updates.send(message);
    }
}

pub(super) fn reload_event_kind(kind: &notify::EventKind) -> bool {
    matches!(
        kind,
        notify::EventKind::Any
            | notify::EventKind::Create(_)
            | notify::EventKind::Modify(_)
            | notify::EventKind::Remove(_)
    )
}

pub(super) fn relevant_watch_path(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    let Some(first) = relative.components().next() else {
        return false;
    };
    let first = first.as_os_str().to_string_lossy();
    if matches!(first.as_ref(), "target" | ".git" | ".shadertoy") {
        return false;
    }
    true
}
