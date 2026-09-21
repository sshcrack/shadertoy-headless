use super::*;
use std::ffi::OsStr;
use std::process::Command;
use tempfile::TempDir;

const MAX_BLIND_SOURCES: usize = 64;
const MAX_BLIND_IMAGES_PER_SOURCE: usize = 128;

struct PreparedSource {
    identity: String,
    images: Vec<RgbImage>,
}

struct GitWorktree {
    repo_root: PathBuf,
    checkout: PathBuf,
    _parent: TempDir,
}

impl Drop for GitWorktree {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .arg("-C")
            .arg(&self.repo_root)
            .args(["worktree", "remove", "--force"])
            .arg(&self.checkout)
            .status();
    }
}

pub fn create_blind_comparison(options: &BlindCreateOptions) -> Result<Output> {
    if !(2..=MAX_BLIND_SOURCES).contains(&options.sources.len()) {
        bail!(
            "blind create requires 2..={MAX_BLIND_SOURCES} sources (got {})",
            options.sources.len()
        );
    }
    let frames = normalize_frames(&options.frames)?;
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/blind-comparison"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let git_root = if options
        .sources
        .iter()
        .any(|source| source.starts_with("git:"))
    {
        Some(resolve_git_root(options.git_root.as_deref())?)
    } else {
        None
    };

    let mut worktrees = Vec::new();
    let mut prepared = Vec::with_capacity(options.sources.len());
    for source in &options.sources {
        if source.starts_with("git:") {
            let root = git_root.as_deref().expect("git root resolved above");
            let (identity, project, worktree) = materialize_git_source(root, source)?;
            let images = prepare_path_source(
                &project,
                &frames,
                options.width,
                options.height,
                options.fps,
            )?;
            prepared.push(PreparedSource { identity, images });
            worktrees.push(worktree);
        } else {
            let path = PathBuf::from(source);
            let images =
                prepare_path_source(&path, &frames, options.width, options.height, options.fps)?;
            prepared.push(PreparedSource {
                identity: path.display().to_string(),
                images,
            });
        }
    }

    let image_count = prepared
        .first()
        .map(|source| source.images.len())
        .unwrap_or(0);
    if image_count == 0 {
        bail!("blind sources produced no images");
    }
    if image_count > MAX_BLIND_IMAGES_PER_SOURCE {
        bail!("blind source contains {image_count} images; limit is {MAX_BLIND_IMAGES_PER_SOURCE}");
    }
    for source in &prepared {
        if source.images.len() != image_count {
            bail!(
                "blind sources must contain the same number of images; '{}' has {}, expected {}",
                source.identity,
                source.images.len(),
                image_count
            );
        }
    }

    let width = prepared[0].images[0].width;
    let height = prepared[0].images[0].height;
    for source in &prepared {
        for (index, image) in source.images.iter().enumerate() {
            if image.width != width || image.height != height {
                bail!(
                    "all blinded images must have identical dimensions; '{}' image {} is {}x{}, expected {}x{}",
                    source.identity,
                    index,
                    image.width,
                    image.height,
                    width,
                    height
                );
            }
        }
    }

    let entropy_context = format!(
        "external|{}|{}|{}|{}",
        output_dir.display(),
        width,
        height,
        prepared
            .iter()
            .map(|source| source.identity.as_str())
            .collect::<Vec<_>>()
            .join("|")
    );
    let plan = super::blind::BlindPlan::new(prepared.len(), &entropy_context)?;
    let contact_path = output_dir.join("blind-contact-sheet.png");
    let cell_count = prepared
        .len()
        .checked_mul(image_count)
        .context("blind contact-sheet cell count overflow")?;
    let mut sheet = render::prepare_contact_sheet(
        &contact_path,
        cell_count,
        Some(prepared.len() as u32),
        width,
        height,
    )?;

    let variants_dir = output_dir.join("variants");
    if variants_dir.exists() {
        fs::remove_dir_all(&variants_dir).with_context(|| {
            format!(
                "failed to clear stale blind outputs {}",
                variants_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&variants_dir)?;

    let mut outputs = Vec::with_capacity(prepared.len());
    let mut public_variants = Vec::with_capacity(prepared.len());
    for (position, original_index) in plan.order().iter().copied().enumerate() {
        let label = plan.label(position);
        let anonymous_dir = variants_dir.join(label);
        fs::create_dir_all(&anonymous_dir)?;
        for (image_index, image) in prepared[original_index].images.iter().enumerate() {
            let path = anonymous_dir.join(format!("image-{image_index:03}.png"));
            save_rgb_png(image, &path)?;
            sheet.blit(image_index * prepared.len() + position, image)?;
        }
        outputs.push(anonymous_dir.clone());
        public_variants.push(json!({
            "label": label,
            "output": anonymous_dir,
        }));
    }
    let contact_sheet = sheet.save()?;

    let variants = prepared
        .iter()
        .map(|source| vec![format!("source={}", source.identity)])
        .collect::<Vec<_>>();
    let session = super::blind::write_blind_session(
        &plan,
        &super::blind::BlindSessionSpec {
            output_dir: &output_dir,
            project: "external-comparison",
            frame: *frames.first().unwrap_or(&0),
            pass: "external",
            width,
            height,
            contact_sheet: &contact_sheet,
            outputs: &outputs,
            variants: &variants,
        },
    )?;

    drop(worktrees);

    Ok(Output {
        human: format!(
            "Created blinded comparison with {} sources x {} image(s) -> {}; inspect {} and record a judgment with 'shadertoy blind judge {} --pick LABEL --reason ...' before revealing",
            prepared.len(),
            image_count,
            output_dir.display(),
            contact_sheet.display(),
            session.display()
        ),
        json: json!({
            "ok": true,
            "mode": "external",
            "output_dir": output_dir,
            "contact_sheet": contact_sheet,
            "blind_session": session,
            "variant_count": prepared.len(),
            "images_per_variant": image_count,
            "width": width,
            "height": height,
            "variants": public_variants,
        }),
    })
}

fn prepare_path_source(
    path: &Path,
    frames: &[i32],
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f32>,
) -> Result<Vec<RgbImage>> {
    if path.is_file() {
        if is_image_path(path) {
            return Ok(vec![load_image(path)?]);
        }
        if path.extension().and_then(OsStr::to_str) == Some("sttf") {
            return render_sttf_frames(path, frames, width, height, fps);
        }
        bail!(
            "unsupported blind source file {}; expected PNG/JPEG or .sttf",
            path.display()
        );
    }
    if !path.is_dir() {
        bail!("blind source does not exist: {}", path.display());
    }
    if path.join("ShaderToy.toml").is_file() {
        return render_project_frames(path, frames, width, height, fps);
    }

    let mut paths = Vec::new();
    collect_images(path, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        bail!(
            "blind source directory {} contains no PNG/JPEG images and is not a ShaderToy project",
            path.display()
        );
    }
    if paths.len() > MAX_BLIND_IMAGES_PER_SOURCE {
        bail!(
            "blind source directory {} contains {} images; limit is {}",
            path.display(),
            paths.len(),
            MAX_BLIND_IMAGES_PER_SOURCE
        );
    }
    paths.iter().map(|path| load_image(path)).collect()
}

fn render_project_frames(
    project: &Path,
    frames: &[i32],
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f32>,
) -> Result<Vec<RgbImage>> {
    let temp = tempfile::tempdir().context("failed to create temporary blind render directory")?;
    let output = render::render_frames_project(&RenderFramesOptions {
        project: project.to_path_buf(),
        output_dir: Some(temp.path().to_path_buf()),
        contact_sheet: None,
        columns: None,
        pass: None,
        width,
        height,
        fps,
        frames: frames.to_vec(),
        range: None,
        set_uniforms: Vec::new(),
    })?;
    let paths = output.json["outputs"]
        .as_array()
        .context("project blind render did not report outputs")?;
    paths
        .iter()
        .map(|path| {
            let path = path
                .as_str()
                .context("project blind render output path is not a string")?;
            load_image(Path::new(path))
        })
        .collect()
}

fn render_sttf_frames(
    path: &Path,
    frames: &[i32],
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f32>,
) -> Result<Vec<RgbImage>> {
    let width = width.unwrap_or(1280);
    let height = height.unwrap_or(720);
    let fps = fps.unwrap_or(60.0);
    validate_dimensions(width, height)?;
    validate_fps(fps)?;
    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for STTF blind comparison")?;
    let mut runtime = Runtime::new(&context)?;
    runtime
        .load_sttf(path)
        .with_context(|| format!("failed to load STTF build {}", path.display()))?;

    let mut images = Vec::with_capacity(frames.len());
    let mut next = 0usize;
    let max_frame = *frames.last().expect("normalized frames are non-empty");
    for frame in 0..=max_frame {
        if frame > 0 {
            runtime.tick_fixed(1.0 / fps, fps)?;
        }
        let image = runtime.render(width, height)?;
        if frame == frames[next] {
            images.push(image);
            next += 1;
            if next == frames.len() {
                break;
            }
        }
    }
    Ok(images)
}

fn load_image(path: &Path) -> Result<RgbImage> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open blind image {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode blind image {}", path.display()))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    let mut pixels = image.into_raw();
    images::flip_rgb_rows(&mut pixels, width, height);
    Ok(RgbImage::new(width, height, pixels))
}

fn collect_images(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read blind source directory {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_images(&path, out)?;
        } else if file_type.is_file() && is_image_path(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "png" | "jpg" | "jpeg")
    )
}

fn normalize_frames(frames: &[i32]) -> Result<Vec<i32>> {
    if frames.is_empty() {
        bail!("--frames must contain at least one frame");
    }
    let mut frames = frames.to_vec();
    if frames.iter().any(|frame| *frame < 0) {
        bail!("--frames values must be non-negative");
    }
    frames.sort_unstable();
    frames.dedup();
    if frames.len() > MAX_BLIND_IMAGES_PER_SOURCE {
        bail!(
            "--frames accepts at most {MAX_BLIND_IMAGES_PER_SOURCE} unique entries for blind comparisons"
        );
    }
    Ok(frames)
}

fn resolve_git_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(root) = explicit {
        return Ok(root.to_path_buf());
    }
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("failed to run git while resolving blind git source")?;
    if !output.status.success() {
        bail!("git: blind sources require --git-root or running inside a Git repository");
    }
    let root = String::from_utf8(output.stdout).context("git root is not valid UTF-8")?;
    Ok(PathBuf::from(root.trim()))
}

fn materialize_git_source(
    repo_root: &Path,
    source: &str,
) -> Result<(String, PathBuf, GitWorktree)> {
    let spec = source
        .strip_prefix("git:")
        .context("internal blind git source parse error")?;
    let (git_ref, subdir) = spec.split_once("::").unwrap_or((spec, "."));
    if git_ref.trim().is_empty() {
        bail!("git blind source must name a revision, e.g. git:HEAD::path/to/project");
    }
    let subdir = Path::new(subdir);
    if subdir.is_absolute()
        || subdir
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        bail!("git blind source subdirectory must stay inside the worktree");
    }

    let parent = tempfile::tempdir().context("failed to create temporary git worktree parent")?;
    let checkout = parent.path().join("checkout");
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["worktree", "add", "--detach", "--quiet"])
        .arg(&checkout)
        .arg(git_ref)
        .output()
        .with_context(|| format!("failed to run git worktree for '{source}'"))?;
    if !output.status.success() {
        bail!(
            "failed to materialize git revision '{}': {}",
            git_ref,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let project = checkout.join(subdir);
    if !project.exists() {
        let _ = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["worktree", "remove", "--force"])
            .arg(&checkout)
            .status();
        bail!(
            "git blind source '{}' does not contain {}",
            git_ref,
            subdir.display()
        );
    }
    let identity = format!(
        "git:{}{}",
        git_ref,
        if subdir == Path::new(".") {
            String::new()
        } else {
            format!("::{}", subdir.display())
        }
    );
    Ok((
        identity,
        project,
        GitWorktree {
            repo_root: repo_root.to_path_buf(),
            checkout,
            _parent: parent,
        },
    ))
}
