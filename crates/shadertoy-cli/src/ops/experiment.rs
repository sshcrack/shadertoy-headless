use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::process::Command;

const MAX_EXPERIMENT_SOURCES: usize = 16;

#[derive(Debug, Clone)]
pub struct ExperimentOptions {
    pub baseline: String,
    pub candidates: Vec<String>,
    pub output_dir: Option<PathBuf>,
    pub frames: Vec<i32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub metrics: Vec<String>,
    pub blind: bool,
    pub git_root: Option<PathBuf>,
    pub profile_samples: u32,
}

#[derive(Debug, Clone, Serialize)]
struct SourceMetadata {
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    manifest_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<serde_json::Value>,
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct FrameMetrics {
    frame_index: usize,
    rmse: Option<f64>,
    ssim: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateMetrics {
    rmse_mean: Option<f64>,
    rmse_max: Option<f64>,
    ssim_mean: Option<f64>,
    ssim_min: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct PairComparison {
    left: String,
    right: String,
    frames: Vec<FrameMetrics>,
    aggregate: AggregateMetrics,
}

struct PreparedExperimentSource {
    prepared: super::blind_create::PreparedSource,
    git_commit: Option<String>,
    manifest_sha256: Option<String>,
    profile: Option<serde_json::Value>,
}

pub fn run_experiment(options: &ExperimentOptions) -> Result<Output> {
    let mut sources = Vec::with_capacity(1 + options.candidates.len());
    sources.push(options.baseline.clone());
    sources.extend(options.candidates.iter().cloned());
    if !(2..=MAX_EXPERIMENT_SOURCES).contains(&sources.len()) {
        bail!(
            "experiment requires 2..={MAX_EXPERIMENT_SOURCES} sources (got {})",
            sources.len()
        );
    }
    if options.profile_samples > 10_000 {
        bail!("--profile-samples must be at most 10000");
    }
    let metrics = normalize_metrics(&options.metrics)?;
    let frames = super::blind_create::normalize_frames(&options.frames)?;
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/experiment"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let git_root = if sources.iter().any(|source| source.starts_with("git:")) {
        Some(super::blind_create::resolve_git_root(
            options.git_root.as_deref(),
        )?)
    } else {
        None
    };
    let implicit_git_subdir = git_root
        .as_deref()
        .map(super::blind_create::current_project_subdir)
        .transpose()?
        .flatten();

    let mut worktrees = Vec::new();
    let mut prepared = Vec::with_capacity(sources.len());
    for source in &sources {
        if source.starts_with("git:") {
            let root = git_root.as_deref().expect("git root resolved above");
            let (identity, project, worktree) = super::blind_create::materialize_git_source(
                root,
                source,
                implicit_git_subdir.as_deref(),
            )?;
            let images = super::blind_create::prepare_path_source(
                &project,
                &frames,
                options.width,
                options.height,
                options.fps,
                None,
            )?;
            let profile = profile_project_source(&project, None, &frames, options)?;
            let git_commit = git_commit(&project).ok();
            let manifest_sha256 = manifest_hash(&project);
            prepared.push(PreparedExperimentSource {
                prepared: super::blind_create::PreparedSource {
                    identity,
                    images,
                    normalize_dimensions: true,
                },
                git_commit,
                manifest_sha256,
                profile,
            });
            worktrees.push(worktree);
        } else {
            let spec = super::blind_create::parse_path_source(source)?;
            let images = super::blind_create::prepare_path_source(
                &spec.path,
                &frames,
                options.width,
                options.height,
                options.fps,
                spec.preset.as_deref(),
            )?;
            let is_direct_project = spec.path.join("ShaderToy.toml").is_file();
            let profile = if spec.explicit_project || is_direct_project {
                profile_project_source(&spec.path, spec.preset.as_deref(), &frames, options)?
            } else {
                None
            };
            let manifest_sha256 = if spec.explicit_project || is_direct_project {
                manifest_hash(&spec.path)
            } else {
                None
            };
            prepared.push(PreparedExperimentSource {
                prepared: super::blind_create::PreparedSource {
                    identity: source.clone(),
                    images,
                    normalize_dimensions: spec.explicit_project,
                },
                git_commit: None,
                manifest_sha256,
                profile,
            });
        }
    }

    let image_count = prepared
        .first()
        .map(|source| source.prepared.images.len())
        .unwrap_or(0);
    if image_count == 0 {
        bail!("experiment sources produced no images");
    }
    for source in &prepared {
        if source.prepared.images.len() != image_count {
            bail!(
                "experiment sources must contain the same number of images; '{}' has {}, expected {}",
                source.prepared.identity,
                source.prepared.images.len(),
                image_count
            );
        }
    }
    let mut normalize = prepared
        .iter()
        .map(|source| source.prepared.clone())
        .collect::<Vec<_>>();
    super::blind_create::normalize_project_source_dimensions(&mut normalize)?;
    for (source, normalized) in prepared.iter_mut().zip(normalize) {
        source.prepared.images = normalized.images;
    }
    let width = prepared[0].prepared.images[0].width;
    let height = prepared[0].prepared.images[0].height;

    let plan = if options.blind {
        Some(super::blind::BlindPlan::new(
            prepared.len(),
            &format!("experiment|{}|{}", output_dir.display(), sources.join("|")),
        )?)
    } else {
        None
    };
    let order = plan
        .as_ref()
        .map(|plan| plan.order().to_vec())
        .unwrap_or_else(|| (0..prepared.len()).collect());
    let labels = if let Some(plan) = &plan {
        (0..prepared.len())
            .map(|index| plan.label(index).to_string())
            .collect::<Vec<_>>()
    } else {
        (0..prepared.len())
            .map(|index| match index {
                0 => "baseline".to_string(),
                1 => "candidate".to_string(),
                other => format!("variant-{other}"),
            })
            .collect::<Vec<_>>()
    };

    let variants_dir = output_dir.join("variants");
    if variants_dir.exists() {
        fs::remove_dir_all(&variants_dir)
            .with_context(|| format!("failed to clear {}", variants_dir.display()))?;
    }
    fs::create_dir_all(&variants_dir)?;
    let contact_path = output_dir.join("contact-sheet.png");
    let cell_count = prepared
        .len()
        .checked_mul(image_count)
        .context("experiment contact-sheet cell count overflow")?;
    let mut sheet = render::prepare_contact_sheet(
        &contact_path,
        cell_count,
        Some(prepared.len() as u32),
        width,
        height,
    )?;

    let mut outputs = Vec::with_capacity(prepared.len());
    let mut public_sources = Vec::with_capacity(prepared.len());
    for (position, original_index) in order.iter().copied().enumerate() {
        let label = &labels[position];
        let output = variants_dir.join(label);
        fs::create_dir_all(&output)?;
        for (image_index, image) in prepared[original_index].prepared.images.iter().enumerate() {
            let path = output.join(format!("image-{image_index:03}.png"));
            save_rgb_png(image, &path)?;
            sheet.blit(image_index * prepared.len() + position, image)?;
        }
        outputs.push(output.clone());
        public_sources.push(SourceMetadata {
            label: label.clone(),
            source: (!options.blind).then(|| prepared[original_index].prepared.identity.clone()),
            git_commit: (!options.blind)
                .then(|| prepared[original_index].git_commit.clone())
                .flatten(),
            manifest_sha256: prepared[original_index].manifest_sha256.clone(),
            profile: prepared[original_index].profile.clone(),
            output,
        });
    }
    let contact_sheet = sheet.save()?;

    let comparisons = pairwise_comparisons(&prepared, &order, &labels, &metrics)?;

    let blind_session = if let Some(plan) = &plan {
        let variants = prepared
            .iter()
            .map(|source| vec![format!("source={}", source.prepared.identity)])
            .collect::<Vec<_>>();
        Some(super::blind::write_blind_session(
            plan,
            &super::blind::BlindSessionSpec {
                output_dir: &output_dir,
                project: "experiment",
                frame: *frames.first().unwrap_or(&0),
                pass: "external",
                width,
                height,
                contact_sheet: &contact_sheet,
                outputs: &outputs,
                variants: &variants,
            },
        )?)
    } else {
        None
    };

    let report_path = output_dir.join("experiment-report.json");
    let report = json!({
        "format": 1,
        "blind": options.blind,
        "frames": frames,
        "width": width,
        "height": height,
        "metrics": metrics,
        "sources": public_sources,
        "comparisons": comparisons,
        "contact_sheet": contact_sheet,
        "blind_session": blind_session,
    });
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("failed to write {}", report_path.display()))?;

    drop(worktrees);

    let mut human = format!(
        "Experiment: {} variants x {} image(s) -> {}\nContact sheet: {}\nReport: {}",
        prepared.len(),
        image_count,
        output_dir.display(),
        contact_sheet.display(),
        report_path.display()
    );
    if let Some(session) = &blind_session {
        human.push_str(&format!(
            "\nBlind session: {} (judge before reveal)",
            session.display()
        ));
    }
    for comparison in &comparisons {
        human.push_str(&format!("\n{} vs {}", comparison.left, comparison.right));
        if let Some(value) = comparison.aggregate.rmse_mean {
            human.push_str(&format!("  RMSE mean={value:.6}"));
        }
        if let Some(value) = comparison.aggregate.ssim_mean {
            human.push_str(&format!("  SSIM mean={value:.6}"));
        }
    }

    Ok(Output {
        human,
        json: json!({
            "ok": true,
            "output_dir": output_dir,
            "report": report_path,
            "contact_sheet": contact_sheet,
            "blind_session": blind_session,
            "variant_count": prepared.len(),
            "images_per_variant": image_count,
            "comparisons": comparisons,
        }),
    })
}

fn profile_project_source(
    project: &Path,
    preset: Option<&str>,
    frames: &[i32],
    options: &ExperimentOptions,
) -> Result<Option<serde_json::Value>> {
    if options.profile_samples == 0 {
        return Ok(None);
    }
    let profile = super::profile_project(&ProfileOptions {
        project: project.to_path_buf(),
        preset: preset.map(str::to_owned),
        width: options.width,
        height: options.height,
        fps: options.fps,
        frame: frames.first().copied(),
        time: None,
        warmup: 2,
        samples: options.profile_samples,
        sync_per_pass: false,
        set_uniforms: Vec::new(),
    })?;
    Ok(Some(profile.json))
}

fn pairwise_comparisons(
    prepared: &[PreparedExperimentSource],
    order: &[usize],
    labels: &[String],
    metrics: &[String],
) -> Result<Vec<PairComparison>> {
    let mut comparisons = Vec::new();
    for left in 0..order.len() {
        for right in (left + 1)..order.len() {
            let left_source = &prepared[order[left]].prepared;
            let right_source = &prepared[order[right]].prepared;
            let mut frames = Vec::with_capacity(left_source.images.len());
            let mut rmse_values = Vec::new();
            let mut ssim_values = Vec::new();
            for (frame_index, (a, b)) in left_source
                .images
                .iter()
                .zip(&right_source.images)
                .enumerate()
            {
                if a.width != b.width || a.height != b.height {
                    bail!("experiment metric inputs have different dimensions");
                }
                let rmse = metrics
                    .iter()
                    .any(|metric| metric == "rmse")
                    .then(|| normalized_rmse(&a.pixels, &b.pixels));
                let ssim = metrics
                    .iter()
                    .any(|metric| metric == "ssim")
                    .then(|| windowed_ssim(a, b));
                if let Some(value) = rmse {
                    rmse_values.push(value);
                }
                if let Some(value) = ssim {
                    ssim_values.push(value);
                }
                frames.push(FrameMetrics {
                    frame_index,
                    rmse,
                    ssim,
                });
            }
            comparisons.push(PairComparison {
                left: labels[left].clone(),
                right: labels[right].clone(),
                aggregate: AggregateMetrics {
                    rmse_mean: mean(&rmse_values),
                    rmse_max: rmse_values.iter().copied().reduce(f64::max),
                    ssim_mean: mean(&ssim_values),
                    ssim_min: ssim_values.iter().copied().reduce(f64::min),
                },
                frames,
            });
        }
    }
    Ok(comparisons)
}

fn normalize_metrics(metrics: &[String]) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    for metric in metrics {
        let metric = metric.trim().to_ascii_lowercase();
        if metric.is_empty() || normalized.contains(&metric) {
            continue;
        }
        match metric.as_str() {
            "rmse" | "ssim" => normalized.push(metric),
            _ => bail!("unknown experiment metric '{metric}'; supported metrics: rmse, ssim"),
        }
    }
    if normalized.is_empty() {
        bail!("experiment requires at least one metric");
    }
    Ok(normalized)
}

fn normalized_rmse(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let delta = f64::from(*a) - f64::from(*b);
            delta * delta
        })
        .sum::<f64>();
    (sum / a.len().max(1) as f64).sqrt() / 255.0
}

fn windowed_ssim(a: &RgbImage, b: &RgbImage) -> f64 {
    debug_assert_eq!((a.width, a.height), (b.width, b.height));
    const WINDOW: u32 = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    let mut total = 0.0;
    let mut windows = 0u64;
    for y0 in (0..a.height).step_by(WINDOW as usize) {
        for x0 in (0..a.width).step_by(WINDOW as usize) {
            let y1 = (y0 + WINDOW).min(a.height);
            let x1 = (x0 + WINDOW).min(a.width);
            let mut left = Vec::new();
            let mut right = Vec::new();
            for y in y0..y1 {
                for x in x0..x1 {
                    left.push(luma(a, x, y));
                    right.push(luma(b, x, y));
                }
            }
            let n = left.len();
            if n == 0 {
                continue;
            }
            let mean_a = left.iter().sum::<f64>() / n as f64;
            let mean_b = right.iter().sum::<f64>() / n as f64;
            let mut var_a = 0.0;
            let mut var_b = 0.0;
            let mut covariance = 0.0;
            for (left, right) in left.iter().zip(&right) {
                let da = *left - mean_a;
                let db = *right - mean_b;
                var_a += da * da;
                var_b += db * db;
                covariance += da * db;
            }
            let denominator = (n.saturating_sub(1)).max(1) as f64;
            var_a /= denominator;
            var_b /= denominator;
            covariance /= denominator;
            let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * covariance + C2);
            let divisor = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
            total += if divisor == 0.0 {
                1.0
            } else {
                numerator / divisor
            };
            windows += 1;
        }
    }
    if windows == 0 {
        1.0
    } else {
        total / windows as f64
    }
}

fn luma(image: &RgbImage, x: u32, y: u32) -> f64 {
    let offset = ((y as usize * image.width as usize) + x as usize) * 3;
    let r = f64::from(image.pixels[offset]) / 255.0;
    let g = f64::from(image.pixels[offset + 1]) / 255.0;
    let b = f64::from(image.pixels[offset + 2]) / 255.0;
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn manifest_hash(project: &Path) -> Option<String> {
    let path = project.join(crate::manifest::MANIFEST_NAME);
    let data = fs::read(path).ok()?;
    Some(hex_sha256(&data))
}

fn git_commit(project: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(project)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("failed to resolve experiment git commit")?;
    if !output.status.success() {
        bail!("git rev-parse failed for {}", project.display());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn hex_sha256(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_identity_for_equal_images() {
        let image = RgbImage::new(8, 8, vec![128; 8 * 8 * 3]);
        assert_eq!(normalized_rmse(&image.pixels, &image.pixels), 0.0);
        assert!((windowed_ssim(&image, &image) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn metric_validation_is_strict() {
        assert_eq!(
            normalize_metrics(&["RMSE".into(), "ssim".into()]).unwrap(),
            vec!["rmse", "ssim"]
        );
        assert!(normalize_metrics(&["lpips".into()]).is_err());
    }
}
