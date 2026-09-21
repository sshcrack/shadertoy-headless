use super::*;
use crate::manifest::TestCase;

#[derive(Debug, serde::Serialize)]
struct CaseReport {
    name: String,
    pass: String,
    frame: i32,
    passed: bool,
    reasons: Vec<String>,
    rmse: Option<f64>,
    max_error: Option<f64>,
    changed_fraction: Option<f64>,
    artifacts: Vec<PathBuf>,
    reference_updated: bool,
}

pub fn test_project(options: &TestOptions) -> Result<Output> {
    let loaded = LoadedManifest::load(&options.project)?;
    ensure_source_files_exist(&loaded)?;

    let selected = loaded
        .manifest
        .tests
        .iter()
        .filter(|test| {
            options
                .filter
                .as_ref()
                .is_none_or(|filter| test.name.contains(filter))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!(
            "no [[test]] cases{}",
            options
                .filter
                .as_ref()
                .map(|filter| format!(" match filter '{filter}'"))
                .unwrap_or_default()
        );
    }

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for regression tests")?;
    let mut reports = Vec::with_capacity(selected.len());
    for test in selected {
        reports.push(run_case(&context, &loaded, test, options.update));
    }

    let passed = reports.iter().filter(|report| report.passed).count();
    let failed = reports.len() - passed;
    let mut human = format!("Shader tests: {passed} passed, {failed} failed\n");
    for report in &reports {
        human.push_str(&format!(
            "{} {} (pass {}, frame {})",
            if report.passed { "PASS" } else { "FAIL" },
            report.name,
            report.pass,
            report.frame,
        ));
        if let Some(rmse) = report.rmse {
            human.push_str(&format!(", RMSE={rmse:.6}"));
        }
        if !report.reasons.is_empty() {
            human.push_str(&format!(": {}", report.reasons.join("; ")));
        }
        human.push('\n');
        for artifact in &report.artifacts {
            human.push_str(&format!("  {}\n", artifact.display()));
        }
    }

    Ok(Output {
        human: human.trim_end().to_string(),
        json: json!({
            "ok": failed == 0,
            "project": loaded.manifest.project.name,
            "passed": passed,
            "failed": failed,
            "updated": options.update,
            "cases": reports,
        }),
    })
}

fn run_case(
    context: &HeadlessContext,
    loaded: &LoadedManifest,
    test: &TestCase,
    update: bool,
) -> CaseReport {
    match run_case_inner(context, loaded, test, update) {
        Ok(report) => report,
        Err(error) => CaseReport {
            name: test.name.clone(),
            pass: test
                .pass
                .clone()
                .unwrap_or_else(|| loaded.manifest.final_pass().name.clone()),
            frame: test.frame.unwrap_or(-1),
            passed: false,
            reasons: vec![format!("{error:#}")],
            rmse: None,
            max_error: None,
            changed_fraction: None,
            artifacts: Vec::new(),
            reference_updated: false,
        },
    }
}

#[derive(Debug)]
struct RenderedVariant {
    frame: i32,
    width: u32,
    height: u32,
    image: RgbImage,
    raw: Option<Vec<f32>>,
}

fn run_case_inner(
    context: &HeadlessContext,
    loaded: &LoadedManifest,
    test: &TestCase,
    update: bool,
) -> Result<CaseReport> {
    let fps = loaded.manifest.render.fps;
    render::validate_fps(fps)?;
    let frames = test_frames(loaded, test, fps)?;
    let resolutions = test_resolutions(loaded, test);

    let selected_name = test
        .pass
        .as_deref()
        .unwrap_or(&loaded.manifest.final_pass().name);
    let pass = loaded
        .manifest
        .passes
        .iter()
        .find(|pass| pass.name == selected_name)
        .with_context(|| format!("unknown test pass '{selected_name}'"))?;
    if pass.kind == PassKind::Cubemap {
        bail!("test pass '{selected_name}' is a cubemap; only 2D passes are supported");
    }

    let uniform_values = crate::uniforms::merge_values(&loaded.manifest.uniforms, &test.uniforms)?;
    let raw_requested = matches!(pass.kind, PassKind::Buffer | PassKind::Compute)
        && (test.assert_no_nan
            || test.assert_no_inf
            || test.mean_range.is_some()
            || test.assert_deterministic
            || test.assert_resolution_independent);

    let mut variants = Vec::with_capacity(frames.len() * resolutions.len());
    for frame in &frames {
        for (width, height) in &resolutions {
            variants.push(render_variant(
                context,
                loaded,
                pass,
                *frame,
                fps,
                *width,
                *height,
                &uniform_values,
                raw_requested,
            )?);
        }
    }

    let mut reasons = Vec::new();
    for variant in &variants {
        if let Some(values) = &variant.raw {
            check_numeric_assertions(test, variant, values, &mut reasons);
        }
    }

    if test.assert_deterministic {
        for variant in &variants {
            let repeated = render_variant(
                context,
                loaded,
                pass,
                variant.frame,
                fps,
                variant.width,
                variant.height,
                &uniform_values,
                raw_requested,
            )?;
            if let (Some(first), Some(second)) = (&variant.raw, &repeated.raw) {
                let compared = compare_raw(first, second, test.raw_tolerance)?;
                if compared.mismatches != 0 {
                    reasons.push(format!(
                        "frame {} at {}x{} is not deterministic: {} raw values differ (max abs error {:.9})",
                        variant.frame,
                        variant.width,
                        variant.height,
                        compared.mismatches,
                        compared.max_error
                    ));
                }
            } else if variant.image.pixels != repeated.image.pixels {
                reasons.push(format!(
                    "frame {} at {}x{} is not deterministic: RGB output differs between fresh runs",
                    variant.frame, variant.width, variant.height
                ));
            }
        }
    }

    if test.assert_resolution_independent {
        for frame in &frames {
            let same_frame = variants
                .iter()
                .filter(|variant| variant.frame == *frame)
                .collect::<Vec<_>>();
            let baseline = same_frame
                .first()
                .and_then(|variant| variant.raw.as_ref())
                .context("resolution-independence test did not capture raw buffer data")?;
            for variant in same_frame.iter().skip(1) {
                let values = variant
                    .raw
                    .as_ref()
                    .context("resolution-independence test did not capture raw buffer data")?;
                let compared = compare_raw(baseline, values, test.raw_tolerance)?;
                if compared.mismatches != 0 {
                    let first = same_frame[0];
                    reasons.push(format!(
                        "frame {} fixed pass depends on output resolution: {}x{} vs {}x{} differ in {} raw values (max abs error {:.9})",
                        frame,
                        first.width,
                        first.height,
                        variant.width,
                        variant.height,
                        compared.mismatches,
                        compared.max_error
                    ));
                }
            }
        }
    }

    let baseline = variants
        .first()
        .context("regression test produced no frame/resolution variants")?;
    let artifact_dir = loaded
        .root
        .join("target/tests")
        .join(file_safe_name(&test.name));
    let mut artifacts = Vec::new();
    let mut rmse = None;
    let mut max_error = None;
    let mut changed_fraction = None;
    let mut reference_updated = false;

    if let Some(reference) = &test.reference {
        let reference_path = safe_reference_path(&loaded.root, reference, update)?;
        if update {
            if let Some(parent) = reference_path.parent() {
                fs::create_dir_all(parent)?;
            }
            save_rgb_png(&baseline.image, &reference_path)?;
            reference_updated = true;
        } else if !reference_path.is_file() {
            reasons.push(format!(
                "reference image is missing: {}",
                reference_path.display()
            ));
        } else {
            let expected = ImageReader::open(&reference_path)
                .with_context(|| format!("failed to open {}", reference_path.display()))?
                .decode()
                .with_context(|| format!("failed to decode {}", reference_path.display()))?
                .to_rgb8();
            if expected.width() != baseline.image.width
                || expected.height() != baseline.image.height
            {
                reasons.push(format!(
                    "reference is {}x{} but baseline variant is {}x{}",
                    expected.width(),
                    expected.height(),
                    baseline.image.width,
                    baseline.image.height
                ));
            } else {
                let actual = top_down_rgb(&baseline.image);
                let comparison = compare_rgb(expected.as_raw(), &actual, test.tolerance);
                rmse = Some(comparison.rmse);
                max_error = Some(comparison.max_error);
                changed_fraction = Some(comparison.changed_fraction);
                if comparison.rmse > f64::from(test.tolerance) {
                    reasons.push(format!(
                        "RMSE {:.6} exceeds tolerance {:.6}",
                        comparison.rmse, test.tolerance
                    ));
                }
                if !reasons.is_empty() {
                    fs::create_dir_all(&artifact_dir)?;
                    let actual_path = artifact_dir.join("actual.png");
                    save_rgb_png(&baseline.image, &actual_path)?;
                    artifacts.push(actual_path);

                    let expected_path = artifact_dir.join("expected.png");
                    fs::copy(&reference_path, &expected_path)?;
                    artifacts.push(expected_path);

                    let mut diff_bottom_up = comparison.diff;
                    super::images::flip_rgb_rows(
                        &mut diff_bottom_up,
                        baseline.image.width,
                        baseline.image.height,
                    );
                    let diff_path = artifact_dir.join("diff.png");
                    save_rgb_png(
                        &RgbImage::new(baseline.image.width, baseline.image.height, diff_bottom_up),
                        &diff_path,
                    )?;
                    artifacts.push(diff_path);
                }
            }
        }
    }

    Ok(CaseReport {
        name: test.name.clone(),
        pass: pass.name.clone(),
        frame: baseline.frame,
        passed: reasons.is_empty(),
        reasons,
        rmse,
        max_error,
        changed_fraction,
        artifacts,
        reference_updated,
    })
}

fn test_frames(loaded: &LoadedManifest, test: &TestCase, fps: f32) -> Result<Vec<i32>> {
    if !test.frames.is_empty() {
        let mut frames = test.frames.clone();
        frames.sort_unstable();
        return Ok(frames);
    }
    Ok(vec![resolve_target_frame(
        loaded, None, test.frame, test.time, fps,
    )?])
}

fn test_resolutions(loaded: &LoadedManifest, test: &TestCase) -> Vec<(u32, u32)> {
    if !test.resolutions.is_empty() {
        return test
            .resolutions
            .iter()
            .map(|[width, height]| (*width, *height))
            .collect();
    }
    vec![(
        test.width.unwrap_or(loaded.manifest.render.width),
        test.height.unwrap_or(loaded.manifest.render.height),
    )]
}

#[allow(clippy::too_many_arguments)]
fn render_variant(
    context: &HeadlessContext,
    loaded: &LoadedManifest,
    pass: &crate::manifest::Pass,
    frame: i32,
    fps: f32,
    width: u32,
    height: u32,
    uniform_values: &BTreeMap<String, crate::uniforms::UniformValue>,
    raw_requested: bool,
) -> Result<RenderedVariant> {
    render::validate_dimensions(width, height)?;
    let (pass_width, pass_height) = loaded.manifest.pass_dimensions(pass, width, height);
    let media = crate::media::MediaInputs::new_headless(loaded)?;
    let mut runtime = Runtime::new(context)?;
    let project = build_native_project(loaded)?;
    runtime.load_project(&project)?;
    crate::uniforms::apply_to_runtime(&mut runtime, uniform_values)?;
    let final_image = render_from_zero(&mut runtime, frame, fps, width, height, &[], &media)?
        .context("test render did not produce a final image")?;
    let image = if pass.name == loaded.manifest.final_pass().name {
        final_image
    } else {
        runtime.snapshot_pass_rgb(&pass.name, pass_width, pass_height)?
    };
    let raw = if raw_requested {
        Some(runtime.snapshot_pass_rgba32f(&pass.name, pass_width, pass_height)?)
    } else {
        None
    };
    Ok(RenderedVariant {
        frame,
        width,
        height,
        image,
        raw,
    })
}

fn check_numeric_assertions(
    test: &TestCase,
    variant: &RenderedVariant,
    values: &[f32],
    reasons: &mut Vec<String>,
) {
    let label = format!(
        "frame {} at {}x{}",
        variant.frame, variant.width, variant.height
    );
    let nan = values.iter().filter(|value| value.is_nan()).count();
    let inf = values.iter().filter(|value| value.is_infinite()).count();
    if test.assert_no_nan && nan != 0 {
        reasons.push(format!("{label}: found {nan} NaN values"));
    }
    if test.assert_no_inf && inf != 0 {
        reasons.push(format!("{label}: found {inf} infinite values"));
    }
    if let Some([min, max]) = test.mean_range {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for value in values.iter().copied().filter(|value| value.is_finite()) {
            sum += f64::from(value);
            count += 1;
        }
        if count == 0 {
            reasons.push(format!(
                "{label}: mean_range has no finite values to inspect"
            ));
        } else {
            let mean = sum / count as f64;
            if mean < f64::from(min) || mean > f64::from(max) {
                reasons.push(format!(
                    "{label}: finite RGBA mean {mean:.6} is outside [{min:.6}, {max:.6}]"
                ));
            }
        }
    }
}

struct RawComparison {
    mismatches: usize,
    max_error: f64,
}

fn compare_raw(expected: &[f32], actual: &[f32], tolerance: f32) -> Result<RawComparison> {
    if expected.len() != actual.len() {
        bail!(
            "raw regression buffers have different lengths ({} vs {})",
            expected.len(),
            actual.len()
        );
    }
    let tolerance = f64::from(tolerance);
    let mut mismatches = 0usize;
    let mut max_error = 0.0f64;
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.to_bits() == actual.to_bits() {
            continue;
        }
        if !expected.is_finite() || !actual.is_finite() {
            mismatches += 1;
            max_error = f64::INFINITY;
            continue;
        }
        let error = (f64::from(*expected) - f64::from(*actual)).abs();
        max_error = max_error.max(error);
        if error > tolerance {
            mismatches += 1;
        }
    }
    Ok(RawComparison {
        mismatches,
        max_error,
    })
}

struct Comparison {
    rmse: f64,
    max_error: f64,
    changed_fraction: f64,
    diff: Vec<u8>,
}

fn compare_rgb(expected: &[u8], actual: &[u8], tolerance: f32) -> Comparison {
    debug_assert_eq!(expected.len(), actual.len());
    let threshold = f64::from(tolerance) * 255.0;
    let mut squared = 0.0f64;
    let mut max = 0.0f64;
    let mut changed_pixels = 0usize;
    let mut diff = Vec::with_capacity(expected.len());

    for (expected_pixel, actual_pixel) in expected.as_chunks::<3>().0.iter().zip(actual.as_chunks::<3>().0.iter()) {
        let mut pixel_changed = false;
        for channel in 0..3 {
            let delta =
                (f64::from(expected_pixel[channel]) - f64::from(actual_pixel[channel])).abs();
            squared += delta * delta;
            max = max.max(delta);
            pixel_changed |= delta > threshold;
            diff.push(delta.min(255.0).round() as u8);
        }
        changed_pixels += usize::from(pixel_changed);
    }

    let values = expected.len().max(1) as f64;
    let pixels = (expected.len() / 3).max(1) as f64;
    Comparison {
        rmse: (squared / values).sqrt() / 255.0,
        max_error: max / 255.0,
        changed_fraction: changed_pixels as f64 / pixels,
        diff,
    }
}

fn top_down_rgb(image: &RgbImage) -> Vec<u8> {
    let mut pixels = image.pixels.clone();
    super::images::flip_rgb_rows(&mut pixels, image.width, image.height);
    pixels
}

fn safe_reference_path(root: &Path, relative: &str, create: bool) -> Result<PathBuf> {
    crate::manifest::validate_project_relative_path(relative, "test reference")?;
    let path = root.join(relative);
    let canonical_root = fs::canonicalize(root)?;
    if path.exists() {
        let canonical = fs::canonicalize(&path)?;
        if !canonical.starts_with(&canonical_root) {
            bail!(
                "test reference resolves outside the project root: {}",
                path.display()
            );
        }
    } else if create && let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
        let canonical_parent = fs::canonicalize(parent)?;
        if !canonical_parent.starts_with(&canonical_root) {
            bail!(
                "test reference parent resolves outside the project root: {}",
                parent.display()
            );
        }
    }
    Ok(path)
}

fn file_safe_name(value: &str) -> String {
    let result = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let trimmed = result.trim_matches('-');
    if trimmed.is_empty() {
        "test".into()
    } else {
        trimmed.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_reports_normalized_error() {
        let compared = compare_rgb(&[0, 0, 0], &[255, 0, 0], 0.0);
        assert!((compared.rmse - (1.0f64 / 3.0).sqrt()).abs() < 1e-9);
        assert_eq!(compared.max_error, 1.0);
        assert_eq!(compared.changed_fraction, 1.0);
    }
}
