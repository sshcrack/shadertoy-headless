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

fn run_case_inner(
    context: &HeadlessContext,
    loaded: &LoadedManifest,
    test: &TestCase,
    update: bool,
) -> Result<CaseReport> {
    let width = test.width.unwrap_or(loaded.manifest.render.width);
    let height = test.height.unwrap_or(loaded.manifest.render.height);
    render::validate_dimensions(width, height)?;
    let fps = loaded.manifest.render.fps;
    render::validate_fps(fps)?;
    let frame = resolve_target_frame(loaded, None, test.frame, test.time, fps)?;

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
    let (pass_width, pass_height) = loaded.manifest.pass_dimensions(pass, width, height);

    let mut runtime = Runtime::new(context)?;
    let project = build_native_project(loaded)?;
    runtime.load_project(&project)?;
    let final_image = render_from_zero(&mut runtime, frame, fps, width, height, &[])?
        .context("test render did not produce a final image")?;
    let image = if pass.name == loaded.manifest.final_pass().name {
        final_image
    } else {
        runtime.snapshot_pass_rgb(&pass.name, pass_width, pass_height)?
    };

    let numeric_requested = test.assert_no_nan || test.assert_no_inf || test.mean_range.is_some();
    let numeric = if numeric_requested {
        if pass.kind != PassKind::Buffer {
            bail!(
                "numeric assertions require a buffer pass; '{}' is {:?}",
                pass.name,
                pass.kind
            );
        }
        Some(runtime.snapshot_pass_rgba32f(&pass.name, pass_width, pass_height)?)
    } else {
        None
    };

    let mut reasons = Vec::new();
    if let Some(values) = &numeric {
        let nan = values.iter().filter(|value| value.is_nan()).count();
        let inf = values.iter().filter(|value| value.is_infinite()).count();
        if test.assert_no_nan && nan != 0 {
            reasons.push(format!("found {nan} NaN values"));
        }
        if test.assert_no_inf && inf != 0 {
            reasons.push(format!("found {inf} infinite values"));
        }
        if let Some([min, max]) = test.mean_range {
            let mut sum = 0.0f64;
            let mut count = 0u64;
            for value in values.iter().copied().filter(|value| value.is_finite()) {
                sum += f64::from(value);
                count += 1;
            }
            if count == 0 {
                reasons.push("mean_range has no finite values to inspect".into());
            } else {
                let mean = sum / count as f64;
                if mean < f64::from(min) || mean > f64::from(max) {
                    reasons.push(format!(
                        "finite RGBA mean {mean:.6} is outside [{min:.6}, {max:.6}]"
                    ));
                }
            }
        }
    }

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
            save_rgb_png(&image, &reference_path)?;
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
            if expected.width() != image.width || expected.height() != image.height {
                reasons.push(format!(
                    "reference is {}x{} but actual is {}x{}",
                    expected.width(),
                    expected.height(),
                    image.width,
                    image.height
                ));
            } else {
                let actual = top_down_rgb(&image);
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
                    save_rgb_png(&image, &actual_path)?;
                    artifacts.push(actual_path);

                    let expected_path = artifact_dir.join("expected.png");
                    fs::copy(&reference_path, &expected_path)?;
                    artifacts.push(expected_path);

                    let mut diff_bottom_up = comparison.diff;
                    super::images::flip_rgb_rows(&mut diff_bottom_up, image.width, image.height);
                    let diff_path = artifact_dir.join("diff.png");
                    save_rgb_png(
                        &RgbImage::new(image.width, image.height, diff_bottom_up),
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
        frame,
        passed: reasons.is_empty(),
        reasons,
        rmse,
        max_error,
        changed_fraction,
        artifacts,
        reference_updated,
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

    for (expected_pixel, actual_pixel) in expected.chunks_exact(3).zip(actual.chunks_exact(3)) {
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
