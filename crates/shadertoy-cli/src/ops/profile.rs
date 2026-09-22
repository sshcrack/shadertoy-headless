use super::*;
use std::time::Instant;

const MAX_PROFILE_SAMPLES: u32 = 10_000;

#[derive(Default)]
struct TimingAggregate {
    width: u32,
    height: u32,
    samples: Vec<u64>,
}

pub fn profile_project(options: &ProfileOptions) -> Result<Output> {
    if options.samples == 0 || options.samples > MAX_PROFILE_SAMPLES {
        bail!("--samples must be in 1..={MAX_PROFILE_SAMPLES}");
    }
    if options.warmup > MAX_PROFILE_SAMPLES {
        bail!("--warmup must be at most {MAX_PROFILE_SAMPLES}");
    }

    let loaded = LoadedManifest::load_with_preset(&options.project, options.preset.as_deref())?;
    ensure_source_files_exist(&loaded)?;
    let media = crate::media::MediaInputs::new_headless(&loaded)?;
    let (width, height) = render::resolve_dimensions(&loaded, None, options.width, options.height)?;
    let fps = render::resolve_fps(&loaded, None, options.fps)?;
    let target_frame = resolve_target_frame(&loaded, None, options.frame, options.time, fps)?;

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for profiling")?;
    let mut runtime = Runtime::new(&context)?;
    let project = build_native_project(&loaded)?;
    runtime.load_project(&project)?;
    let uniform_values =
        crate::uniforms::parse_assignments(&loaded.manifest.uniforms, &options.set_uniforms)?;
    crate::uniforms::apply_to_runtime(&mut runtime, &uniform_values)?;

    let _ = render_from_zero(&mut runtime, target_frame, fps, width, height, &[], &media)?;
    for _ in 0..options.warmup {
        runtime.tick_fixed(1.0 / fps, fps)?;
        let media_time = runtime.time();
        media.update(&mut runtime, media_time)?;
        let _ = runtime.render(width, height)?;
    }

    runtime.set_profiling_mode(true, options.sync_per_pass)?;
    let first_sample_frame = runtime.frame().saturating_add(1);
    let mut cpu_samples = Vec::with_capacity(options.samples as usize);
    let mut gpu_total_samples = Vec::with_capacity(options.samples as usize);
    let mut passes: BTreeMap<String, TimingAggregate> = BTreeMap::new();

    for _ in 0..options.samples {
        runtime.tick_fixed(1.0 / fps, fps)?;
        let media_time = runtime.time();
        media.update(&mut runtime, media_time)?;
        let started = Instant::now();
        let _ = runtime.render(width, height)?;
        cpu_samples.push(started.elapsed().as_nanos() as u64);
        let timings = runtime.pass_timings()?;
        gpu_total_samples.push(runtime.frame_gpu_nanoseconds()?);
        for timing in timings {
            let aggregate = passes.entry(timing.name).or_default();
            aggregate.width = timing.width;
            aggregate.height = timing.height;
            aggregate.samples.push(timing.gpu_nanoseconds);
        }
    }
    runtime.set_profiling(false)?;

    let pass_rows = passes
        .iter()
        .map(|(name, aggregate)| {
            let stats = timing_stats(&aggregate.samples);
            (name.clone(), aggregate.width, aggregate.height, stats)
        })
        .collect::<Vec<_>>();
    let cpu = timing_stats(&cpu_samples);
    let gpu_total = timing_stats(&gpu_total_samples);
    let persistent_buffer_bytes = estimated_persistent_buffer_bytes(&loaded, width, height)?;

    let mut human = format!(
        "Profile: {}x{} @ {} fps, {} samples (frames {}..={})\n",
        width,
        height,
        fps,
        options.samples,
        first_sample_frame,
        runtime.frame()
    );
    let timing_mode = if options.sync_per_pass {
        "hard sync-per-pass GPU timestamps (maximum diagnostic isolation)"
    } else {
        "completion-synchronized GPU timestamps (precise pass attribution)"
    };
    human.push_str(&format!("Timing mode: {timing_mode}\n"));
    human.push_str(
        "Pass                         Resolution       mean     median        p95        min        max\n",
    );
    for (name, pass_width, pass_height, stats) in &pass_rows {
        human.push_str(&format!(
            "{name:<28} {pass_width:>5}x{pass_height:<5} {:>8.3}ms {:>8.3}ms {:>8.3}ms {:>8.3}ms {:>8.3}ms\n",
            stats.mean_ns / 1_000_000.0,
            stats.median_ns / 1_000_000.0,
            stats.p95_ns / 1_000_000.0,
            stats.min_ns as f64 / 1_000_000.0,
            stats.max_ns as f64 / 1_000_000.0,
        ));
    }
    human.push_str(&format!(
        "GPU total (sum of precisely attributed passes): mean {:.3} ms, median {:.3} ms, p95 {:.3} ms\nCPU profiled render call: mean {:.3} ms, median {:.3} ms, p95 {:.3} ms\nPersistent GPU state VRAM estimate: {:.2} MiB",
        gpu_total.mean_ns / 1_000_000.0,
        gpu_total.median_ns / 1_000_000.0,
        gpu_total.p95_ns / 1_000_000.0,
        cpu.mean_ns / 1_000_000.0,
        cpu.median_ns / 1_000_000.0,
        cpu.p95_ns / 1_000_000.0,
        persistent_buffer_bytes as f64 / (1024.0 * 1024.0),
    ));

    let json_passes = pass_rows
        .iter()
        .map(|(name, pass_width, pass_height, stats)| {
            json!({
                "name": name,
                "width": pass_width,
                "height": pass_height,
                "samples": stats.count,
                "mean_ms": stats.mean_ns / 1_000_000.0,
                "median_ms": stats.median_ns / 1_000_000.0,
                "p95_ms": stats.p95_ns / 1_000_000.0,
                "min_ms": stats.min_ns as f64 / 1_000_000.0,
                "max_ms": stats.max_ns as f64 / 1_000_000.0,
            })
        })
        .collect::<Vec<_>>();

    Ok(Output {
        human,
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "preset": options.preset,
            "width": width,
            "height": height,
            "fps": fps,
            "warmup": options.warmup,
            "samples": options.samples,
            "first_sample_frame": first_sample_frame,
            "last_sample_frame": runtime.frame(),
            "passes": json_passes,
            "timing_mode": if options.sync_per_pass { "sync_per_pass" } else { "completion_synchronized" },
            "sync_per_pass": options.sync_per_pass,
            "gpu_frame_mode": "attributed_pass_sum",
            "gpu_frame_note": "Portable whole-frame timer queries undercount asynchronous compute on some drivers. This total is the sum of the same completion-synchronized pass intervals used for precise attribution.",
            "gpu_frame": {
                "mean_ms": gpu_total.mean_ns / 1_000_000.0,
                "median_ms": gpu_total.median_ns / 1_000_000.0,
                "p95_ms": gpu_total.p95_ns / 1_000_000.0,
                "min_ms": gpu_total.min_ns as f64 / 1_000_000.0,
                "max_ms": gpu_total.max_ns as f64 / 1_000_000.0,
            },
            "gpu_pass_total_mean_ms": gpu_total.mean_ns / 1_000_000.0,
            "gpu_pass_total": {
                "mean_ms": gpu_total.mean_ns / 1_000_000.0,
                "median_ms": gpu_total.median_ns / 1_000_000.0,
                "p95_ms": gpu_total.p95_ns / 1_000_000.0,
                "min_ms": gpu_total.min_ns as f64 / 1_000_000.0,
                "max_ms": gpu_total.max_ns as f64 / 1_000_000.0,
            },
            "cpu_render": {
                "mean_ms": cpu.mean_ns / 1_000_000.0,
                "median_ms": cpu.median_ns / 1_000_000.0,
                "p95_ms": cpu.p95_ns / 1_000_000.0,
                "min_ms": cpu.min_ns as f64 / 1_000_000.0,
                "max_ms": cpu.max_ns as f64 / 1_000_000.0,
            },
            "estimated_persistent_gpu_state_vram_bytes": persistent_buffer_bytes,
        }),
    })
}

struct TimingStats {
    count: usize,
    mean_ns: f64,
    median_ns: f64,
    p95_ns: f64,
    min_ns: u64,
    max_ns: u64,
}

fn timing_stats(samples: &[u64]) -> TimingStats {
    let min_ns = samples.iter().copied().min().unwrap_or(0);
    let max_ns = samples.iter().copied().max().unwrap_or(0);
    let mean_ns = if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(|value| *value as f64).sum::<f64>() / samples.len() as f64
    };
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let median_ns = match sorted.len() {
        0 => 0.0,
        len if len % 2 == 1 => sorted[len / 2] as f64,
        len => (sorted[len / 2 - 1] as f64 + sorted[len / 2] as f64) / 2.0,
    };
    let p95_ns = if sorted.is_empty() {
        0.0
    } else {
        let rank = ((sorted.len() as f64 * 0.95).ceil() as usize).clamp(1, sorted.len());
        sorted[rank - 1] as f64
    };
    TimingStats {
        count: samples.len(),
        mean_ns,
        median_ns,
        p95_ns,
        min_ns,
        max_ns,
    }
}

fn estimated_persistent_buffer_bytes(
    loaded: &LoadedManifest,
    output_width: u32,
    output_height: u32,
) -> Result<u64> {
    let feedback_sources = loaded
        .manifest
        .passes
        .iter()
        .flat_map(|pass| &pass.inputs)
        .filter(|input| input.frame == crate::manifest::FrameRef::Previous)
        .map(|input| input.source.as_str())
        .collect::<std::collections::HashSet<_>>();

    let mut bytes = 0u64;
    for pass in &loaded.manifest.passes {
        if !matches!(pass.kind, PassKind::Buffer | PassKind::Compute) {
            continue;
        }
        let (width, height) = loaded
            .manifest
            .pass_dimensions(pass, output_width, output_height);
        let bytes_per_pixel = |format| match format {
            crate::manifest::RenderFormat::R32f => 4u64,
            crate::manifest::RenderFormat::Rg32f | crate::manifest::RenderFormat::Rgba16f => 8,
            crate::manifest::RenderFormat::Rgba32f => 16,
        };
        let target_bytes = std::iter::once(pass.format)
            .chain(pass.extra_outputs.iter().copied())
            .try_fold(0u64, |total, format| {
                total
                    .checked_add(bytes_per_pixel(format))
                    .context("persistent MRT VRAM estimate overflow")
            })?;
        let copies = if feedback_sources.contains(pass.name.as_str()) {
            2u64
        } else {
            1
        };
        let pass_bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(target_bytes))
            .and_then(|value| value.checked_mul(copies))
            .context("persistent buffer VRAM estimate overflow")?;
        bytes = bytes
            .checked_add(pass_bytes)
            .context("persistent buffer VRAM estimate overflow")?;
    }

    let mut storage = std::collections::HashMap::new();
    for pass in &loaded.manifest.passes {
        for buffer in &pass.storage {
            storage.entry(buffer.name.as_str()).or_insert(buffer.size);
        }
    }
    for size in storage.into_values() {
        bytes = bytes
            .checked_add(size)
            .context("persistent storage VRAM estimate overflow")?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_stats_handles_empty_and_values() {
        let empty = timing_stats(&[]);
        assert_eq!(empty.count, 0);
        assert_eq!(empty.mean_ns, 0.0);
        assert_eq!(empty.median_ns, 0.0);
        assert_eq!(empty.p95_ns, 0.0);

        let values = timing_stats(&[10, 20, 30, 40]);
        assert_eq!(values.min_ns, 10);
        assert_eq!(values.max_ns, 40);
        assert_eq!(values.mean_ns, 25.0);
        assert_eq!(values.median_ns, 25.0);
        assert_eq!(values.p95_ns, 40.0);
    }
}
