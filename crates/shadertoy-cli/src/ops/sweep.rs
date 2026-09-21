use super::*;
use crate::uniforms::UniformDefinition;
use std::collections::HashSet;

const MAX_SWEEP_VARIANTS: usize = 256;

#[derive(Debug, Clone)]
struct SweepDimension {
    name: String,
    values: Vec<String>,
}

pub fn sweep_project(options: &SweepOptions) -> Result<Output> {
    let loaded = LoadedManifest::load(&options.project)?;
    ensure_source_files_exist(&loaded)?;
    let media = crate::media::MediaInputs::new_headless(&loaded)?;
    let dimensions = parse_sweep_dimensions(&loaded.manifest.uniforms, &options.sweep_uniforms)?;
    let variants = expand_variants(&dimensions)?;
    let (width, height) = render::resolve_dimensions(&loaded, None, options.width, options.height)?;
    let fps = render::resolve_fps(&loaded, None, options.fps)?;
    let target_frame = resolve_target_frame(&loaded, None, options.frame, options.time, fps)?;

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
    if matches!(pass.kind, PassKind::Cubemap | PassKind::Sound) {
        bail!("sweep only supports the final image and 2D buffer/compute passes");
    }
    let (selected_width, selected_height) = loaded.manifest.pass_dimensions(pass, width, height);

    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| loaded.root.join("target/sweep"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let contact_path = if options.no_contact_sheet {
        None
    } else {
        Some(
            options
                .contact_sheet
                .clone()
                .unwrap_or_else(|| output_dir.join("contact-sheet.png")),
        )
    };
    let mut contact_sheet = contact_path
        .as_ref()
        .map(|path| {
            render::prepare_contact_sheet(
                path,
                variants.len(),
                options.columns,
                selected_width,
                selected_height,
            )
        })
        .transpose()?;

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for parameter sweep")?;
    let project = build_native_project(&loaded)?;
    let final_pass = loaded.manifest.final_pass().name.as_str();
    let mut outputs = Vec::with_capacity(variants.len());

    for (index, assignments) in variants.iter().enumerate() {
        let mut runtime = Runtime::new(&context)?;
        runtime.load_project(&project)?;
        let values = crate::uniforms::parse_assignments(&loaded.manifest.uniforms, assignments)?;
        crate::uniforms::apply_to_runtime(&mut runtime, &values)?;
        let final_image =
            render_from_zero(&mut runtime, target_frame, fps, width, height, &[], &media)?
                .context("sweep render did not produce a final image")?;
        let image = if selected_pass == final_pass {
            final_image
        } else {
            runtime.snapshot_pass_rgb(selected_pass, selected_width, selected_height)?
        };
        let path = output_dir.join(format!("variant-{index:03}.png"));
        save_rgb_png(&image, &path)?;
        if let Some(sheet) = &mut contact_sheet {
            sheet.blit(index, &image)?;
        }
        outputs.push(json!({
            "index": index,
            "output": path,
            "set": assignments,
        }));
    }

    let contact_sheet_output = if let Some(sheet) = contact_sheet {
        Some(sheet.save()?)
    } else {
        None
    };
    let names = dimensions
        .iter()
        .map(|dimension| dimension.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    Ok(Output {
        human: format!(
            "Rendered {} sweep variants for [{}] at frame {}{} -> {}{}",
            variants.len(),
            names,
            target_frame,
            options
                .pass
                .as_ref()
                .map(|name| format!(" pass '{name}'"))
                .unwrap_or_default(),
            output_dir.display(),
            contact_sheet_output
                .as_ref()
                .map(|path| format!("; contact sheet {}", path.display()))
                .unwrap_or_default(),
        ),
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "output_dir": output_dir,
            "contact_sheet": contact_sheet_output,
            "width": selected_width,
            "height": selected_height,
            "fps": fps,
            "frame": target_frame,
            "pass": selected_pass,
            "variant_count": variants.len(),
            "variants": outputs,
        }),
    })
}

fn parse_sweep_dimensions(
    definitions: &[UniformDefinition],
    specs: &[String],
) -> Result<Vec<SweepDimension>> {
    if specs.is_empty() {
        bail!("sweep requires at least one --set NAME=VALUE1,VALUE2,... assignment");
    }
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(specs.len());
    for spec in specs {
        let (name, raw) = spec
            .split_once('=')
            .with_context(|| format!("sweep assignment '{spec}' must use NAME=VALUES"))?;
        let name = name.trim();
        if !seen.insert(name.to_string()) {
            bail!("sweep uniform '{name}' was specified more than once");
        }
        let definition = definitions
            .iter()
            .find(|definition| definition.name() == name)
            .with_context(|| format!("unknown custom uniform '{name}'"))?;
        let values = split_values(definition, raw)?;
        result.push(SweepDimension {
            name: name.to_string(),
            values,
        });
    }
    Ok(result)
}

fn split_values(definition: &UniformDefinition, raw: &str) -> Result<Vec<String>> {
    let parts = match definition {
        UniformDefinition::Vec2 { .. }
        | UniformDefinition::Vec3 { .. }
        | UniformDefinition::Vec4 { .. } => raw.split(';').collect::<Vec<_>>(),
        _ => raw.split(',').collect::<Vec<_>>(),
    };
    let mut values = Vec::with_capacity(parts.len());
    for value in parts {
        let value = value.trim();
        if value.is_empty() {
            bail!(
                "sweep uniform '{}' contains an empty value",
                definition.name()
            );
        }
        definition.parse_value(value).map_err(|error| {
            anyhow::anyhow!(
                "invalid sweep value for uniform '{}': {error:#}",
                definition.name()
            )
        })?;
        values.push(value.to_string());
    }
    Ok(values)
}

fn expand_variants(dimensions: &[SweepDimension]) -> Result<Vec<Vec<String>>> {
    let count = dimensions.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(dimension.values.len())
            .context("sweep variant count overflow")
    })?;
    if count > MAX_SWEEP_VARIANTS {
        bail!(
            "sweep expands to {count} variants; reduce values or dimensions (limit {MAX_SWEEP_VARIANTS})"
        );
    }

    let mut variants = vec![Vec::with_capacity(dimensions.len())];
    for dimension in dimensions {
        let mut next = Vec::with_capacity(variants.len() * dimension.values.len());
        for variant in &variants {
            for value in &dimension.values {
                let mut expanded = variant.clone();
                expanded.push(format!("{}={value}", dimension.name));
                next.push(expanded);
            }
        }
        variants = next;
    }
    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_cartesian_product_and_supports_vector_semicolons() {
        let definitions = vec![
            UniformDefinition::Float {
                name: "gain".into(),
                default: 1.0,
                min: None,
                max: None,
                step: None,
            },
            UniformDefinition::Vec2 {
                name: "wind".into(),
                default: [1.0, 0.0],
                min: None,
                max: None,
                step: None,
            },
        ];
        let dimensions = parse_sweep_dimensions(
            &definitions,
            &["gain=0.5,1.0".into(), "wind=1,0;0,1".into()],
        )
        .unwrap();
        let variants = expand_variants(&dimensions).unwrap();
        assert_eq!(variants.len(), 4);
        assert_eq!(variants[0], ["gain=0.5", "wind=1,0"]);
        assert_eq!(variants[3], ["gain=1.0", "wind=0,1"]);
    }
}
