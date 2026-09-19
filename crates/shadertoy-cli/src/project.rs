use crate::manifest::{
    AssetKind, Filter, FrameRef, InputKind, LoadedManifest, PassKind, Wrap,
    validate_project_relative_path,
};
use anyhow::{Context, Result, bail};
use image::ImageReader;
use shadertoy::{
    Filter as NativeFilter, InputKind as NativeInputKind, PassKind as NativePassKind, Project,
    Wrap as NativeWrap,
};
use std::fs;
use std::path::Path;

pub fn build_native_project(loaded: &LoadedManifest) -> Result<Project> {
    let mut project = Project::new(&loaded.manifest.project.name)?;

    for asset in &loaded.manifest.assets {
        match asset.kind {
            AssetKind::Texture => {
                let path = existing_project_file(&loaded.root, &asset.path, "asset", &asset.name)?;
                let image = ImageReader::open(&path)
                    .with_context(|| format!("failed to open texture {}", path.display()))?
                    .decode()
                    .with_context(|| format!("failed to decode texture {}", path.display()))?
                    .to_rgba8();
                let (width, height) = image.dimensions();
                project.add_texture_rgba8(&asset.name, width, height, image.as_raw())?;
            }
        }
    }

    for pass in &loaded.manifest.passes {
        let source_path =
            existing_project_file(&loaded.root, &pass.source, "shader source", &pass.name)?;
        let source = fs::read_to_string(&source_path).with_context(|| {
            format!(
                "failed to read source for pass '{}' at {}",
                pass.name,
                source_path.display()
            )
        })?;
        project.add_pass(
            &pass.name,
            match pass.kind {
                PassKind::Image => NativePassKind::Image,
                PassKind::Buffer => NativePassKind::Buffer,
                PassKind::Cubemap => NativePassKind::Cubemap,
            },
            &source,
        )?;
    }

    for pass in &loaded.manifest.passes {
        for input in &pass.inputs {
            let kind = loaded.manifest.infer_input_kind(input)?;
            project.add_input(
                &pass.name,
                input.channel.into(),
                match kind {
                    InputKind::Pass => NativeInputKind::Pass,
                    InputKind::Texture => NativeInputKind::Texture,
                    InputKind::Keyboard => NativeInputKind::Keyboard,
                    InputKind::Music => NativeInputKind::Music,
                },
                &input.source,
                input.frame == FrameRef::Previous,
                match input.filter {
                    Filter::Mipmap => NativeFilter::Mipmap,
                    Filter::Linear => NativeFilter::Linear,
                    Filter::Nearest => NativeFilter::Nearest,
                },
                match input.wrap {
                    Wrap::Clamp => NativeWrap::Clamp,
                    Wrap::Repeat => NativeWrap::Repeat,
                },
            )?;
        }
    }

    Ok(project)
}

pub fn ensure_source_files_exist(loaded: &LoadedManifest) -> Result<()> {
    for pass in &loaded.manifest.passes {
        existing_project_file(&loaded.root, &pass.source, "shader source", &pass.name)?;
    }
    for asset in &loaded.manifest.assets {
        existing_project_file(&loaded.root, &asset.path, "asset", &asset.name)?;
    }
    Ok(())
}

fn existing_project_file(
    root: &Path,
    relative: &str,
    kind: &str,
    name: &str,
) -> Result<std::path::PathBuf> {
    validate_project_relative_path(relative, &format!("{kind} path for '{name}'"))?;
    let path = root.join(relative);
    if !path.is_file() {
        bail!("{kind} '{name}' does not exist at {}", path.display());
    }

    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("failed to resolve project root {}", root.display()))?;
    let canonical_path = fs::canonicalize(&path)
        .with_context(|| format!("failed to resolve {kind} {}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        bail!(
            "{kind} '{name}' resolves outside the project root: {}",
            path.display()
        );
    }
    Ok(canonical_path)
}
