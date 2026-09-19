use crate::manifest::Manifest;
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    Minimal,
    Multipass,
}

pub fn create_project(
    root: &Path,
    name: &str,
    template: Template,
    allow_existing: bool,
) -> Result<()> {
    if root.exists() && !allow_existing && root.read_dir()?.next().is_some() {
        bail!(
            "destination {} already exists and is not empty",
            root.display()
        );
    }
    if root.join("ShaderToy.toml").exists() {
        bail!("{} already contains ShaderToy.toml", root.display());
    }

    fs::create_dir_all(root.join("shaders"))?;
    fs::create_dir_all(root.join("assets"))?;
    fs::create_dir_all(root.join(".shadertoy"))?;

    let manifest = match template {
        Template::Minimal => Manifest::minimal(name),
        Template::Multipass => Manifest::multipass(name),
    };

    for pass in &manifest.passes {
        let path = root.join(&pass.source);
        if path.exists() {
            bail!(
                "refusing to overwrite existing shader source {} while initializing project",
                path.display()
            );
        }
    }

    write_manifest(root, &manifest)?;
    fs::write(
        root.join(".shadertoy/shadertoy.schema.json"),
        crate::include_file!("schema/shadertoy.schema.json"),
    )
    .context("failed to write project-local manifest schema")?;

    let taplo = root.join(".taplo.toml");
    if !taplo.exists() {
        fs::write(taplo, crate::include_file!("templates/project/.taplo.toml"))?;
    }

    ensure_gitignore(root)?;
    let readme = root.join("README.md");
    if !readme.exists() {
        fs::write(readme, project_readme(name, template))?;
    }

    match template {
        Template::Minimal => fs::write(
            root.join("shaders/image.frag"),
            crate::include_file!("templates/minimal/image.frag"),
        )?,
        Template::Multipass => {
            fs::write(
                root.join("shaders/buffer-a.frag"),
                crate::include_file!("templates/multipass/buffer-a.frag"),
            )?;
            fs::write(
                root.join("shaders/image.frag"),
                crate::include_file!("templates/multipass/image.frag"),
            )?;
        }
    }

    Ok(())
}

fn ensure_gitignore(root: &Path) -> Result<()> {
    let path = root.join(".gitignore");
    if !path.exists() {
        fs::write(path, "target/\n")?;
        return Ok(());
    }

    let mut contents = fs::read_to_string(&path)?;
    if !contents.lines().any(|line| line.trim() == "target/") {
        if !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str("target/\n");
        fs::write(path, contents)?;
    }
    Ok(())
}

pub fn write_manifest(root: &Path, manifest: &Manifest) -> Result<()> {
    manifest.validate_structure()?;
    let body = manifest.to_pretty_toml()?;
    let document = format!(
        "#:schema ./.shadertoy/shadertoy.schema.json\n# Generated/validated by the shadertoy CLI.\n{body}"
    );
    fs::write(root.join("ShaderToy.toml"), document)
        .with_context(|| format!("failed to write {}/ShaderToy.toml", root.display()))
}

fn project_readme(name: &str, template: Template) -> String {
    let template_note = match template {
        Template::Minimal => "This project starts with one final `image` pass.",
        Template::Multipass => {
            "This project demonstrates previous-frame Buffer A feedback and wiring Buffer A into the final Image pass through `iChannel0`."
        }
    };

    crate::include_file!("templates/project/README.md")
        .replace("{{name}}", name)
        .replace("{{template_note}}", template_note)
}
