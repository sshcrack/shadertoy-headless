use super::*;

pub fn new_project(path: &Path, template: Template) -> Result<Output> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .context("project path must end in a valid UTF-8 directory name")?;
    create_project(path, name, template, false)?;
    Ok(Output {
        human: format!("Created ShaderToy project {}", path.display()),
        json: json!({
            "ok": true,
            "action": "new",
            "project": path,
            "name": name,
            "template": template_name(template),
        }),
    })
}

pub fn init_project(path: &Path, template: Template) -> Result<Output> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let name = absolute
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .context("project directory must have a valid UTF-8 name")?;
    create_project(&absolute, name, template, true)?;
    Ok(Output {
        human: format!("Initialized ShaderToy project {}", absolute.display()),
        json: json!({
            "ok": true,
            "action": "init",
            "project": absolute,
            "name": name,
            "template": template_name(template),
        }),
    })
}

pub fn check_project(path: &Path) -> Result<Output> {
    let loaded = LoadedManifest::load(path)?;
    ensure_source_files_exist(&loaded)?;

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for shader compilation")?;
    let mut runtime = Runtime::new(&context)?;
    let project = build_native_project(&loaded)?;
    runtime.load_project(&project)?;

    Ok(Output {
        human: format!(
            "OK: {} ({} passes, {} assets)",
            loaded.manifest.project.name,
            loaded.manifest.passes.len(),
            loaded.manifest.assets.len()
        ),
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "root": loaded.root,
            "passes": loaded.manifest.passes.len(),
            "assets": loaded.manifest.assets.len(),
            "compiled": true,
        }),
    })
}

pub fn build_project(path: &Path, output: Option<&Path>) -> Result<Output> {
    let loaded = LoadedManifest::load(path)?;
    ensure_source_files_exist(&loaded)?;
    let output = output.map(PathBuf::from).unwrap_or_else(|| {
        loaded.root.join("target").join(format!(
            "{}.sttf",
            file_safe_name(&loaded.manifest.project.name)
        ))
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let context = HeadlessContext::new(64, 64)
        .context("failed to create headless OpenGL context for shader compilation")?;
    let mut runtime = Runtime::new(&context)?;
    let project = build_native_project(&loaded)?;
    runtime.load_project(&project)?;
    runtime.save_sttf(&output)?;

    Ok(Output {
        human: format!("Built {}", output.display()),
        json: json!({
            "ok": true,
            "project": loaded.manifest.project.name,
            "artifact": output,
            "format": "sttf",
        }),
    })
}

fn template_name(template: Template) -> &'static str {
    match template {
        Template::Minimal => "minimal",
        Template::Multipass => "multipass",
    }
}

fn file_safe_name(value: &str) -> String {
    let mut result = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
            result.push(character);
        } else if !result.ends_with('-') {
            result.push('-');
        }
    }
    result.trim_matches('-').to_string()
}
