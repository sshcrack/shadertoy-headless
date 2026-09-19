use super::*;

pub fn add_pass(
    project_path: &Path,
    name: &str,
    kind: PassKind,
    source: Option<&Path>,
) -> Result<Output> {
    use crate::manifest::Pass;
    use crate::scaffold::write_manifest;

    let mut loaded = LoadedManifest::load(project_path)?;
    if loaded.manifest.passes.iter().any(|pass| pass.name == name)
        || loaded
            .manifest
            .assets
            .iter()
            .any(|asset| asset.name == name)
    {
        bail!("pass/asset name '{}' already exists", name);
    }
    if kind == PassKind::Image {
        bail!(
            "project already has its required final image pass; add a buffer or cubemap pass instead"
        );
    }

    let source_rel = source
        .map(|path| path.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(format!("shaders/{}.frag", slug(name))));
    if source_rel.is_absolute() {
        bail!("pass source must be relative to the project root");
    }
    let source_abs = loaded.root.join(&source_rel);
    if !source_abs.exists() {
        if let Some(parent) = source_abs.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&source_abs, pass_stub(kind))?;
    }

    loaded.manifest.passes.insert(
        loaded.manifest.passes.len().saturating_sub(1),
        Pass {
            name: name.to_string(),
            kind,
            source: source_rel.to_string_lossy().replace('\\', "/"),
            inputs: Vec::new(),
        },
    );
    write_manifest(&loaded.root, &loaded.manifest)?;

    Ok(Output {
        human: format!(
            "Added {:?} pass '{}' -> {}",
            kind,
            name,
            source_rel.display()
        ),
        json: json!({
            "ok": true,
            "action": "pass-add",
            "pass": name,
            "kind": format!("{kind:?}").to_lowercase(),
            "source": source_rel,
        }),
    })
}

pub fn remove_pass(project_path: &Path, name: &str, force: bool) -> Result<Output> {
    use crate::scaffold::write_manifest;

    let mut loaded = LoadedManifest::load(project_path)?;
    let index = loaded
        .manifest
        .passes
        .iter()
        .position(|pass| pass.name == name)
        .with_context(|| format!("unknown pass '{name}'"))?;
    if loaded.manifest.passes[index].kind == PassKind::Image {
        bail!("cannot remove the required final image pass");
    }

    let consumers = loaded
        .manifest
        .passes
        .iter()
        .filter(|pass| pass.inputs.iter().any(|input| input.source == name))
        .map(|pass| pass.name.clone())
        .collect::<Vec<_>>();
    if !consumers.is_empty() && !force {
        bail!(
            "pass '{}' is used by {}; remove those channels first or use --force",
            name,
            consumers.join(", ")
        );
    }
    if force {
        for pass in &mut loaded.manifest.passes {
            pass.inputs.retain(|input| input.source != name);
        }
    }
    let removed = loaded.manifest.passes.remove(index);
    write_manifest(&loaded.root, &loaded.manifest)?;

    Ok(Output {
        human: format!("Removed pass '{}'", name),
        json: json!({
            "ok": true,
            "action": "pass-remove",
            "pass": name,
            "source_left_on_disk": removed.source,
            "removed_consumer_channels": if force { consumers } else { Vec::<String>::new() },
        }),
    })
}

#[derive(Debug, Clone)]
pub struct ChannelSetOptions {
    pub pass: String,
    pub channel: u8,
    pub source: String,
    pub kind: Option<crate::manifest::InputKind>,
    pub previous: bool,
    pub filter: crate::manifest::Filter,
    pub wrap: crate::manifest::Wrap,
}

pub fn set_channel(project_path: &Path, options: &ChannelSetOptions) -> Result<Output> {
    let pass_name = options.pass.as_str();
    let channel = options.channel;
    let source = options.source.as_str();
    let kind = options.kind;
    let previous = options.previous;
    let filter = options.filter;
    let wrap = options.wrap;
    use crate::manifest::{FrameRef, Input};
    use crate::scaffold::write_manifest;

    if channel > 3 {
        bail!("channel must be between 0 and 3");
    }
    let mut loaded = LoadedManifest::load(project_path)?;
    let pass = loaded
        .manifest
        .passes
        .iter_mut()
        .find(|pass| pass.name == pass_name)
        .with_context(|| format!("unknown pass '{pass_name}'"))?;
    let input = Input {
        channel,
        source: source.to_string(),
        kind,
        frame: if previous {
            FrameRef::Previous
        } else {
            FrameRef::Current
        },
        filter,
        wrap,
    };
    if let Some(existing) = pass
        .inputs
        .iter_mut()
        .find(|input| input.channel == channel)
    {
        *existing = input;
    } else {
        pass.inputs.push(input);
        pass.inputs.sort_by_key(|input| input.channel);
    }
    loaded.manifest.validate_structure()?;
    write_manifest(&loaded.root, &loaded.manifest)?;

    Ok(Output {
        human: format!("Set {} iChannel{} <- {}", pass_name, channel, source),
        json: json!({
            "ok": true,
            "action": "channel-set",
            "pass": pass_name,
            "channel": channel,
            "source": source,
            "previous": previous,
            "filter": format!("{filter:?}").to_lowercase(),
            "wrap": format!("{wrap:?}").to_lowercase(),
        }),
    })
}

pub fn remove_channel(project_path: &Path, pass_name: &str, channel: u8) -> Result<Output> {
    use crate::scaffold::write_manifest;

    if channel > 3 {
        bail!("channel must be between 0 and 3");
    }
    let mut loaded = LoadedManifest::load(project_path)?;
    let pass = loaded
        .manifest
        .passes
        .iter_mut()
        .find(|pass| pass.name == pass_name)
        .with_context(|| format!("unknown pass '{pass_name}'"))?;
    let before = pass.inputs.len();
    pass.inputs.retain(|input| input.channel != channel);
    if pass.inputs.len() == before {
        bail!("{} has no iChannel{} binding", pass_name, channel);
    }
    write_manifest(&loaded.root, &loaded.manifest)?;

    Ok(Output {
        human: format!("Removed {} iChannel{}", pass_name, channel),
        json: json!({
            "ok": true,
            "action": "channel-remove",
            "pass": pass_name,
            "channel": channel,
        }),
    })
}

fn slug(value: &str) -> String {
    let mut result = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            result.push(character.to_ascii_lowercase());
        } else if !result.ends_with('-') {
            result.push('-');
        }
    }
    let result = result.trim_matches('-');
    if result.is_empty() {
        "pass".to_string()
    } else {
        result.to_string()
    }
}

fn pass_stub(kind: PassKind) -> &'static str {
    match kind {
        PassKind::Buffer | PassKind::Image => crate::include_file!("templates/pass/buffer.frag"),
        PassKind::Cubemap => crate::include_file!("templates/pass/cubemap.frag"),
    }
}
