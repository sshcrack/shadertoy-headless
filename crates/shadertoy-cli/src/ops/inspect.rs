use super::*;

pub fn inspect_project(path: &Path, mode: InspectMode) -> Result<Output> {
    let loaded = LoadedManifest::load(path)?;
    let manifest = &loaded.manifest;

    match mode {
        InspectMode::Summary => {
            let human = format!(
                "Project: {}\nRoot: {}\nPasses: {}\nAssets: {}\nFinal output: {}\nRender: {}x{} @ {} fps",
                manifest.project.name,
                loaded.root.display(),
                manifest.passes.len(),
                manifest.assets.len(),
                manifest.final_pass().name,
                manifest.render.width,
                manifest.render.height,
                manifest.render.fps,
            );
            let passes = manifest
                .passes
                .iter()
                .map(|pass| {
                    json!({
                        "name": pass.name,
                        "kind": format!("{:?}", pass.kind).to_lowercase(),
                        "source": pass.source,
                        "inputs": pass.inputs.len(),
                    })
                })
                .collect::<Vec<_>>();
            Ok(Output {
                human,
                json: json!({
                    "ok": true,
                    "project": manifest.project.name,
                    "root": loaded.root,
                    "final_output": manifest.final_pass().name,
                    "render": manifest.render,
                    "passes": passes,
                    "assets": manifest.assets,
                }),
            })
        }
        InspectMode::Graph => {
            let mut human = format!("Project graph: {}\n", manifest.project.name);
            let mut edges = Vec::new();
            for pass in &manifest.passes {
                human.push_str(&format!("  {} ({:?})\n", pass.name, pass.kind));
                for input in &pass.inputs {
                    let kind = manifest.infer_input_kind(input)?;
                    human.push_str(&format!(
                        "    iChannel{} <- {} ({:?}, {:?}, {:?}, {:?})\n",
                        input.channel, input.source, kind, input.frame, input.filter, input.wrap
                    ));
                    edges.push(json!({
                        "from": input.source,
                        "to": pass.name,
                        "channel": input.channel,
                        "kind": format!("{kind:?}").to_lowercase(),
                        "frame": format!("{:?}", input.frame).to_lowercase(),
                        "filter": format!("{:?}", input.filter).to_lowercase(),
                        "wrap": format!("{:?}", input.wrap).to_lowercase(),
                    }));
                }
            }
            Ok(Output {
                human,
                json: json!({
                    "ok": true,
                    "project": manifest.project.name,
                    "passes": manifest.passes,
                    "edges": edges,
                }),
            })
        }
        InspectMode::Pass(name) => {
            let pass = manifest
                .passes
                .iter()
                .find(|pass| pass.name == name)
                .with_context(|| format!("unknown pass '{name}'"))?;
            let human = format!(
                "Pass: {}\nKind: {:?}\nSource: {}\nInputs: {}",
                pass.name,
                pass.kind,
                pass.source,
                pass.inputs.len()
            );
            Ok(Output {
                human,
                json: json!({
                    "ok": true,
                    "project": manifest.project.name,
                    "pass": pass,
                    "is_final": pass.name == manifest.final_pass().name,
                }),
            })
        }
        InspectMode::Channels(name) => {
            let pass = manifest
                .passes
                .iter()
                .find(|pass| pass.name == name)
                .with_context(|| format!("unknown pass '{name}'"))?;
            let mut human = format!("Channels for {}:\n", pass.name);
            let mut channels = Vec::new();
            if pass.inputs.is_empty() {
                human.push_str("  (none)");
            }
            for input in &pass.inputs {
                let kind = manifest.infer_input_kind(input)?;
                human.push_str(&format!(
                    "  iChannel{} <- {} ({:?}, {:?}, {:?}, {:?})\n",
                    input.channel, input.source, kind, input.frame, input.filter, input.wrap
                ));
                channels.push(json!({
                    "channel": input.channel,
                    "source": input.source,
                    "kind": format!("{kind:?}").to_lowercase(),
                    "frame": format!("{:?}", input.frame).to_lowercase(),
                    "filter": format!("{:?}", input.filter).to_lowercase(),
                    "wrap": format!("{:?}", input.wrap).to_lowercase(),
                }));
            }
            Ok(Output {
                human,
                json: json!({
                    "ok": true,
                    "project": manifest.project.name,
                    "pass": pass.name,
                    "channels": channels,
                }),
            })
        }
    }
}
