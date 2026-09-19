use crate::manifest::schema_json;
use anyhow::{Result, bail};

pub fn topic(name: &str, schema: bool) -> Result<String> {
    if schema {
        if name != "manifest" {
            bail!("--schema is only valid with 'shadertoy docs manifest'");
        }
        return schema_json();
    }

    let text = match name {
        "agent" => crate::include_file!("docs/agent.md"),
        "project" => crate::include_file!("docs/project.md"),
        "manifest" => crate::include_file!("docs/manifest.md"),
        "passes" => crate::include_file!("docs/passes.md"),
        "buffers" => crate::include_file!("docs/buffers.md"),
        "channels" => crate::include_file!("docs/channels.md"),
        "state" => crate::include_file!("docs/state.md"),
        "preview" => crate::include_file!("docs/preview.md"),
        other => bail!(
            "unknown documentation topic '{other}'; expected agent, project, manifest, passes, buffers, channels, state, or preview"
        ),
    };
    Ok(text.trim().to_string())
}
