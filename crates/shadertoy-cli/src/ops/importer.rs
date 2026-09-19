use super::*;
use crate::importer::{BrowserImporter, CamoufoxImporter, materialize_capture};

pub fn import_project(source: &str, destination: Option<&Path>) -> Result<Output> {
    let staging =
        tempfile::tempdir().context("failed to create ShaderToy import staging directory")?;
    let browser = CamoufoxImporter;
    let capture = browser.capture(source, staging.path())?;
    let imported = materialize_capture(&capture, destination)?;

    let mut human = format!(
        "Imported {} to {} ({} passes, {} assets)",
        imported.source_url,
        imported.root.display(),
        imported.pass_count,
        imported.asset_count
    );
    for warning in &imported.warnings {
        human.push_str("\nwarning: ");
        human.push_str(warning);
    }

    Ok(Output {
        human,
        json: json!({
            "ok": true,
            "action": "import",
            "project": imported.root,
            "name": imported.name,
            "shader_id": imported.shader_id,
            "source_url": imported.source_url,
            "passes": imported.pass_count,
            "assets": imported.asset_count,
            "warnings": imported.warnings,
        }),
    })
}
