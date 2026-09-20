use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub const LOCAL_SCHEMA_PATH: &str = ".shadertoy/shadertoy.schema.json";

pub fn write_current(root: &Path) -> Result<()> {
    let path = root.join(LOCAL_SCHEMA_PATH);
    fs::write(&path, crate::include_file!("schema/shadertoy.schema.json")).with_context(|| {
        format!(
            "failed to write project-local manifest schema {}",
            path.display()
        )
    })
}

pub fn refresh_existing(root: &Path) -> Result<bool> {
    let path = root.join(LOCAL_SCHEMA_PATH);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect project-local manifest schema {}",
                    path.display()
                )
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        anyhow::bail!(
            "project-local manifest schema must be a regular file: {}",
            path.display()
        );
    }

    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("failed to resolve project root {}", root.display()))?;
    let canonical_path = fs::canonicalize(&path).with_context(|| {
        format!(
            "failed to resolve project-local manifest schema {}",
            path.display()
        )
    })?;
    if !canonical_path.starts_with(&canonical_root) {
        anyhow::bail!(
            "project-local manifest schema resolves outside the project root: {}",
            path.display()
        );
    }

    let current = fs::read(&canonical_path).with_context(|| {
        format!(
            "failed to read project-local manifest schema {}",
            path.display()
        )
    })?;
    let expected = crate::include_file!("schema/shadertoy.schema.json").as_bytes();
    if current == expected {
        return Ok(false);
    }

    fs::write(&canonical_path, expected).with_context(|| {
        format!(
            "failed to refresh project-local manifest schema {}",
            path.display()
        )
    })?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn refresh_existing_updates_stale_schema() {
        let root = tempdir().unwrap();
        let schema = root.path().join(LOCAL_SCHEMA_PATH);
        fs::create_dir_all(schema.parent().unwrap()).unwrap();
        fs::write(&schema, b"{\"stale\":true}").unwrap();

        assert!(refresh_existing(root.path()).unwrap());
        assert_eq!(
            fs::read_to_string(schema).unwrap(),
            crate::include_file!("schema/shadertoy.schema.json")
        );
    }

    #[test]
    fn refresh_existing_leaves_missing_schema_absent() {
        let root = tempdir().unwrap();

        assert!(!refresh_existing(root.path()).unwrap());
        assert!(!root.path().join(LOCAL_SCHEMA_PATH).exists());
    }

    #[test]
    fn refresh_existing_skips_current_schema() {
        let root = tempdir().unwrap();
        let schema = root.path().join(LOCAL_SCHEMA_PATH);
        fs::create_dir_all(schema.parent().unwrap()).unwrap();
        fs::write(
            &schema,
            crate::include_file!("schema/shadertoy.schema.json"),
        )
        .unwrap();

        assert!(!refresh_existing(root.path()).unwrap());
    }

    #[test]
    fn loading_project_refreshes_existing_schema() {
        let root = tempdir().unwrap();
        let schema = root.path().join(LOCAL_SCHEMA_PATH);
        fs::create_dir_all(schema.parent().unwrap()).unwrap();
        fs::write(&schema, b"{\"stale\":true}").unwrap();
        fs::write(
            root.path().join("ShaderToy.toml"),
            r#"format = 1
[project]
name = "schema-refresh"
[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"
"#,
        )
        .unwrap();

        crate::manifest::LoadedManifest::load(root.path()).unwrap();

        assert_eq!(
            fs::read_to_string(schema).unwrap(),
            crate::include_file!("schema/shadertoy.schema.json")
        );
    }
}
