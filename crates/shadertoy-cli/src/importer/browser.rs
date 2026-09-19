use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CAMOUFOX_VERSION: &str = "0.5.6";
const PYTHON_OVERRIDE: &str = "SHADERTOY_CAMOUFOX_PYTHON";
const CACHE_OVERRIDE: &str = "SHADERTOY_CACHE_DIR";

#[derive(Debug)]
pub struct BrowserCapture {
    pub shader_id: String,
    pub source_url: String,
    pub response: PathBuf,
    pub resources: HashMap<String, PathBuf>,
}

pub trait BrowserImporter {
    fn capture(&self, source: &str, output: &Path) -> Result<BrowserCapture>;
}

#[derive(Debug, Default)]
pub struct CamoufoxImporter;

#[derive(Debug, Deserialize)]
struct CaptureMetadata {
    shader_id: String,
    source_url: String,
    response: String,
    resources: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct PythonCommand {
    executable: OsString,
    prefix: Vec<OsString>,
}

impl PythonCommand {
    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command.args(&self.prefix);
        command
    }
}

impl BrowserImporter for CamoufoxImporter {
    fn capture(&self, source: &str, output: &Path) -> Result<BrowserCapture> {
        let shader_id = shader_id_from(source)?;
        fs::create_dir_all(output).with_context(|| {
            format!(
                "failed to create import staging directory {}",
                output.display()
            )
        })?;
        let helper = output.join("camoufox_import.py");
        fs::write(&helper, crate::include_file!("import/camoufox_import.py"))
            .with_context(|| format!("failed to write Camoufox helper {}", helper.display()))?;

        let python = camoufox_python()?;
        let profile = cache_root()?.join("camoufox").join("profile");
        fs::create_dir_all(&profile)
            .with_context(|| format!("failed to create Camoufox profile {}", profile.display()))?;
        let status = python
            .command()
            .arg(&helper)
            .arg("--shader-id")
            .arg(&shader_id)
            .arg("--output")
            .arg(output)
            .arg("--profile")
            .arg(&profile)
            .status()
            .context("failed to start Camoufox import helper")?;
        if !status.success() {
            bail!("Camoufox import helper exited with status {status}");
        }

        let metadata_path = output.join("browser-import.json");
        let metadata: CaptureMetadata = serde_json::from_slice(
            &fs::read(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .context("failed to parse Camoufox import metadata")?;
        if metadata.shader_id != shader_id {
            bail!(
                "Camoufox helper returned shader id '{}' while '{}' was requested",
                metadata.shader_id,
                shader_id
            );
        }

        let response = checked_staging_path(output, &metadata.response)?;
        let mut resources = HashMap::new();
        for (remote, relative) in metadata.resources {
            resources.insert(remote, checked_staging_path(output, &relative)?);
        }
        Ok(BrowserCapture {
            shader_id: metadata.shader_id,
            source_url: metadata.source_url,
            response,
            resources,
        })
    }
}

pub fn shader_id_from(source: &str) -> Result<String> {
    let source = source.trim();
    if source.is_empty() {
        bail!("ShaderToy URL/id must not be empty");
    }

    let id = if source.contains("://") {
        let prefixes = [
            "https://www.shadertoy.com/view/",
            "https://shadertoy.com/view/",
            "http://www.shadertoy.com/view/",
            "http://shadertoy.com/view/",
        ];
        let rest = prefixes
            .into_iter()
            .find_map(|prefix| source.strip_prefix(prefix))
            .context("expected a ShaderToy URL like https://www.shadertoy.com/view/XXXXXX")?;
        rest.split(['?', '#'])
            .next()
            .unwrap_or_default()
            .trim_matches('/')
            .to_string()
    } else {
        source.trim_matches('/').to_string()
    };

    if id.is_empty()
        || id.len() > 32
        || id.contains('/')
        || !id.bytes().all(|byte| byte.is_ascii_alphanumeric())
    {
        bail!("invalid ShaderToy shader id '{id}'");
    }
    Ok(id)
}

fn camoufox_python() -> Result<PythonCommand> {
    if let Some(value) = env::var_os(PYTHON_OVERRIDE).filter(|value| !value.is_empty()) {
        let command = PythonCommand {
            executable: value,
            prefix: Vec::new(),
        };
        verify_camoufox(&command).with_context(|| {
            format!("{PYTHON_OVERRIDE} is set but that interpreter cannot import camoufox")
        })?;
        return Ok(command);
    }

    let root = cache_root()?
        .join("camoufox")
        .join(format!("v{CAMOUFOX_VERSION}"));
    let venv_python = if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    };
    let marker = root.join(format!(".ready-{CAMOUFOX_VERSION}"));
    let managed = PythonCommand {
        executable: venv_python.as_os_str().to_owned(),
        prefix: Vec::new(),
    };
    if marker.is_file() && venv_python.is_file() && verify_camoufox(&managed).is_ok() {
        return Ok(managed);
    }

    let bootstrap = find_python()?;
    if !venv_python.is_file() {
        if let Some(parent) = root.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create Camoufox cache {}", parent.display()))?;
        }
        let mut command = bootstrap.command();
        command.args(["-m", "venv"]).arg(&root);
        require_success(
            command
                .output()
                .context("failed to create Camoufox Python environment")?,
            "creating Camoufox Python environment",
        )?;
    }

    let mut install = managed.command();
    install.args([
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--quiet",
        "--upgrade",
        &format!("camoufox=={CAMOUFOX_VERSION}"),
    ]);
    require_success(
        install.output().context("failed to install Camoufox")?,
        "installing Camoufox",
    )?;

    let mut fetch = managed.command();
    fetch.args(["-m", "camoufox", "fetch"]);
    require_success(
        fetch
            .output()
            .context("failed to fetch the Camoufox browser")?,
        "fetching the Camoufox browser",
    )?;

    verify_camoufox(&managed)?;
    fs::write(&marker, CAMOUFOX_VERSION).with_context(|| {
        format!(
            "failed to write Camoufox readiness marker {}",
            marker.display()
        )
    })?;
    Ok(managed)
}

fn verify_camoufox(python: &PythonCommand) -> Result<()> {
    let mut command = python.command();
    command.args(["-c", "import camoufox"]);
    require_success(
        command
            .output()
            .context("failed to verify Camoufox Python environment")?,
        "verifying Camoufox Python environment",
    )
}

fn find_python() -> Result<PythonCommand> {
    let candidates: [(&str, &[&str]); 3] = [("python3", &[]), ("python", &[]), ("py", &["-3"])];
    for (executable, prefix) in candidates {
        let command = PythonCommand {
            executable: OsString::from(executable),
            prefix: prefix.iter().map(OsString::from).collect(),
        };
        let mut check = command.command();
        check.args([
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)",
        ]);
        if check.output().is_ok_and(|output| output.status.success()) {
            return Ok(command);
        }
    }
    bail!("Camoufox import requires Python 3.10+; install Python or set {PYTHON_OVERRIDE}")
}

fn cache_root() -> Result<PathBuf> {
    if let Some(path) = env::var_os(CACHE_OVERRIDE).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if cfg!(windows) {
        if let Some(path) = env::var_os("LOCALAPPDATA") {
            return Ok(PathBuf::from(path).join("shadertoy"));
        }
    } else {
        if let Some(path) = env::var_os("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(path).join("shadertoy"));
        }
        if let Some(path) = env::var_os("HOME") {
            return Ok(PathBuf::from(path).join(".cache").join("shadertoy"));
        }
    }
    Ok(env::temp_dir().join("shadertoy-cache"))
}

fn checked_staging_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
    {
        bail!(
            "Camoufox helper returned unsafe staging path {}",
            relative.display()
        );
    }
    let path = root.join(relative);
    if !path.is_file() {
        bail!("Camoufox helper output is missing {}", path.display());
    }
    Ok(path)
}

fn require_success(output: Output, action: &str) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if !stderr.trim().is_empty() {
        stderr.trim()
    } else if !stdout.trim().is_empty() {
        stdout.trim()
    } else {
        "process exited without diagnostics"
    };
    bail!("{action} failed: {detail}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_id_accepts_id_and_canonical_url() {
        assert_eq!(shader_id_from("lsX3W4").unwrap(), "lsX3W4");
        assert_eq!(
            shader_id_from("https://www.shadertoy.com/view/lsX3W4?foo=1").unwrap(),
            "lsX3W4"
        );
    }

    #[test]
    fn shader_id_rejects_other_hosts_and_paths() {
        assert!(shader_id_from("https://example.com/view/lsX3W4").is_err());
        assert!(shader_id_from("https://www.shadertoy.com/embed/lsX3W4").is_err());
        assert!(shader_id_from("../lsX3W4").is_err());
    }

    #[test]
    fn staging_path_rejects_escape() {
        let root = Path::new("/tmp/import");
        assert!(checked_staging_path(root, "../escape").is_err());
    }
}
