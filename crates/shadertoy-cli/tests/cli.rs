use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-cli-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temporary CLI test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn shadertoy(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_shadertoy"))
        .args(args)
        .output()
        .expect("run shadertoy test binary")
}

#[test]
fn json_mode_covers_clap_parse_errors() {
    let output = shadertoy(&[
        "--json",
        "channel",
        "--project",
        ".",
        "set",
        "image",
        "not-a-channel",
        "keyboard",
    ]);

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse CLI error JSON");
    assert_eq!(value["ok"], false);
    assert_eq!(value["kind"], "ValueValidation");
}

#[test]
fn check_accepts_project_option_without_dropping_positional_form() {
    let missing = "definitely-missing-shadertoy-project";
    let option = shadertoy(&["--json", "check", "--project", missing]);
    assert_eq!(option.status.code(), Some(1));
    let value: serde_json::Value =
        serde_json::from_slice(&option.stdout).expect("parse check error JSON");
    assert_eq!(value["ok"], false);

    let positional = shadertoy(&["--json", "check", missing]);
    assert_eq!(positional.status.code(), Some(1));
    let value: serde_json::Value =
        serde_json::from_slice(&positional.stdout).expect("parse positional check error JSON");
    assert_eq!(value["ok"], false);
}

#[test]
fn pass_add_rejects_escape_without_leaving_files() {
    let temp = TempRoot::new("pass-add");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();

    let created = shadertoy(&["new", &project_arg]);
    assert!(created.status.success(), "{:?}", created);

    let traversal = shadertoy(&[
        "pass",
        "--project",
        &project_arg,
        "add",
        "escaped",
        "--source",
        "../outside.frag",
    ]);
    assert!(!traversal.status.success());
    assert!(!temp.path().join("outside.frag").exists());

    let invalid = shadertoy(&["pass", "--project", &project_arg, "add", ""]);
    assert!(!invalid.status.success());
    assert!(!project.join("shaders/pass.frag").exists());

    let reserved = shadertoy(&["pass", "--project", &project_arg, "add", "keyboard"]);
    assert!(!reserved.status.success());
    assert!(!project.join("shaders/keyboard.frag").exists());
}

#[cfg(unix)]
#[test]
fn pass_add_rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let temp = TempRoot::new("symlink");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();
    assert!(shadertoy(&["new", &project_arg]).status.success());

    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).expect("create outside directory");
    symlink(&outside, project.join("shaders/link")).expect("create escape symlink");

    let output = shadertoy(&[
        "pass",
        "--project",
        &project_arg,
        "add",
        "escaped",
        "--source",
        "shaders/link/escaped.frag",
    ]);
    assert!(!output.status.success());
    assert!(!outside.join("escaped.frag").exists());
}

#[cfg(unix)]
#[test]
fn pass_add_rejects_dangling_file_symlink_escape() {
    use std::os::unix::fs::symlink;

    let temp = TempRoot::new("dangling-symlink");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();
    assert!(shadertoy(&["new", &project_arg]).status.success());

    let outside = temp.path().join("outside.frag");
    symlink(&outside, project.join("shaders/escape.frag")).expect("create dangling escape symlink");

    let output = shadertoy(&[
        "pass",
        "--project",
        &project_arg,
        "add",
        "escaped",
        "--source",
        "shaders/escape.frag",
    ]);
    assert!(!output.status.success());
    assert!(!outside.exists());
}

#[cfg(target_os = "linux")]
#[test]
fn render_supports_rgb_widths_that_are_not_four_pixel_aligned() {
    let temp = TempRoot::new("odd-width-render");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();
    let created = shadertoy(&["new", &project_arg, "--template", "multipass"]);
    assert!(created.status.success(), "{created:?}");

    for (width, pass) in [("641", None), ("643", Some("buffer-a"))] {
        let output = temp
            .path()
            .join(format!("render-{width}-{}.png", pass.unwrap_or("image")));
        let output_arg = output.to_string_lossy().into_owned();
        let mut args = vec![
            "render",
            "--project",
            &project_arg,
            "--width",
            width,
            "--height",
            "97",
            "-o",
            &output_arg,
        ];
        if let Some(pass) = pass {
            args.extend(["--pass", pass]);
        }

        let rendered = shadertoy(&args);
        assert!(
            rendered.status.success(),
            "odd-width render failed for width={width} pass={pass:?}: {rendered:?}"
        );
        let image = image::open(&output)
            .unwrap_or_else(|error| panic!("open rendered image {}: {error}", output.display()));
        assert_eq!(image.width(), width.parse::<u32>().unwrap());
        assert_eq!(image.height(), 97);
    }
}
