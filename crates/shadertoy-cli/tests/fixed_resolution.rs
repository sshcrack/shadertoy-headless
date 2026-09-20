use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-fixed-resolution-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test directory");
        Self(path)
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
        .expect("run shadertoy")
}

fn write_project(root: &Path) {
    std::fs::create_dir_all(root.join("shaders")).expect("create shader directory");
    std::fs::write(
        root.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "fixed-resolution-test"

[render]
width = 7
height = 5
fps = 60.0
preview_time = 0.0

[[pass]]
name = "spectrum"
kind = "buffer"
source = "shaders/spectrum.frag"
width = 2
height = 2

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "spectrum"
kind = "pass"
frame = "current"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        root.join("shaders/spectrum.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(iResolution.xy / 10.0, 0.0, 1.0);
}
"#,
    )
    .expect("write spectrum shader");
    std::fs::write(
        root.join("shaders/image.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(iChannelResolution[0].xy / 10.0, 0.0, 1.0);
}
"#,
    )
    .expect("write image shader");
}

#[cfg(target_os = "linux")]
#[test]
fn fixed_pass_resolution_drives_uniforms_and_named_renders() {
    let temp = TempRoot::new("render");
    let project = temp.0.join("project");
    write_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let final_path = temp.0.join("final.png");
    let final_arg = final_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "-o",
        &final_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let final_image = image::open(&final_path)
        .expect("open final render")
        .to_rgb8();
    assert_eq!(final_image.dimensions(), (7, 5));
    let final_pixel = final_image.get_pixel(0, 0).0;
    assert!(
        (i16::from(final_pixel[0]) - 51).abs() <= 1,
        "{final_pixel:?}"
    );
    assert!(
        (i16::from(final_pixel[1]) - 51).abs() <= 1,
        "{final_pixel:?}"
    );

    let pass_path = temp.0.join("spectrum.png");
    let pass_arg = pass_path.to_string_lossy().into_owned();
    let named = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--width",
        "11",
        "--height",
        "6",
        "--frame",
        "0",
        "--pass",
        "spectrum",
        "-o",
        &pass_arg,
    ]);
    assert!(named.status.success(), "{named:?}");
    let pass_image = image::open(&pass_path)
        .expect("open named pass render")
        .to_rgb8();
    assert_eq!(pass_image.dimensions(), (2, 2));
    let pass_pixel = pass_image.get_pixel(0, 0).0;
    assert!((i16::from(pass_pixel[0]) - 51).abs() <= 1, "{pass_pixel:?}");
    assert!((i16::from(pass_pixel[1]) - 51).abs() <= 1, "{pass_pixel:?}");

    let frames_dir = temp.0.join("frames");
    let frames_arg = frames_dir.to_string_lossy().into_owned();
    let sheet_path = temp.0.join("sheet.png");
    let sheet_arg = sheet_path.to_string_lossy().into_owned();
    let frames = shadertoy(&[
        "render-frames",
        "--project",
        &project_arg,
        "--width",
        "13",
        "--height",
        "9",
        "--pass",
        "spectrum",
        "--frames",
        "0,1",
        "--output-dir",
        &frames_arg,
        "--contact-sheet",
        &sheet_arg,
    ]);
    assert!(frames.status.success(), "{frames:?}");
    for frame in ["frame-000000.png", "frame-000001.png"] {
        let image = image::open(frames_dir.join(frame)).expect("open named frame");
        assert_eq!((image.width(), image.height()), (2, 2));
    }
    let sheet = image::open(&sheet_path).expect("open contact sheet");
    assert_eq!((sheet.width(), sheet.height()), (4, 2));
}

#[cfg(target_os = "linux")]
#[test]
fn fixed_pass_state_and_overrides_use_pass_dimensions() {
    let temp = TempRoot::new("state");
    let project = temp.0.join("project");
    write_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let state_path = temp.0.join("fixed.ststate");
    let state_arg = state_path.to_string_lossy().into_owned();
    let captured = shadertoy(&[
        "state",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "1",
        "-o",
        &state_arg,
    ]);
    assert!(captured.status.success(), "{captured:?}");

    let inspected = shadertoy(&["--json", "state", "inspect", &state_arg]);
    assert!(inspected.status.success(), "{inspected:?}");
    let inspection: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse state inspection");
    assert_eq!(inspection["header"]["format"], 3);
    assert_eq!(
        inspection["header"]["buffer_dimensions"]["spectrum"]["width"],
        2
    );
    assert_eq!(
        inspection["header"]["buffer_dimensions"]["spectrum"]["height"],
        2
    );

    let resumed_path = temp.0.join("resumed.png");
    let resumed_arg = resumed_path.to_string_lossy().into_owned();
    let resumed = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--state",
        &state_arg,
        "--frame",
        "2",
        "--width",
        "11",
        "--height",
        "6",
        "-o",
        &resumed_arg,
    ]);
    assert!(resumed.status.success(), "{resumed:?}");
    let resumed_image = image::open(&resumed_path).expect("open resumed render");
    assert_eq!((resumed_image.width(), resumed_image.height()), (11, 6));

    let exact_override = temp.0.join("exact.png");
    image::RgbaImage::from_pixel(2, 2, image::Rgba([64, 32, 16, 255]))
        .save(&exact_override)
        .expect("write exact override");
    let assignment = format!("spectrum={}", exact_override.to_string_lossy());
    let updated_state = temp.0.join("updated.ststate");
    let updated_state_arg = updated_state.to_string_lossy().into_owned();
    let state_set = shadertoy(&[
        "state",
        "set",
        &state_arg,
        &assignment,
        "-o",
        &updated_state_arg,
    ]);
    assert!(state_set.status.success(), "{state_set:?}");

    let override_render = temp.0.join("override-render.png");
    let override_render_arg = override_render.to_string_lossy().into_owned();
    let accepted = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set-buffer",
        &assignment,
        "-o",
        &override_render_arg,
    ]);
    assert!(accepted.status.success(), "{accepted:?}");

    let wrong_override = temp.0.join("wrong.png");
    image::RgbaImage::from_pixel(7, 5, image::Rgba([0, 0, 0, 255]))
        .save(&wrong_override)
        .expect("write wrong override");
    let wrong_assignment = format!("spectrum={}", wrong_override.to_string_lossy());
    let rejected = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set-buffer",
        &wrong_assignment,
    ]);
    assert!(!rejected.status.success());
}

#[cfg(target_os = "linux")]
#[test]
fn state_restore_rejects_buffer_resolution_semantic_changes() {
    let temp = TempRoot::new("state-resolution-change");
    let project = temp.0.join("project");
    write_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let state_path = temp.0.join("fixed.ststate");
    let state_arg = state_path.to_string_lossy().into_owned();
    let captured = shadertoy(&[
        "state",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "1",
        "-o",
        &state_arg,
    ]);
    assert!(captured.status.success(), "{captured:?}");

    let manifest_path = project.join("ShaderToy.toml");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest = manifest.replacen("width = 2\nheight = 2\n", "", 1);
    std::fs::write(&manifest_path, manifest).expect("make spectrum output-sized");

    let output_path = temp.0.join("should-not-render.png");
    let output_arg = output_path.to_string_lossy().into_owned();
    let resumed = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--state",
        &state_arg,
        "--frame",
        "2",
        "-o",
        &output_arg,
    ]);
    assert!(!resumed.status.success(), "{resumed:?}");
    let stderr = String::from_utf8_lossy(&resumed.stderr);
    assert!(
        stderr.contains("state buffer 'spectrum' is 2x2")
            && stderr.contains("current pass expects 7x5"),
        "{stderr}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn fixed_pass_reports_dynamic_input_resolution() {
    let temp = TempRoot::new("dynamic-input-resolution");
    let project = temp.0.join("project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shader directory");
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "fixed-dynamic-input-resolution"

[render]
width = 7
height = 5
fps = 60.0
preview_time = 0.0

[[pass]]
name = "dynamic"
kind = "buffer"
source = "shaders/dynamic.frag"

[[pass]]
name = "fixed"
kind = "buffer"
source = "shaders/fixed.frag"
width = 2
height = 2

[[pass.input]]
channel = 0
source = "dynamic"
filter = "nearest"
wrap = "clamp"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "fixed"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        project.join("shaders/dynamic.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.0, 0.0, 0.0, 1.0); }",
    )
    .expect("write dynamic shader");
    std::fs::write(
        project.join("shaders/fixed.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(iChannelResolution[0].xy / 10.0, 0.0, 1.0); }",
    )
    .expect("write fixed shader");
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, vec2(0.5)); }",
    )
    .expect("write image shader");

    let project_arg = project.to_string_lossy().into_owned();
    let output_path = temp.0.join("fixed.png");
    let output_arg = output_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--pass",
        "fixed",
        "-o",
        &output_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");

    let image = image::open(&output_path)
        .expect("open fixed pass render")
        .to_rgb8();
    assert_eq!(image.dimensions(), (2, 2));
    let pixel = image.get_pixel(0, 0).0;
    assert!((i16::from(pixel[0]) - 179).abs() <= 2, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 128).abs() <= 2, "{pixel:?}");
}
