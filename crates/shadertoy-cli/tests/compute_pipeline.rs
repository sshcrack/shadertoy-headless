use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-compute-pipeline-{}-{}",
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
    std::fs::create_dir_all(root.join("shaders")).expect("create shaders");
    std::fs::write(
        root.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "compute-cli-test"

[render]
width = 4
height = 2
fps = 60.0
preview_time = 0.0

[[pass]]
name = "simulation"
kind = "compute"
source = "shaders/simulation.comp"
width = 4
height = 2
format = "rg32f"
iterations = 3
local_size = [4, 1, 1]

[[pass.storage]]
binding = 3
name = "counters"
size = 32

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "simulation"
filter = "nearest"
wrap = "clamp"

[[test]]
name = "compute-numeric"
pass = "simulation"
frame = 0
assert_no_nan = true
assert_no_inf = true
mean_range = [0.374, 0.376]
"#,
    )
    .expect("write manifest");

    std::fs::write(
        root.join("shaders/simulation.comp"),
        r#"layout(std430, binding = 3) buffer Counters {
    uint counters[];
};

void mainCompute(ivec2 coord) {
    uint index = uint(coord.y) * uint(iResolution.x) + uint(coord.x);
    counters[index] += 1u;
    imageStore(iOutput, coord, vec4(float(counters[index]) / 10.0, float(iIteration) / 10.0, 0.75, 1.0));
}
"#,
    )
    .expect("write compute shader");

    std::fs::write(
        root.join("shaders/image.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 value = texture(iChannel0, fragCoord / iResolution.xy).rg;
    fragColor = vec4(value, 0.25, 1.0);
}
"#,
    )
    .expect("write image shader");
}

#[cfg(target_os = "linux")]
#[test]
fn cli_checks_renders_inspects_and_tests_compute_projects() {
    let temp = TempRoot::new();
    let project = temp.0.join("project");
    write_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let named_path = temp.0.join("compute.png");
    let named_arg = named_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--pass",
        "simulation",
        "-o",
        &named_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let image = image::open(&named_path)
        .expect("open compute image")
        .to_rgb8();
    let pixel = image.get_pixel(0, 0).0;
    assert!((i16::from(pixel[0]) - 77).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 51).abs() <= 1, "{pixel:?}");
    assert!(pixel[2] <= 1, "{pixel:?}");

    let inspected = shadertoy(&[
        "--json",
        "inspect",
        "--project",
        &project_arg,
        "buffer",
        "simulation",
        "--frame",
        "0",
        "--pixel",
        "0,0",
    ]);
    assert!(inspected.status.success(), "{inspected:?}");
    let value: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse inspect json");
    assert_eq!(value["pass"], "simulation");
    assert!((value["pixel"]["rgba"][0].as_f64().unwrap() - 0.3).abs() < 1e-5);
    assert!((value["pixel"]["rgba"][1].as_f64().unwrap() - 0.2).abs() < 1e-5);

    let tested = shadertoy(&["test", "--project", &project_arg]);
    assert!(tested.status.success(), "{tested:?}");
}

#[test]
fn pass_add_compute_creates_fixed_compute_stub() {
    let temp = TempRoot::new();
    let project = temp.0.join("project");
    let project_arg = project.to_string_lossy().into_owned();
    assert!(shadertoy(&["new", &project_arg]).status.success());

    let added = shadertoy(&[
        "pass",
        "--project",
        &project_arg,
        "add",
        "simulation",
        "--kind",
        "compute",
    ]);
    assert!(added.status.success(), "{added:?}");
    assert!(project.join("shaders/simulation.comp").is_file());

    let manifest = std::fs::read_to_string(project.join("ShaderToy.toml")).expect("read manifest");
    assert!(manifest.contains("kind = \"compute\""));
    assert!(manifest.contains("width = 256"));
    assert!(manifest.contains("height = 256"));
}

#[cfg(target_os = "linux")]
#[test]
fn typed_compute_feedback_survives_state_capture_and_restore() {
    let temp = TempRoot::new();
    let project = temp.0.join("state-project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shaders");
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "compute-state-roundtrip"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[pass]]
name = "simulation"
kind = "compute"
source = "shaders/simulation.comp"
width = 1
height = 1
format = "rg32f"
local_size = [1, 1, 1]

[[pass.input]]
channel = 0
source = "simulation"
frame = "previous"
filter = "nearest"
wrap = "clamp"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "simulation"
frame = "current"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        project.join("shaders/simulation.comp"),
        r#"void mainCompute(ivec2 coord) {
    float previous = iFrame == 0 ? 0.0 : texture(iChannel0, vec2(0.5)).r;
    imageStore(iOutput, coord, vec4(previous + 0.1, float(iFrame), 0.0, 1.0));
}
"#,
    )
    .expect("write compute shader");
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 color, in vec2 fragCoord) { color = texture(iChannel0, vec2(0.5)); }",
    )
    .expect("write image shader");

    let project_arg = project.to_string_lossy().into_owned();
    let state_path = temp.0.join("frame1.ststate");
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
    let state_json: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse state inspection");
    assert_eq!(state_json["header"]["format"], 3);
    assert_eq!(
        state_json["header"]["buffer_formats"]["simulation"],
        "rg32f"
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
        "-o",
        &resumed_arg,
    ]);
    assert!(resumed.status.success(), "{resumed:?}");
    let resumed_image = image::open(&resumed_path)
        .expect("open resumed image")
        .to_rgb8();
    let resumed_pixel = resumed_image.get_pixel(0, 0).0;
    assert!(
        (i16::from(resumed_pixel[0]) - 77).abs() <= 1,
        "{resumed_pixel:?}"
    );

    let direct_path = temp.0.join("direct.png");
    let direct_arg = direct_path.to_string_lossy().into_owned();
    let direct = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "2",
        "-o",
        &direct_arg,
    ]);
    assert!(direct.status.success(), "{direct:?}");
    let direct_image = image::open(&direct_path)
        .expect("open direct image")
        .to_rgb8();
    assert_eq!(resumed_image, direct_image);
}

#[cfg(target_os = "linux")]
#[test]
fn state_restore_rejects_typed_buffer_format_changes() {
    let temp = TempRoot::new();
    let project = temp.0.join("format-state-project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shaders");
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "typed-state-format"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[pass]]
name = "simulation"
kind = "compute"
source = "shaders/simulation.comp"
width = 1
height = 1
format = "rg32f"
local_size = [1, 1, 1]

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "simulation"
frame = "current"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        project.join("shaders/simulation.comp"),
        "void mainCompute(ivec2 coord) { imageStore(iOutput, coord, vec4(0.25, 0.75, 0.0, 1.0)); }",
    )
    .expect("write compute shader");
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 color, in vec2 fragCoord) { color = texture(iChannel0, vec2(0.5)); }",
    )
    .expect("write image shader");

    let project_arg = project.to_string_lossy().into_owned();
    let state_path = temp.0.join("typed.ststate");
    let state_arg = state_path.to_string_lossy().into_owned();
    let captured = shadertoy(&[
        "state",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "-o",
        &state_arg,
    ]);
    assert!(captured.status.success(), "{captured:?}");

    let manifest_path = project.join("ShaderToy.toml");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
    std::fs::write(
        &manifest_path,
        manifest.replacen("format = \"rg32f\"", "format = \"r32f\"", 1),
    )
    .expect("change compute output format");

    let resumed = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--state",
        &state_arg,
        "--frame",
        "1",
    ]);
    assert!(
        !resumed.status.success(),
        "restoring typed state into a different render format must be rejected: {resumed:?}"
    );
    let stderr = String::from_utf8_lossy(&resumed.stderr);
    assert!(
        stderr.contains("format") && stderr.contains("simulation"),
        "{stderr}"
    );
}
