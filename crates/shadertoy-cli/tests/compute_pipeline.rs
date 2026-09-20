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
