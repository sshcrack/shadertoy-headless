#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-cli-agent-workflows-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temporary test directory");
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

fn write_color_project(root: &Path, name: &str, gain: f32, with_unused: bool) {
    std::fs::create_dir_all(root.join("shaders")).expect("create shaders");
    let unused = if with_unused {
        r#"
[[pass]]
name = "unused"
kind = "buffer"
source = "shaders/unused.frag"
"#
    } else {
        ""
    };
    std::fs::write(
        root.join("ShaderToy.toml"),
        format!(
            r#"format = 1

[project]
name = "{name}"

[render]
width = 8
height = 8
fps = 60.0
preview_time = 0.0

[[uniform]]
name = "gain"
type = "float"
default = {gain}
min = 0.0
max = 1.0

[[pass]]
name = "buffer"
kind = "buffer"
source = "shaders/buffer.frag"
width = 8
height = 8

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "buffer"
kind = "pass"
frame = "current"
filter = "nearest"
wrap = "clamp"
{unused}
"#
        ),
    )
    .expect("write manifest");
    std::fs::write(
        root.join("shaders/buffer.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(gain, 0.0, 0.0, 1.0); }\n",
    )
    .expect("write buffer");
    std::fs::write(
        root.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, p / iResolution.xy); }\n",
    )
    .expect("write image");
    if with_unused {
        std::fs::write(
            root.join("shaders/unused.frag"),
            "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.0); }\n",
        )
        .expect("write unused");
    }
}

#[test]
fn graph_exports_dot_and_pedantic_check_surfaces_unreachable_pass() {
    let temp = TempRoot::new("graph");
    let project = temp.0.join("project");
    write_color_project(&project, "graph-test", 0.25, true);
    let project_arg = project.to_string_lossy().into_owned();
    let dot = temp.0.join("graph.dot");
    let dot_arg = dot.to_string_lossy().into_owned();

    let graph = shadertoy(&[
        "--json",
        "graph",
        "--project",
        &project_arg,
        "--dot",
        &dot_arg,
    ]);
    assert!(graph.status.success(), "{graph:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&graph.stdout).expect("parse graph json");
    assert!(
        report["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|diagnostic| diagnostic["code"] == "unreachable-pass")
    );
    assert!(
        std::fs::read_to_string(dot)
            .unwrap()
            .contains("digraph ShaderToy")
    );

    let checked = shadertoy(&["--json", "check", "--project", &project_arg, "--pedantic"]);
    assert_eq!(checked.status.code(), Some(1), "{checked:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("parse check json");
    assert_eq!(report["compiled"], true);
    assert!(report["warnings"].as_u64().unwrap() >= 1);
}

#[test]
fn experiment_compares_arbitrary_projects_with_metrics() {
    let temp = TempRoot::new("experiment");
    let baseline = temp.0.join("baseline");
    let candidate = temp.0.join("candidate");
    write_color_project(&baseline, "baseline", 0.2, false);
    write_color_project(&candidate, "candidate", 0.8, false);

    let baseline_source = format!("project:{}", baseline.display());
    let candidate_source = format!("project:{}", candidate.display());
    let output = temp.0.join("experiment");
    let output_arg = output.to_string_lossy().into_owned();
    let run = shadertoy(&[
        "--json",
        "experiment",
        "--baseline",
        &baseline_source,
        "--candidate",
        &candidate_source,
        "--frames",
        "0",
        "--profile-samples",
        "0",
        "--output-dir",
        &output_arg,
    ]);
    assert!(run.status.success(), "{run:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&run.stdout).expect("parse experiment json");
    assert_eq!(report["variant_count"], 2);
    let comparison = &report["comparisons"][0]["aggregate"];
    assert!(comparison["rmse_mean"].as_f64().unwrap() > 0.1);
    assert!(comparison["ssim_mean"].as_f64().unwrap() < 1.0);
    assert!(output.join("contact-sheet.png").is_file());
    assert!(output.join("experiment-report.json").is_file());
}

#[test]
fn trace_capture_inspect_and_replay_are_self_contained() {
    let temp = TempRoot::new("trace");
    let project = temp.0.join("project");
    write_color_project(&project, "trace-test", 0.375, false);
    let project_arg = project.to_string_lossy().into_owned();
    let trace = temp.0.join("capture.sttrace");
    let trace_arg = trace.to_string_lossy().into_owned();

    let capture = shadertoy(&[
        "--json",
        "trace",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "2",
        "--set",
        "gain=0.75",
        "--include-intermediates",
        "-o",
        &trace_arg,
    ]);
    assert!(capture.status.success(), "{capture:?}");
    assert!(trace.join("trace.json").is_file());
    assert!(trace.join("project.sttf").is_file());
    assert!(trace.join("state.ststate").is_file());

    let inspect = shadertoy(&["--json", "trace", "inspect", &trace_arg]);
    assert!(inspect.status.success(), "{inspect:?}");
    let inspected: serde_json::Value =
        serde_json::from_slice(&inspect.stdout).expect("parse trace inspect json");
    assert_eq!(inspected["verified"], true);

    let replay = shadertoy(&["--json", "trace", "replay", &trace_arg]);
    assert!(replay.status.success(), "{replay:?}");
    let replayed: serde_json::Value =
        serde_json::from_slice(&replay.stdout).expect("parse trace replay json");
    assert_eq!(replayed["exact"], true);
    assert_eq!(replayed["state_valid"], true);
}

#[test]
fn test_supports_uniform_comparisons_gpu_budgets_and_state_roundtrip() {
    let temp = TempRoot::new("test-framework");
    let project = temp.0.join("project");
    write_color_project(&project, "test-framework", 0.25, false);
    let manifest = project.join("ShaderToy.toml");
    let mut text = std::fs::read_to_string(&manifest).unwrap();
    text.push_str(
        r#"

[[test]]
name = "variant-diff"
pass = "buffer"
frame = 0
uniforms = { gain = 0.8 }
reference_uniforms = { gain = 0.2 }
min_rmse = 0.1
max_gpu_ms = 1000.0
max_pass_gpu_ms = { buffer = 1000.0, image = 1000.0 }
assert_state_roundtrip = true
assert_deterministic = true
"#,
    );
    std::fs::write(&manifest, text).unwrap();
    let project_arg = project.to_string_lossy().into_owned();

    let tested = shadertoy(&["--json", "test", "--project", &project_arg, "--ci"]);
    assert!(tested.status.success(), "{tested:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&tested.stdout).expect("parse test json");
    assert_eq!(report["passed"], 1);
    assert!(report["cases"][0]["rmse"].as_f64().unwrap() > 0.1);
    assert_eq!(report["cases"][0]["state_roundtrip"], true);
    assert!(report["cases"][0]["gpu_total_ms"].as_f64().unwrap() >= 0.0);
}

#[test]
fn test_supports_exact_ssbo_fixtures() {
    let temp = TempRoot::new("test-storage-fixture");
    let project = temp.0.join("project");
    std::fs::create_dir_all(project.join("shaders")).unwrap();
    std::fs::create_dir_all(project.join("tests")).unwrap();
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "storage-fixture"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[pass]]
name = "writer"
kind = "buffer"
source = "shaders/writer.frag"
width = 1
height = 1

[[pass.storage]]
binding = 0
name = "shared-data"
size = 16

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "writer"
kind = "pass"
frame = "current"
filter = "nearest"
wrap = "clamp"

[[test]]
name = "ssbo-fixture"
pass = "writer"
frame = 0
assert_state_roundtrip = true

[[test.storage]]
name = "shared-data"
reference = "tests/shared-data.bin"
"#,
    )
    .unwrap();
    std::fs::write(
        project.join("shaders/writer.frag"),
        r#"layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 c, in vec2 p) {
    value = vec4(0.125, 0.25, 0.5, 1.0);
    c = value;
}
"#,
    )
    .unwrap();
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, vec2(0.5)); }\n",
    )
    .unwrap();
    let mut fixture = Vec::new();
    for value in [0.125f32, 0.25, 0.5, 1.0] {
        fixture.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(project.join("tests/shared-data.bin"), fixture).unwrap();

    let project_arg = project.to_string_lossy().into_owned();
    let tested = shadertoy(&["--json", "test", "--project", &project_arg, "--ci"]);
    assert!(tested.status.success(), "{tested:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&tested.stdout).expect("parse storage fixture test json");
    assert_eq!(report["passed"], 1);
    assert_eq!(report["cases"][0]["storage_assertions"], 1);
    assert_eq!(report["cases"][0]["state_roundtrip"], true);
}
