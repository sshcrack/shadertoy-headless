use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-cli-storage-{}-{}",
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

fn write_shared_storage_project(root: &Path) {
    std::fs::create_dir_all(root.join("shaders")).expect("create shaders directory");
    std::fs::write(
        root.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "shared-storage-cli-repro"

[render]
width = 2
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

[[pass.storage]]
binding = 0
name = "shared-data"
size = 16

[[pass.input]]
channel = 0
source = "writer"
frame = "current"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");

    std::fs::write(
        root.join("shaders/writer.frag"),
        r#"layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (iFrame == 0)
        value = vec4(0.125, 0.25, 0.5, 1.0);
    fragColor = value;
}
"#,
    )
    .expect("write writer shader");

    std::fs::write(
        root.join("shaders/image.frag"),
        r#"layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 textureValue = texture(iChannel0, vec2(0.5));
    fragColor = fragCoord.x < 1.0 ? textureValue : value;
}
"#,
    )
    .expect("write image shader");
}

#[cfg(target_os = "linux")]
#[test]
fn cli_render_shares_named_ssbo_between_fragment_passes() {
    let temp = TempRoot::new();
    let project = temp.0.join("project");
    write_shared_storage_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg, "--json"]);
    assert!(checked.status.success(), "{checked:?}");

    let output = temp.0.join("result.png");
    let output_arg = output.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "-o",
        &output_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");

    let image = image::open(&output).expect("open render").to_rgb8();
    assert_eq!(image.dimensions(), (2, 1));
    let texture = image.get_pixel(0, 0).0;
    let storage = image.get_pixel(1, 0).0;
    for actual in [texture, storage] {
        assert!((i16::from(actual[0]) - 32).abs() <= 1, "{actual:?}");
        assert!((i16::from(actual[1]) - 64).abs() <= 1, "{actual:?}");
        assert!((i16::from(actual[2]) - 128).abs() <= 1, "{actual:?}");
    }

    let inspected = shadertoy(&[
        "--json",
        "inspect",
        "--project",
        &project_arg,
        "storage",
        "shared-data",
        "--frame",
        "0",
        "--type",
        "f32",
        "--count",
        "4",
    ]);
    assert!(inspected.status.success(), "{inspected:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse storage inspection");
    let values = report["values"].as_array().expect("storage values");
    assert_eq!(values.len(), 4);
    for (actual, expected) in values.iter().zip([0.125, 0.25, 0.5, 1.0]) {
        let actual = actual.as_f64().expect("finite f32 value");
        assert!((actual - expected).abs() < 1e-6, "{report}");
    }

    let state = temp.0.join("with-storage.ststate");
    let state_arg = state.to_string_lossy().into_owned();
    let captured = shadertoy(&[
        "state",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--include-storage",
        "-o",
        &state_arg,
    ]);
    assert!(captured.status.success(), "{captured:?}");

    let inspected_state = shadertoy(&["--json", "state", "inspect", &state_arg]);
    assert!(inspected_state.status.success(), "{inspected_state:?}");
    let header: serde_json::Value =
        serde_json::from_slice(&inspected_state.stdout).expect("parse state header");
    assert_eq!(header["header"]["format"], 4);
    assert_eq!(header["header"]["storage_buffers"]["shared-data"], 16);

    let replacement = temp.0.join("replacement.bin");
    let mut replacement_bytes = Vec::new();
    for value in [0.75f32, 0.5, 0.25, 1.0] {
        replacement_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&replacement, replacement_bytes).unwrap();
    let replacement_assignment = format!("shared-data={}", replacement.to_string_lossy());
    let modified = temp.0.join("modified.ststate");
    let modified_arg = modified.to_string_lossy().into_owned();
    let set = shadertoy(&[
        "state",
        "set-storage",
        &state_arg,
        &replacement_assignment,
        "-o",
        &modified_arg,
    ]);
    assert!(set.status.success(), "{set:?}");

    let resumed_path = temp.0.join("resumed.png");
    let resumed_arg = resumed_path.to_string_lossy().into_owned();
    let resumed = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--state",
        &modified_arg,
        "--frame",
        "1",
        "-o",
        &resumed_arg,
    ]);
    assert!(resumed.status.success(), "{resumed:?}");
    let resumed_image = image::open(&resumed_path).unwrap().to_rgb8();
    let pixel = resumed_image.get_pixel(1, 0).0;
    assert!((i16::from(pixel[0]) - 191).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 128).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[2]) - 64).abs() <= 1, "{pixel:?}");
}
