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
}
