use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-mrt-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp root");
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

#[cfg(target_os = "linux")]
#[test]
fn manifest_mrt_output_is_rendered_and_consumed() {
    let temp = TempRoot::new();
    let project = temp.0.join("project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shaders");

    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "mrt-cli"

[render]
width = 2
height = 2
fps = 60.0
preview_time = 0.0

[[pass]]
name = "gbuffer"
kind = "buffer"
source = "shaders/gbuffer.frag"
format = "rgba16f"
extra_outputs = ["rg32f"]

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "gbuffer"
output = 1
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");

    std::fs::write(
        project.join("shaders/gbuffer.frag"),
        r#"layout(location = 1) out vec4 auxiliary;

void mainImage(out vec4 color, in vec2 fragCoord) {
    color = vec4(0.1, 0.2, 0.3, 1.0);
    auxiliary = vec4(0.75, 0.25, 0.0, 1.0);
}
"#,
    )
    .expect("write gbuffer");

    std::fs::write(
        project.join("shaders/image.frag"),
        r#"void mainImage(out vec4 color, in vec2 fragCoord) {
    vec2 value = texture(iChannel0, fragCoord / iResolution.xy).rg;
    color = vec4(value, 0.0, 1.0);
}
"#,
    )
    .expect("write image");

    let project_arg = project.to_string_lossy().into_owned();
    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let output = temp.0.join("mrt.png");
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

    let image = image::open(&output).expect("open output").to_rgb8();
    let pixel = image.get_pixel(0, 0).0;
    assert!((i16::from(pixel[0]) - 191).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 64).abs() <= 1, "{pixel:?}");
    assert!(pixel[2] <= 1, "{pixel:?}");
}

#[test]
fn channel_set_can_select_pass_output() {
    let temp = TempRoot::new();
    let project = temp.0.join("project");
    let project_arg = project.to_string_lossy().into_owned();
    assert!(
        shadertoy(&["new", &project_arg, "--template", "multipass"])
            .status
            .success()
    );

    let manifest_path = project.join("ShaderToy.toml");
    let mut manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
    manifest = manifest.replacen(
        "kind = \"buffer\"",
        "kind = \"buffer\"\nextra_outputs = [\"rg32f\"]",
        1,
    );
    std::fs::write(&manifest_path, manifest).expect("write manifest");

    let set = shadertoy(&[
        "channel",
        "--project",
        &project_arg,
        "set",
        "image",
        "0",
        "buffer-a",
        "--output",
        "1",
        "--filter",
        "nearest",
        "--wrap",
        "clamp",
    ]);
    assert!(set.status.success(), "{set:?}");

    let manifest = std::fs::read_to_string(&manifest_path).expect("read updated manifest");
    assert!(manifest.contains("output = 1"), "{manifest}");
}
