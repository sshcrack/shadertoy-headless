use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "shadertoy-sampler-controls-{label}-{}-{}",
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
name = "sampler-controls-test"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[pass]]
name = "source"
kind = "buffer"
source = "shaders/source.frag"
width = 2
height = 1

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "source"
filter = "nearest"
wrap = "clamp"

[[pass.input]]
channel = 1
source = "source"
filter = "linear"
wrap = "clamp"

[[pass.input]]
channel = 2
source = "source"
filter = "nearest"
wrap = "clamp"

[[pass.input]]
channel = 3
source = "source"
filter = "nearest"
wrap = "repeat"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        root.join("shaders/source.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = fragCoord.x < 1.0
        ? vec4(1.0, 0.0, 0.0, 1.0)
        : vec4(0.0, 0.0, 1.0, 1.0);
}
"#,
    )
    .expect("write source shader");
    std::fs::write(
        root.join("shaders/image.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float nearestSample = texture(iChannel0, vec2(0.49, 0.5)).r;
    float linearSample = texture(iChannel1, vec2(0.49, 0.5)).r;
    float clampedSample = texture(iChannel2, vec2(1.25, 0.5)).r;
    float repeatedSample = texture(iChannel3, vec2(1.25, 0.5)).r;
    fragColor = vec4(
        nearestSample,
        linearSample,
        0.5 * clampedSample + 0.5 * repeatedSample,
        1.0
    );
}
"#,
    )
    .expect("write image shader");
}

#[cfg(target_os = "linux")]
#[test]
fn channels_can_sample_the_same_pass_with_independent_filter_and_wrap_modes() {
    let temp = TempRoot::new("independent");
    let project = temp.0.join("project");
    write_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let output_path = temp.0.join("samplers.png");
    let output_arg = output_path.to_string_lossy().into_owned();
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

    let image = image::open(&output_path)
        .expect("open sampler render")
        .to_rgb8();
    assert_eq!(image.dimensions(), (1, 1));
    let pixel = image.get_pixel(0, 0).0;

    // Nearest samples the left texel exactly.
    assert!(pixel[0] >= 250, "{pixel:?}");
    // Linear sampling at u=0.49 blends the red and blue texels about 52/48.
    assert!((i16::from(pixel[1]) - 133).abs() <= 4, "{pixel:?}");
    // Clamp reads the right (blue) edge while repeat wraps to the left (red)
    // texel, so their red values average to 0.5.
    assert!((i16::from(pixel[2]) - 128).abs() <= 3, "{pixel:?}");
}

#[cfg(target_os = "linux")]
#[test]
fn mipmap_filter_builds_and_samples_the_pass_mipmap_chain() {
    let temp = TempRoot::new("mipmap");
    let project = temp.0.join("project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shader directory");
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "sampler-mipmap-test"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[pass]]
name = "source"
kind = "buffer"
source = "shaders/source.frag"
width = 4
height = 4

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "source"
filter = "mipmap"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        project.join("shaders/source.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float checker = mod(floor(fragCoord.x) + floor(fragCoord.y), 2.0);
    fragColor = checker < 0.5
        ? vec4(1.0, 0.0, 0.0, 1.0)
        : vec4(0.0, 0.0, 1.0, 1.0);
}
"#,
    )
    .expect("write source shader");
    std::fs::write(
        project.join("shaders/image.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = textureLod(iChannel0, vec2(0.5), 2.0);
}
"#,
    )
    .expect("write image shader");

    let project_arg = project.to_string_lossy().into_owned();
    let output_path = temp.0.join("mipmap.png");
    let output_arg = output_path.to_string_lossy().into_owned();
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

    let image = image::open(&output_path)
        .expect("open mipmap render")
        .to_rgb8();
    let pixel = image.get_pixel(0, 0).0;
    assert!((i16::from(pixel[0]) - 128).abs() <= 3, "{pixel:?}");
    assert!(pixel[1] <= 3, "{pixel:?}");
    assert!((i16::from(pixel[2]) - 128).abs() <= 3, "{pixel:?}");
}

#[cfg(target_os = "linux")]
#[test]
fn extended_texture_channel_keeps_mipmap_repeat_and_inspection_support() {
    let temp = TempRoot::new("extended-texture");
    let project = temp.0.join("project");
    std::fs::create_dir_all(project.join("shaders")).expect("create shader directory");
    std::fs::create_dir_all(project.join("assets")).expect("create asset directory");

    let mut texture = image::RgbaImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            let pixel = if (x + y) % 2 == 0 {
                image::Rgba([255, 0, 0, 255])
            } else {
                image::Rgba([0, 0, 255, 255])
            };
            texture.put_pixel(x, y, pixel);
        }
    }
    texture
        .save(project.join("assets/checker.png"))
        .expect("save texture asset");

    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "extended-texture-channel"

[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[asset]]
name = "checker"
kind = "texture"
path = "assets/checker.png"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 12
source = "checker"
kind = "texture"
filter = "mipmap"
wrap = "repeat"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        project.join("shaders/image.frag"),
        r#"void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 repeated = textureLod(iChannel12, vec2(1.125, 0.125), 0.0);
    vec4 mip = textureLod(iChannel12, vec2(0.5), 2.0);
    float sizeOk = float(
        iChannelResolution[12].x > 3.5 && iChannelResolution[12].x < 4.5 &&
        iChannelResolution[12].y > 3.5 && iChannelResolution[12].y < 4.5
    );
    fragColor = vec4(repeated.r, mip.r, sizeOk, 1.0);
}
"#,
    )
    .expect("write image shader");

    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let graph = shadertoy(&["--json", "graph", "--project", &project_arg]);
    assert!(graph.status.success(), "{graph:?}");
    let graph_json: serde_json::Value =
        serde_json::from_slice(&graph.stdout).expect("parse graph json");
    assert_eq!(graph_json["edges"][0]["channel"], 12);

    let inspected = shadertoy(&[
        "--json",
        "inspect",
        "--project",
        &project_arg,
        "channels",
        "image",
    ]);
    assert!(inspected.status.success(), "{inspected:?}");
    let inspect_json: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse channel inspection json");
    assert_eq!(inspect_json["channels"][0]["channel"], 12);
    assert_eq!(inspect_json["channels"][0]["filter"], "mipmap");
    assert_eq!(inspect_json["channels"][0]["wrap"], "repeat");

    let inspected_pass = shadertoy(&[
        "--json",
        "inspect",
        "--project",
        &project_arg,
        "pass",
        "image",
    ]);
    assert!(inspected_pass.status.success(), "{inspected_pass:?}");
    let pass_json: serde_json::Value =
        serde_json::from_slice(&inspected_pass.stdout).expect("parse pass inspection json");
    assert_eq!(pass_json["pass"]["input"][0]["channel"], 12);
    assert_eq!(pass_json["pass"]["input"][0]["filter"], "mipmap");
    assert_eq!(pass_json["pass"]["input"][0]["wrap"], "repeat");

    let output_path = temp.0.join("extended.png");
    let output_arg = output_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--pass",
        "image",
        "--frame",
        "0",
        "-o",
        &output_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");

    let image = image::open(&output_path)
        .expect("open extended render")
        .to_rgb8();
    let pixel = image.get_pixel(0, 0).0;
    assert!(pixel[0] >= 250, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 128).abs() <= 4, "{pixel:?}");
    assert!(pixel[2] >= 250, "{pixel:?}");
}

#[test]
fn channel_set_and_remove_accept_channel_15() {
    let temp = TempRoot::new("channel-set-15");
    let project = temp.0.join("project");
    let project_arg = project.to_string_lossy().into_owned();

    let created = shadertoy(&["new", &project_arg, "--template", "multipass"]);
    assert!(created.status.success(), "{created:?}");

    let set = shadertoy(&[
        "channel",
        "--project",
        &project_arg,
        "set",
        "image",
        "15",
        "buffer-a",
        "--filter",
        "nearest",
        "--wrap",
        "repeat",
    ]);
    assert!(set.status.success(), "{set:?}");

    let manifest = std::fs::read_to_string(project.join("ShaderToy.toml")).expect("read manifest");
    assert!(manifest.contains("channel = 15"), "{manifest}");

    let remove = shadertoy(&[
        "channel",
        "--project",
        &project_arg,
        "remove",
        "image",
        "15",
    ]);
    assert!(remove.status.success(), "{remove:?}");
}
