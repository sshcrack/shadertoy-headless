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

#[cfg(target_os = "linux")]
#[test]
fn render_applies_declared_custom_uniform_defaults_and_overrides() {
    let temp = TempRoot::new("custom-uniform-render");
    let project = temp.path().join("demo");
    std::fs::create_dir_all(project.join("shaders")).unwrap();
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1
[project]
name = "uniform-test"
[render]
width = 1
height = 1
fps = 60.0
preview_time = 0.0

[[uniform]]
name = "gain"
type = "float"
default = 0.25
min = 0.0
max = 1.0
step = 0.05

[[uniform]]
name = "enabled"
type = "bool"
default = true

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"
"#,
    )
    .unwrap();
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(enabled ? gain : 0.0, 0.0, 0.0, 1.0); }
",
    )
    .unwrap();

    let project_arg = project.to_string_lossy().into_owned();
    let default_path = temp.path().join("default.png");
    let default_arg = default_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "-o",
        &default_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let default = image::open(&default_path).unwrap().to_rgb8();
    assert!((60..=70).contains(&default.get_pixel(0, 0)[0]));

    let override_path = temp.path().join("override.png");
    let override_arg = override_path.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set",
        "gain=0.75",
        "-o",
        &override_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let overridden = image::open(&override_path).unwrap().to_rgb8();
    assert!((185..=195).contains(&overridden.get_pixel(0, 0)[0]));

    let disabled_path = temp.path().join("disabled.png");
    let disabled_arg = disabled_path.to_string_lossy().into_owned();
    let disabled = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set",
        "enabled=false",
        "-o",
        &disabled_arg,
    ]);
    assert!(disabled.status.success(), "{disabled:?}");
    let disabled = image::open(&disabled_path).unwrap().to_rgb8();
    assert!(disabled.get_pixel(0, 0)[0] <= 1);

    let rejected = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set",
        "gain=2.0",
    ]);
    assert!(!rejected.status.success());
}

#[cfg(target_os = "linux")]
#[test]
fn render_frames_accepts_ranges_and_render_video_encodes_one_runtime_sequence() {
    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| !output.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping video integration test because ffmpeg is unavailable");
        return;
    }

    let temp = TempRoot::new("video-render");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();
    let created = shadertoy(&["new", &project_arg]);
    assert!(created.status.success(), "{created:?}");

    let frames_dir = temp.path().join("range-frames");
    let frames_dir_arg = frames_dir.to_string_lossy().into_owned();
    let ranged = shadertoy(&[
        "render-frames",
        "--project",
        &project_arg,
        "--width",
        "16",
        "--height",
        "16",
        "--range",
        "0:2",
        "--output-dir",
        &frames_dir_arg,
    ]);
    assert!(ranged.status.success(), "{ranged:?}");
    for frame in 0..=2 {
        assert!(frames_dir.join(format!("frame-{frame:06}.png")).is_file());
    }

    let video = temp.path().join("clip.mp4");
    let video_arg = video.to_string_lossy().into_owned();
    let encoded = shadertoy(&[
        "render-video",
        "--project",
        &project_arg,
        "--width",
        "16",
        "--height",
        "16",
        "--fps",
        "30",
        "--frames",
        "4",
        "-o",
        &video_arg,
    ]);
    assert!(encoded.status.success(), "{encoded:?}");
    assert!(
        std::fs::metadata(&video).unwrap().len() > 0,
        "video output is empty"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn sound_pass_checks_and_renders_deterministic_pcm_wav() {
    let temp = TempRoot::new("sound-render");
    let project = temp.path().join("demo");
    let project_arg = project.to_string_lossy().into_owned();
    assert!(shadertoy(&["new", &project_arg]).status.success());

    let added = shadertoy(&[
        "pass",
        "--project",
        &project_arg,
        "add",
        "tone",
        "--kind",
        "sound",
    ]);
    assert!(added.status.success(), "{added:?}");

    let source = "shaders/tone.frag";
    std::fs::write(
        project.join(source),
        "vec2 mainSound(int samp, float time) { return vec2(0.25, -0.5); }
",
    )
    .unwrap();

    let checked = shadertoy(&["--json", "check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["sound_passes"], 1);

    let built = shadertoy(&["build", "--project", &project_arg]);
    assert!(
        !built.status.success(),
        "Sound must not be silently omitted from STTF"
    );

    let output = temp.path().join("tone.wav");
    let output_arg = output.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render-audio",
        "--project",
        &project_arg,
        "--pass",
        "tone",
        "--duration",
        "0.01",
        "--sample-rate",
        "8000",
        "-o",
        &output_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");

    let wav = std::fs::read(output).unwrap();
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        8000
    );
    let left = i16::from_le_bytes([wav[44], wav[45]]);
    let right = i16::from_le_bytes([wav[46], wav[47]]);
    assert!((i32::from(left) - 8192).abs() <= 1, "{left}");
    assert!((i32::from(right) + 16384).abs() <= 1, "{right}");
}

#[cfg(target_os = "linux")]
#[test]
fn video_asset_tracks_deterministic_render_time() {
    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| !output.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping video-channel integration test because ffmpeg is unavailable");
        return;
    }

    let temp = TempRoot::new("video-channel");
    let project = temp.path().join("demo");
    std::fs::create_dir_all(project.join("shaders")).unwrap();
    std::fs::create_dir_all(project.join("assets")).unwrap();

    let frames = temp.path().join("video-frames");
    std::fs::create_dir_all(&frames).unwrap();
    let mut red = image::RgbaImage::new(4, 4);
    red.pixels_mut()
        .for_each(|pixel| *pixel = image::Rgba([255, 0, 0, 255]));
    red.save(frames.join("frame-0.png")).unwrap();
    let mut green = image::RgbaImage::new(4, 4);
    green
        .pixels_mut()
        .for_each(|pixel| *pixel = image::Rgba([0, 255, 0, 255]));
    green.save(frames.join("frame-1.png")).unwrap();

    let video = project.join("assets/input.mp4");
    let encoded = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-i",
        ])
        .arg(frames.join("frame-%d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&video)
        .output()
        .unwrap();
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );

    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1
[project]
name = "video-channel"

[render]
width = 4
height = 4
fps = 1.0
preview_time = 0.0

[[asset]]
name = "clip"
kind = "video"
path = "assets/input.mp4"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "clip"
kind = "video"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .unwrap();
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p){ c=texture(iChannel0,p/iResolution.xy); }
",
    )
    .unwrap();

    let project_arg = project.to_string_lossy().into_owned();
    let built = shadertoy(&["build", "--project", &project_arg]);
    assert!(
        !built.status.success(),
        "dynamic video must not be silently frozen into STTF"
    );

    let frame0 = temp.path().join("frame0.png");
    let frame0_arg = frame0.to_string_lossy().into_owned();
    let first = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "-o",
        &frame0_arg,
    ]);
    assert!(first.status.success(), "{first:?}");

    let frame1 = temp.path().join("frame1.png");
    let frame1_arg = frame1.to_string_lossy().into_owned();
    let second = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "1",
        "-o",
        &frame1_arg,
    ]);
    assert!(second.status.success(), "{second:?}");

    let frame2 = temp.path().join("frame2.png");
    let frame2_arg = frame2.to_string_lossy().into_owned();
    let looped = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--frame",
        "2",
        "-o",
        &frame2_arg,
    ]);
    assert!(looped.status.success(), "{looped:?}");

    let first = image::open(frame0).unwrap().to_rgb8().get_pixel(1, 1).0;
    let second = image::open(frame1).unwrap().to_rgb8().get_pixel(1, 1).0;
    let looped = image::open(frame2).unwrap().to_rgb8().get_pixel(1, 1).0;
    assert!(first[0] > 200 && first[1] < 60, "{first:?}");
    assert!(second[1] > 180 && second[0] < 80, "{second:?}");
    assert!(looped[0] > 200 && looped[1] < 60, "{looped:?}");
}

#[cfg(target_os = "linux")]
#[test]
fn webcam_input_is_valid_for_preview_but_rejected_headlessly() {
    let temp = TempRoot::new("webcam-input");
    let project = temp.path().join("demo");
    std::fs::create_dir_all(project.join("shaders")).unwrap();
    std::fs::write(
        project.join("ShaderToy.toml"),
        r#"format = 1
[project]
name = "webcam-input"

[render]
width = 8
height = 8
fps = 30.0
preview_time = 0.0

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "webcam"
kind = "webcam"
filter = "linear"
wrap = "clamp"
"#,
    )
    .unwrap();
    std::fs::write(
        project.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p){ c=texture(iChannel0,p/iResolution.xy); }
",
    )
    .unwrap();

    let project_arg = project.to_string_lossy().into_owned();
    let checked = shadertoy(&["check", "--project", &project_arg]);
    assert!(checked.status.success(), "{checked:?}");

    let rendered = shadertoy(&["render", "--project", &project_arg, "--frame", "0"]);
    assert!(!rendered.status.success(), "{rendered:?}");
    let stderr = String::from_utf8_lossy(&rendered.stderr);
    assert!(stderr.contains("live preview-only"), "{stderr}");
}
