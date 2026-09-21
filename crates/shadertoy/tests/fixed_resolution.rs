use shadertoy::{Filter, HeadlessContext, InputKind, PassKind, Project, Runtime, Wrap};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
static EGL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(target_os = "linux")]
#[test]
fn fixed_buffer_keeps_resolution_and_feedback_across_output_resize() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(64, 64).expect("create headless OpenGL context");
    let mut project = Project::new("fixed-resolution").expect("create project");

    project
        .add_pass(
            "spectrum",
            PassKind::Buffer,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (iFrame == 0) {
        fragColor = vec4(0.25, iResolution.x / 10.0, iResolution.y / 10.0, 1.0);
    } else {
        vec4 previous = texture(iChannel0, fragCoord / iResolution.xy);
        fragColor = previous + vec4(0.10, 0.0, 0.0, 0.0);
    }
}
"#,
        )
        .expect("add buffer")
        .set_pass_resolution("spectrum", 2, 2)
        .expect("set fixed buffer size")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(iChannelResolution[0].xy / 10.0, 0.0, 1.0);
}
"#,
        )
        .expect("add image")
        .add_input(
            "spectrum",
            0,
            InputKind::Pass,
            "spectrum",
            true,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("add feedback")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "spectrum",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("add image input");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");

    let first = runtime.render(4, 3).expect("render first frame");
    assert_eq!((first.width, first.height), (4, 3));
    let first_state = runtime
        .snapshot_pass_rgba32f("spectrum", 2, 2)
        .expect("snapshot fixed buffer");
    assert!((first_state[0] - 0.25).abs() < 1e-5);
    assert!((first_state[1] - 0.20).abs() < 1e-5);
    assert!((first_state[2] - 0.20).abs() < 1e-5);

    runtime.tick_fixed(1.0 / 60.0, 60.0).expect("advance frame");
    let resized = runtime.render(9, 5).expect("render resized output");
    assert_eq!((resized.width, resized.height), (9, 5));

    let second_state = runtime
        .snapshot_pass_rgba32f("spectrum", 2, 2)
        .expect("snapshot fixed buffer after resize");
    assert!((second_state[0] - 0.35).abs() < 1e-5);
    assert!((second_state[1] - 0.20).abs() < 1e-5);
    assert!((second_state[2] - 0.20).abs() < 1e-5);

    // The final pass reports the actual 2x2 input dimensions through
    // iChannelResolution even though the output itself is 9x5.
    let pixel = &resized.pixels[..3];
    assert!((i16::from(pixel[0]) - 51).abs() <= 1, "red={}", pixel[0]);
    assert!((i16::from(pixel[1]) - 51).abs() <= 1, "green={}", pixel[1]);
}

#[test]
fn fixed_resolution_rejects_dimensions_outside_opengl_range() {
    let mut project = Project::new("fixed-resolution-range").expect("create project");
    project
        .add_pass(
            "buffer",
            PassKind::Buffer,
            "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.0); }",
        )
        .expect("add buffer");

    let too_large = (i32::MAX as u32) + 1;
    let error = match project.set_pass_resolution("buffer", too_large, 1) {
        Ok(_) => panic!("oversized fixed pass must fail before OpenGL allocation"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("OpenGL dimension range"),
        "{error}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn pass_reload_preserves_feedback_and_is_transactional() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(64, 64).expect("create headless OpenGL context");
    let mut project = Project::new("reload-feedback").expect("create project");

    project
        .add_pass(
            "feedback",
            PassKind::Buffer,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (iFrame == 0) {
        fragColor = vec4(0.20, 0.0, 0.0, 1.0);
    } else {
        fragColor = texture(iChannel0, fragCoord / iResolution.xy) + vec4(0.10, 0.0, 0.0, 0.0);
    }
}
"#,
        )
        .expect("add feedback pass")
        .set_pass_resolution("feedback", 2, 2)
        .expect("set feedback resolution")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = texture(iChannel0, fragCoord / iResolution.xy);
}
"#,
        )
        .expect("add image")
        .add_input(
            "feedback",
            0,
            InputKind::Pass,
            "feedback",
            true,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("add feedback input")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "feedback",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("add image input");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");

    runtime.render(8, 8).expect("render frame 0");
    runtime.tick_fixed(1.0 / 60.0, 60.0).expect("advance frame");
    runtime.render(8, 8).expect("render frame 1");
    let before = runtime
        .snapshot_pass_rgba32f("feedback", 2, 2)
        .expect("snapshot before reload");
    assert!((before[0] - 0.30).abs() < 1e-5);

    let bad = runtime.reload_pass_source(
        "feedback",
        "void mainImage(out vec4 c, in vec2 p) { syntax error }",
    );
    assert!(
        bad.is_err(),
        "invalid replacement must fail transactionally"
    );

    runtime
        .reload_pass_source(
            "feedback",
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (iFrame == 0) {
        fragColor = vec4(0.20, 0.0, 0.0, 1.0);
    } else {
        fragColor = texture(iChannel0, fragCoord / iResolution.xy) + vec4(0.20, 0.0, 0.0, 0.0);
    }
}
"#,
        )
        .expect("reload feedback source");

    runtime
        .tick_fixed(1.0 / 60.0, 60.0)
        .expect("advance after reload");
    runtime.render(8, 8).expect("render after reload");
    let after = runtime
        .snapshot_pass_rgba32f("feedback", 2, 2)
        .expect("snapshot after reload");
    assert!(
        (after[0] - 0.50).abs() < 1e-5,
        "feedback should survive source replacement, got {}",
        after[0]
    );
}

#[cfg(target_os = "linux")]
#[test]
fn profiling_reports_attributed_compute_and_image_gpu_timings() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(64, 64).expect("create headless OpenGL context");
    let mut project = Project::new("profiling").expect("create project");
    project
        .add_pass(
            "compute",
            PassKind::Compute,
            r#"
void mainCompute(ivec2 coord) {
    float value = float(coord.x + coord.y) * 0.001;
    for (int i = 0; i < 256; ++i)
        value = sin(value + float(i) * 0.0001);
    imageStore(iOutput, coord, vec4(value, value, value, 1.0));
}
"#,
        )
        .expect("add compute")
        .set_pass_resolution("compute", 256, 256)
        .expect("set compute dimensions")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = texture(iChannel0, fragCoord / iResolution.xy);
}
"#,
        )
        .expect("add image")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "compute",
            false,
            Filter::Linear,
            Wrap::Clamp,
        )
        .expect("wire compute result");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    runtime.set_profiling(true).expect("enable profiling");
    runtime.render(32, 16).expect("render profiled frame");

    let timings = runtime.pass_timings().expect("read timings");
    assert_eq!(timings.len(), 2);
    let compute = timings
        .iter()
        .find(|timing| timing.name == "compute")
        .expect("compute timing");
    let image = timings
        .iter()
        .find(|timing| timing.name == "image")
        .expect("image timing");
    assert_eq!((compute.width, compute.height), (256, 256));
    assert_eq!((image.width, image.height), (32, 16));
    assert!(
        compute.gpu_nanoseconds > 0,
        "active compute work must be attributed to the compute pass"
    );
    assert!(image.gpu_nanoseconds > 0);
}
