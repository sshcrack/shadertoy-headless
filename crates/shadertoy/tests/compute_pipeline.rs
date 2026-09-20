use shadertoy::{
    Filter, HeadlessContext, InputKind, PassKind, Project, RenderFormat, Runtime, Wrap,
};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
static EGL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(target_os = "linux")]
#[test]
fn compute_pass_supports_ssbo_iterations_and_typed_output() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("compute-pipeline").expect("create project");

    project
        .add_pass(
            "simulation",
            PassKind::Compute,
            r#"
layout(std430, binding = 3) buffer Counters {
    uint counters[];
};

void mainCompute(ivec2 coord) {
    uint index = uint(coord.y) * uint(iResolution.x) + uint(coord.x);
    counters[index] += 1u;
    imageStore(
        iOutput,
        coord,
        vec4(float(counters[index]) / 10.0, float(iIteration) / 10.0, 0.75, 1.0)
    );
}
"#,
        )
        .expect("add compute pass")
        .set_pass_resolution("simulation", 4, 2)
        .expect("set compute dimensions")
        .set_pass_format("simulation", RenderFormat::Rg32f)
        .expect("set typed compute output")
        .set_pass_iterations("simulation", 3)
        .expect("set iterations")
        .set_compute_local_size("simulation", 4, 1, 1)
        .expect("set workgroup")
        .bind_storage_buffer("simulation", 3, "counters", 4 * 2 * 4)
        .expect("bind ssbo")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 value = texture(iChannel0, fragCoord / iResolution.xy).rg;
    fragColor = vec4(value, 0.25, 1.0);
}
"#,
        )
        .expect("add image pass")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "simulation",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("wire compute result");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(4, 2).expect("render compute project");

    let state = runtime
        .snapshot_pass_rgba32f("simulation", 4, 2)
        .expect("read compute output");
    assert!((state[0] - 0.3).abs() < 1e-5, "counter result={}", state[0]);
    assert!(
        (state[1] - 0.2).abs() < 1e-5,
        "iteration result={}",
        state[1]
    );
    assert!(state[2].abs() < 1e-6, "RG32F must discard blue");
    assert!((state[3] - 1.0).abs() < 1e-6, "missing alpha reads as one");

    let pixel = &image.pixels[..3];
    assert!((i16::from(pixel[0]) - 77).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 51).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[2]) - 64).abs() <= 1, "{pixel:?}");
}

#[cfg(target_os = "linux")]
#[test]
fn storage_buffers_are_shared_across_passes_and_support_atomics() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("shared-storage").expect("create project");

    project
        .add_pass(
            "seed",
            PassKind::Compute,
            r#"
layout(std430, binding = 2) buffer SharedState {
    uint value;
};
void mainCompute(ivec2 coord) {
    atomicAdd(value, 1u);
    imageStore(iOutput, coord, vec4(float(value), 0.0, 0.0, 1.0));
}
"#,
        )
        .expect("add seed")
        .set_pass_resolution("seed", 1, 1)
        .expect("seed size")
        .set_pass_format("seed", RenderFormat::R32f)
        .expect("seed format")
        .set_compute_local_size("seed", 1, 1, 1)
        .expect("seed group")
        .bind_storage_buffer("seed", 2, "shared-counter", 4)
        .expect("seed storage")
        .add_pass(
            "advance",
            PassKind::Compute,
            r#"
layout(std430, binding = 5) buffer SharedState {
    uint value;
};
void mainCompute(ivec2 coord) {
    atomicAdd(value, 4u);
    imageStore(iOutput, coord, vec4(float(value) / 10.0, 0.0, 0.0, 1.0));
}
"#,
        )
        .expect("add advance")
        .set_pass_resolution("advance", 1, 1)
        .expect("advance size")
        .set_pass_format("advance", RenderFormat::R32f)
        .expect("advance format")
        .set_compute_local_size("advance", 1, 1, 1)
        .expect("advance group")
        .bind_storage_buffer("advance", 5, "shared-counter", 4)
        .expect("advance storage")
        .add_input(
            "advance",
            0,
            InputKind::Pass,
            "seed",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("order compute passes")
        .add_pass(
            "image",
            PassKind::Image,
            "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, vec2(0.5)); }",
        )
        .expect("add image")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "advance",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("wire result");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(1, 1).expect("render");
    assert!(
        (i16::from(image.pixels[0]) - 128).abs() <= 1,
        "{:?}",
        image.pixels
    );
}

#[test]
fn compute_local_size_z_must_be_one_for_2d_entrypoint() {
    let mut project = Project::new("compute-local-z").expect("create project");
    project
        .add_pass(
            "simulation",
            PassKind::Compute,
            "void mainCompute(ivec2 coord) { imageStore(iOutput, coord, vec4(1.0)); }",
        )
        .expect("add compute pass")
        .set_pass_resolution("simulation", 1, 1)
        .expect("set compute dimensions");

    let error = match project.set_compute_local_size("simulation", 1, 1, 2) {
        Ok(_) => panic!("2D compute entrypoints must reject multiple local Z lanes"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("z"), "{error}");
    assert!(error.to_string().contains("1"), "{error}");
}
