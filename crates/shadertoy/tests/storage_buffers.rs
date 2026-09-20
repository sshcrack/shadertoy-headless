use shadertoy::{
    Filter, HeadlessContext, InputKind, PassKind, Project, RenderFormat, Runtime, Wrap,
};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
static EGL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn assert_rgb_near(actual: &[u8], expected: [u8; 3]) {
    for (component, expected) in actual.iter().zip(expected) {
        assert!(
            (i16::from(*component) - i16::from(expected)).abs() <= 1,
            "actual={actual:?} expected={expected:?}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn fragment_writer_shares_storage_with_fragment_reader() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("fragment-storage-sharing").expect("create project");

    project
        .add_pass(
            "writer",
            PassKind::Buffer,
            r#"
layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    value = vec4(0.125, 0.25, 0.5, 1.0);
    fragColor = value;
}
"#,
        )
        .expect("add writer")
        .set_pass_resolution("writer", 1, 1)
        .expect("set writer size")
        .bind_storage_buffer("writer", 0, "shared-data", 16)
        .expect("bind writer storage")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 textureValue = texture(iChannel0, vec2(0.5));
    fragColor = fragCoord.x < 1.0 ? textureValue : value;
}
"#,
        )
        .expect("add image")
        .bind_storage_buffer("image", 0, "shared-data", 16)
        .expect("bind image storage")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "writer",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("order writer before image");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(2, 1).expect("render");
    assert_rgb_near(&image.pixels[0..3], [32, 64, 128]);
    assert_rgb_near(&image.pixels[3..6], [32, 64, 128]);
}

#[cfg(target_os = "linux")]
#[test]
fn compute_writer_shares_storage_with_fragment_reader() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("compute-fragment-storage-sharing").expect("create project");

    project
        .add_pass(
            "writer",
            PassKind::Compute,
            r#"
layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainCompute(ivec2 coord) {
    value = vec4(0.125, 0.25, 0.5, 1.0);
    imageStore(iOutput, coord, value);
}
"#,
        )
        .expect("add compute writer")
        .set_pass_resolution("writer", 1, 1)
        .expect("set writer size")
        .set_pass_format("writer", RenderFormat::Rgba32f)
        .expect("set writer format")
        .set_compute_local_size("writer", 1, 1, 1)
        .expect("set workgroup")
        .bind_storage_buffer("writer", 0, "shared-data", 16)
        .expect("bind writer storage")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
layout(std430, binding = 0) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 textureValue = texture(iChannel0, vec2(0.5));
    fragColor = fragCoord.x < 1.0 ? textureValue : value;
}
"#,
        )
        .expect("add image")
        .bind_storage_buffer("image", 0, "shared-data", 16)
        .expect("bind image storage")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "writer",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("order writer before image");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(2, 1).expect("render");
    assert_rgb_near(&image.pixels[0..3], [32, 64, 128]);
    assert_rgb_near(&image.pixels[3..6], [32, 64, 128]);
}

#[cfg(target_os = "linux")]
#[test]
fn fragment_writer_storage_is_visible_to_compute_reader() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("fragment-compute-storage-sharing").expect("create project");

    project
        .add_pass(
            "writer",
            PassKind::Buffer,
            r#"
layout(std430, binding = 1) buffer SharedData { vec4 value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    value = vec4(0.125, 0.25, 0.5, 1.0);
    fragColor = value;
}
"#,
        )
        .expect("add fragment writer")
        .set_pass_resolution("writer", 1, 1)
        .expect("set writer size")
        .bind_storage_buffer("writer", 1, "shared-data", 16)
        .expect("bind writer storage")
        .add_pass(
            "reader",
            PassKind::Compute,
            r#"
layout(std430, binding = 4) buffer SharedData { vec4 value; };
void mainCompute(ivec2 coord) { imageStore(iOutput, coord, value); }
"#,
        )
        .expect("add compute reader")
        .set_pass_resolution("reader", 1, 1)
        .expect("set reader size")
        .set_compute_local_size("reader", 1, 1, 1)
        .expect("set reader workgroup")
        .bind_storage_buffer("reader", 4, "shared-data", 16)
        .expect("bind reader storage")
        .add_input(
            "reader",
            0,
            InputKind::Pass,
            "writer",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("order writer before reader")
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
            "reader",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("wire compute reader");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(1, 1).expect("render");
    assert_rgb_near(&image.pixels[0..3], [32, 64, 128]);
}

#[cfg(target_os = "linux")]
#[test]
fn fragment_storage_persists_across_rendered_frames() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create OpenGL context");
    let mut project = Project::new("fragment-storage-persistence").expect("create project");

    project
        .add_pass(
            "image",
            PassKind::Image,
            r#"
layout(std430, binding = 0) buffer SharedCounter { uint value; };
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    uint nextValue = atomicAdd(value, 1u) + 1u;
    fragColor = vec4(float(nextValue) / 10.0, 0.0, 0.0, 1.0);
}
"#,
        )
        .expect("add image")
        .bind_storage_buffer("image", 0, "shared-counter", 4)
        .expect("bind storage");

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");

    let first = runtime.render(1, 1).expect("render first frame");
    assert_rgb_near(&first.pixels[..3], [26, 0, 0]);

    runtime.tick_fixed(1.0 / 60.0, 60.0).expect("advance frame");
    let second = runtime.render(1, 1).expect("render second frame");
    assert_rgb_near(&second.pixels[..3], [51, 0, 0]);
}
