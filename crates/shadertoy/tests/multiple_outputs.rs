use shadertoy::{
    Filter, HeadlessContext, InputKind, PassKind, Project, RenderFormat, Runtime, Wrap,
};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
static EGL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(target_os = "linux")]
#[test]
fn fragment_mrt_output_can_feed_downstream_pass() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create context");
    let mut project = Project::new("fragment-mrt").expect("project");

    project
        .add_pass(
            "gbuffer",
            PassKind::Buffer,
            r#"
layout(location = 1) out vec4 normalTarget;

void mainImage(out vec4 color, in vec2 fragCoord) {
    color = vec4(0.1, 0.2, 0.3, 1.0);
    normalTarget = vec4(0.75, 0.25, 0.5, 1.0);
}
"#,
        )
        .expect("add gbuffer")
        .set_pass_format("gbuffer", RenderFormat::Rgba16f)
        .expect("set primary format")
        .add_pass_output("gbuffer", RenderFormat::Rg32f)
        .expect("add second target")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 color, in vec2 fragCoord) {
    vec2 normalData = texture(iChannel0, fragCoord / iResolution.xy).rg;
    color = vec4(normalData, 0.0, 1.0);
}
"#,
        )
        .expect("add image")
        .add_input_output(
            "image",
            0,
            InputKind::Pass,
            "gbuffer",
            1,
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("wire output 1");

    let mut runtime = Runtime::new(&context).expect("runtime");
    runtime.load_project(&project).expect("load");
    let image = runtime.render(3, 2).expect("render");
    let pixel = &image.pixels[..3];
    assert!((i16::from(pixel[0]) - 191).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 64).abs() <= 1, "{pixel:?}");
    assert!(pixel[2] <= 1, "{pixel:?}");
}

#[cfg(target_os = "linux")]
#[test]
fn compute_multiple_outputs_can_feed_independent_channels() {
    let _guard = EGL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = HeadlessContext::new(64, 64).expect("create context");
    let mut project = Project::new("compute-multi-output").expect("project");

    project
        .add_pass(
            "simulation",
            PassKind::Compute,
            r#"
void mainCompute(ivec2 coord) {
    imageStore(iOutput, coord, vec4(0.2, 0.0, 0.0, 1.0));
    imageStore(iOutput1, coord, vec4(0.8, 0.4, 0.0, 1.0));
}
"#,
        )
        .expect("add compute")
        .set_pass_resolution("simulation", 2, 2)
        .expect("size")
        .set_pass_format("simulation", RenderFormat::R32f)
        .expect("primary format")
        .add_pass_output("simulation", RenderFormat::Rg32f)
        .expect("second output")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 color, in vec2 fragCoord) {
    float primary = texture(iChannel0, vec2(0.5)).r;
    vec2 secondary = texture(iChannel1, vec2(0.5)).rg;
    color = vec4(primary, secondary, 1.0);
}
"#,
        )
        .expect("image")
        .add_input(
            "image",
            0,
            InputKind::Pass,
            "simulation",
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("primary input")
        .add_input_output(
            "image",
            1,
            InputKind::Pass,
            "simulation",
            1,
            false,
            Filter::Nearest,
            Wrap::Clamp,
        )
        .expect("secondary input");

    let mut runtime = Runtime::new(&context).expect("runtime");
    runtime.load_project(&project).expect("load");
    let image = runtime.render(2, 2).expect("render");
    let pixel = &image.pixels[..3];
    assert!((i16::from(pixel[0]) - 51).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 204).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[2]) - 102).abs() <= 1, "{pixel:?}");
}
