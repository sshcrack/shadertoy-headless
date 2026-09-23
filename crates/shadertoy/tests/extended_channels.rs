use shadertoy::{Filter, HeadlessContext, InputKind, PassKind, Project, Runtime, Wrap};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
static EGL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(target_os = "linux")]
#[test]
fn channels_above_three_and_double_digit_channels_render() {
    let _egl_guard = EGL_TEST_LOCK.lock().expect("lock EGL test context");
    let context = HeadlessContext::new(16, 16).expect("create headless OpenGL context");
    let mut project = Project::new("extended-channels").expect("create project");

    project
        .add_texture_rgba8("red", 1, 1, &[64, 0, 0, 255])
        .expect("add red texture")
        .add_texture_rgba8("green", 1, 1, &[0, 128, 0, 255])
        .expect("add green texture")
        .add_texture_rgba8("blue", 1, 1, &[0, 0, 192, 255])
        .expect("add blue texture")
        .add_pass(
            "image",
            PassKind::Image,
            r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = vec2(0.5);
    float allBound = (
        texture(iChannel0, uv).a + texture(iChannel1, uv).a +
        texture(iChannel2, uv).a + texture(iChannel3, uv).a +
        texture(iChannel4, uv).a + texture(iChannel5, uv).a +
        texture(iChannel6, uv).a + texture(iChannel7, uv).a +
        texture(iChannel8, uv).a + texture(iChannel9, uv).a +
        texture(iChannel10, uv).a + texture(iChannel11, uv).a +
        texture(iChannel12, uv).a + texture(iChannel13, uv).a +
        texture(iChannel14, uv).a + texture(iChannel15, uv).a
    ) / 16.0;
    vec4 a = texture(iChannel4, uv);
    vec4 b = texture(iChannel10, uv);
    vec4 c = texture(iChannel15, uv);
    float resolutionOk = float(
        iChannelResolution[15].x > 0.5 &&
        iChannelResolution[15].x < 1.5 &&
        iChannelResolution[15].y > 0.5 &&
        iChannelResolution[15].y < 1.5
    );
    fragColor = vec4(a.r, b.g, c.b * resolutionOk, 1.0) * allBound;
}
"#,
        )
        .expect("add image");

    for channel in 0..16 {
        let source = match channel {
            10 => "green",
            15 => "blue",
            _ => "red",
        };
        project
            .add_input(
                "image",
                channel,
                InputKind::Texture,
                source,
                false,
                Filter::Nearest,
                Wrap::Clamp,
            )
            .unwrap_or_else(|error| panic!("bind channel {channel}: {error}"));
    }

    let mut runtime = Runtime::new(&context).expect("create runtime");
    runtime.load_project(&project).expect("load project");
    let image = runtime.render(1, 1).expect("render extended channels");

    assert_eq!(image.pixels.len(), 3);
    let pixel = &image.pixels[..3];
    assert!((i16::from(pixel[0]) - 64).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[1]) - 128).abs() <= 1, "{pixel:?}");
    assert!((i16::from(pixel[2]) - 192).abs() <= 1, "{pixel:?}");
}
