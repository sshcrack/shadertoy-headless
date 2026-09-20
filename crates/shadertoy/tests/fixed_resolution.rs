use shadertoy::{Filter, HeadlessContext, InputKind, PassKind, Project, Runtime, Wrap};

#[cfg(target_os = "linux")]
#[test]
fn fixed_buffer_keeps_resolution_and_feedback_across_output_resize() {
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
