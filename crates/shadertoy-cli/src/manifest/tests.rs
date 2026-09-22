use super::*;

#[test]
fn checked_in_schema_matches_manifest_types() {
    let expected = crate::include_file!("schema/shadertoy.schema.json").trim_end();
    let generated = schema_json().expect("schema generation should succeed");
    assert_eq!(expected, generated);
}

#[test]
fn multipass_template_validates() {
    Manifest::multipass("feedback")
        .validate_structure()
        .expect("multipass template should remain valid");
}

#[test]
fn manifest_rejects_paths_outside_project() {
    for source in ["../outside.frag", "/tmp/outside.frag", r"C:\outside.frag"] {
        let mut manifest = Manifest::minimal("demo");
        manifest.passes[0].source = source.into();
        assert!(
            manifest.validate_structure().is_err(),
            "source should be rejected: {source}"
        );
    }
}

#[test]
fn manifest_rejects_reserved_source_names() {
    for name in ["keyboard", "music", "webcam"] {
        let mut manifest = Manifest::minimal("demo");
        manifest.passes[0].name = name.into();
        assert!(
            manifest.validate_structure().is_err(),
            "pass name should be rejected: {name}"
        );
    }
}

#[test]
fn built_in_input_kind_requires_canonical_source_name() {
    let mut manifest = Manifest::minimal("demo");
    manifest.passes[0].inputs.push(Input {
        channel: 0,
        source: "keys".into(),
        kind: Some(InputKind::Keyboard),
        output: 0,
        frame: FrameRef::Current,
        filter: Filter::Linear,
        wrap: Wrap::Clamp,
    });
    assert!(manifest.validate_structure().is_err());
}

#[test]
fn fixed_dimensions_require_complete_buffer_pair() {
    let mut manifest = Manifest::multipass("fixed");
    manifest.passes[0].width = Some(256);
    assert!(manifest.validate_structure().is_err());

    manifest.passes[0].height = Some(256);
    manifest
        .validate_structure()
        .expect("buffer pass should accept fixed dimensions");

    manifest.passes[1].width = Some(256);
    manifest.passes[1].height = Some(256);
    assert!(manifest.validate_structure().is_err());
}

#[test]
fn multiple_outputs_validate_only_for_offscreen_2d_passes() {
    let mut manifest = Manifest::multipass("mrt");
    manifest.passes[0].extra_outputs = vec![RenderFormat::Rg32f, RenderFormat::Rgba16f];
    manifest
        .validate_structure()
        .expect("buffer MRT should validate");

    manifest.passes[1].extra_outputs = vec![RenderFormat::Rg32f];
    assert!(
        manifest.validate_structure().is_err(),
        "final image may not expose extra outputs"
    );
}

#[test]
fn input_output_index_must_exist_and_previous_feedback_uses_primary_output() {
    let mut manifest = Manifest::multipass("mrt-input");
    manifest.passes[0].extra_outputs = vec![RenderFormat::Rg32f];
    manifest.passes[1].inputs[0].output = 1;
    manifest
        .validate_structure()
        .expect("current-frame output 1 should validate");

    manifest.passes[1].inputs[0].output = 2;
    assert!(manifest.validate_structure().is_err());

    manifest.passes[1].inputs[0].output = 1;
    manifest.passes[1].inputs[0].frame = FrameRef::Previous;
    assert!(
        manifest.validate_structure().is_err(),
        "non-primary MRT feedback is intentionally not resumable"
    );
}

#[test]
fn compute_local_size_rejects_multiple_z_lanes() {
    let mut manifest = Manifest::minimal("compute-z");
    manifest.passes.insert(
        0,
        Pass {
            name: "simulation".into(),
            kind: PassKind::Compute,
            source: "shaders/simulation.comp".into(),
            width: Some(1),
            height: Some(1),
            format: RenderFormat::Rgba32f,
            extra_outputs: Vec::new(),
            iterations: 1,
            local_size: Some([1, 1, 2]),
            storage: Vec::new(),
            inputs: Vec::new(),
        },
    );

    let error = manifest
        .validate_structure()
        .expect_err("2D compute entrypoints must reject multiple local Z lanes");
    assert!(error.to_string().contains("z"), "{error}");
    assert!(error.to_string().contains("1"), "{error}");
}

#[test]
fn visual_reference_tests_reject_frame_resolution_matrices() {
    let source = r#"format = 1
[project]
name = "matrix-reference"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[test]]
name = "ambiguous-reference"
frames = [0, 1]
resolutions = [[16, 16], [32, 32]]
reference = "tests/reference.png"
"#;
    let manifest: Manifest = toml::from_str(source).expect("manifest should parse");
    let error = manifest
        .validate_structure()
        .expect_err("one reference path cannot describe a frame/resolution matrix");
    assert!(
        error.to_string().contains("single reference image"),
        "{error}"
    );
}

#[test]
fn named_preset_scales_output_and_overrides_pass_settings() {
    let source = r#"format = 1
[project]
name = "preset-demo"

[render]
width = 1600
height = 900

[[pass]]
name = "waves"
kind = "compute"
source = "shaders/waves.comp"
width = 1024
height = 1024
iterations = 4

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[preset.low]
render_scale = 0.75

[preset.low.pass.waves]
width = 640
height = 360
iterations = 2
local_size = [16, 8, 1]
"#;

    let mut manifest: Manifest = toml::from_str(source).expect("preset manifest should parse");
    manifest
        .validate_structure()
        .expect("preset manifest should validate");
    manifest.apply_preset("low").expect("preset should apply");

    assert_eq!((manifest.render.width, manifest.render.height), (1200, 675));
    let waves = manifest
        .passes
        .iter()
        .find(|pass| pass.name == "waves")
        .expect("waves pass");
    assert_eq!((waves.width, waves.height), (Some(640), Some(360)));
    assert_eq!(waves.iterations, 2);
    assert_eq!(waves.local_size, Some([16, 8, 1]));
}

#[test]
fn named_preset_rejects_unknown_pass_and_unknown_selection() {
    let mut manifest = Manifest::minimal("preset-errors");
    manifest.presets.insert(
        "bad".into(),
        Preset {
            render_scale: Some(0.5),
            passes: BTreeMap::from([(
                "missing".into(),
                PresetPass {
                    width: Some(320),
                    height: Some(180),
                    ..PresetPass::default()
                },
            )]),
        },
    );
    assert!(manifest.validate_structure().is_err());

    let manifest = Manifest::minimal("preset-errors");
    let mut manifest = manifest;
    let error = manifest
        .apply_preset("missing")
        .expect_err("unknown preset should fail");
    assert!(error.to_string().contains("unknown preset"), "{error}");
}
