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
    for name in ["keyboard", "music"] {
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
