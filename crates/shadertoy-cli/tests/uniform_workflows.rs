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

fn write_uniform_project(root: &Path) {
    std::fs::create_dir_all(root.join("shaders")).expect("create shaders");
    std::fs::write(
        root.join("ShaderToy.toml"),
        r#"format = 1

[project]
name = "uniform-workflows"

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

[[pass]]
name = "buffer-a"
kind = "buffer"
source = "shaders/buffer.frag"

[[pass]]
name = "image"
kind = "image"
source = "shaders/image.frag"

[[pass.input]]
channel = 0
source = "buffer-a"
kind = "pass"
frame = "current"
filter = "nearest"
wrap = "clamp"
"#,
    )
    .expect("write manifest");
    std::fs::write(
        root.join("shaders/buffer.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(gain, 0.0, 0.0, 1.0); }\n",
    )
    .expect("write buffer shader");
    std::fs::write(
        root.join("shaders/image.frag"),
        "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, p / iResolution.xy); }\n",
    )
    .expect("write image shader");
}

#[cfg(target_os = "linux")]
#[test]
fn state_capture_and_runtime_inspection_accept_uniform_overrides() {
    let temp = TempRoot::new("state-set");
    let project = temp.path().join("project");
    write_uniform_project(&project);
    let project_arg = project.to_string_lossy().into_owned();

    let inspected = shadertoy(&[
        "--json",
        "inspect",
        "--project",
        &project_arg,
        "buffer",
        "buffer-a",
        "--frame",
        "0",
        "--pixel",
        "0,0",
        "--set",
        "gain=0.75",
    ]);
    assert!(inspected.status.success(), "{inspected:?}");
    let inspected_json: serde_json::Value =
        serde_json::from_slice(&inspected.stdout).expect("parse inspect JSON");
    let inspected_red = inspected_json["pixel"]["rgba"][0]
        .as_f64()
        .expect("red channel");
    assert!((inspected_red - 0.75).abs() < 0.01, "{inspected_red}");

    let state = temp.path().join("storm.ststate");
    let state_arg = state.to_string_lossy().into_owned();
    let captured = shadertoy(&[
        "state",
        "capture",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set",
        "gain=0.80",
        "-o",
        &state_arg,
    ]);
    assert!(captured.status.success(), "{captured:?}");

    let restored = temp.path().join("restored.png");
    let restored_arg = restored.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--state",
        &state_arg,
        "--frame",
        "0",
        "--pass",
        "buffer-a",
        "-o",
        &restored_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let red = image::open(restored)
        .expect("open restored buffer")
        .to_rgb8()
        .get_pixel(0, 0)[0];
    assert!((198..=210).contains(&red), "{red}");
}

#[cfg(target_os = "linux")]
#[test]
fn sweep_renders_variants_and_default_contact_sheet() {
    let temp = TempRoot::new("sweep");
    let project = temp.path().join("project");
    write_uniform_project(&project);
    let project_arg = project.to_string_lossy().into_owned();
    let output_dir = temp.path().join("variants");
    let output_dir_arg = output_dir.to_string_lossy().into_owned();

    let swept = shadertoy(&[
        "--json",
        "sweep",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--set",
        "gain=0.20,0.80",
        "--output-dir",
        &output_dir_arg,
    ]);
    assert!(swept.status.success(), "{swept:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&swept.stdout).expect("parse sweep JSON");
    assert_eq!(report["variant_count"], 2);

    let low = image::open(output_dir.join("variant-000.png"))
        .expect("open low variant")
        .to_rgb8()
        .get_pixel(0, 0)[0];
    let high = image::open(output_dir.join("variant-001.png"))
        .expect("open high variant")
        .to_rgb8()
        .get_pixel(0, 0)[0];
    assert!((45..=60).contains(&low), "{low}");
    assert!((198..=210).contains(&high), "{high}");

    let sheet = image::open(output_dir.join("contact-sheet.png")).expect("open contact sheet");
    assert_eq!((sheet.width(), sheet.height()), (2, 1));
}

#[cfg(target_os = "linux")]
#[test]
fn blind_sweep_requires_judgment_before_reveal_and_reports_mapping_afterward() {
    let temp = TempRoot::new("blind-sweep");
    let project = temp.path().join("project");
    write_uniform_project(&project);
    let project_arg = project.to_string_lossy().into_owned();
    let output_dir = temp.path().join("blind");
    let output_dir_arg = output_dir.to_string_lossy().into_owned();

    let swept = shadertoy(&[
        "--json",
        "sweep",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--blind",
        "--set",
        "gain=0.20,0.80",
        "--output-dir",
        &output_dir_arg,
    ]);
    assert!(swept.status.success(), "{swept:?}");
    let sweep_report: serde_json::Value =
        serde_json::from_slice(&swept.stdout).expect("parse blind sweep JSON");
    assert_eq!(sweep_report["blind"], true);
    assert_eq!(sweep_report["variant_count"], 2);
    assert!(sweep_report["variants"][0].get("set").is_none());
    assert!(sweep_report["variants"][0]["label"].as_str().is_some());

    let session = output_dir.join("blind-session.json");
    let session_arg = session.to_string_lossy().into_owned();
    let public_session = std::fs::read_to_string(&session).expect("read blind session");
    assert!(!public_session.contains("gain=0.20"));
    assert!(!public_session.contains("gain=0.80"));
    assert!(output_dir.join("blind-contact-sheet.png").exists());

    let early_reveal = shadertoy(&["blind", "reveal", &session_arg]);
    assert!(!early_reveal.status.success());
    assert!(String::from_utf8_lossy(&early_reveal.stderr).contains("no judgment yet"));

    let label = sweep_report["variants"][0]["label"]
        .as_str()
        .expect("blind label")
        .to_string();
    let judged = shadertoy(&[
        "--json",
        "blind",
        "judge",
        &session_arg,
        "--pick",
        &label,
        "--reason",
        "Best silhouette and breakup in the blinded comparison.",
    ]);
    assert!(judged.status.success(), "{judged:?}");
    let judgment: serde_json::Value =
        serde_json::from_slice(&judged.stdout).expect("parse judgment JSON");
    assert_eq!(judgment["selected"], label);
    assert_eq!(judgment["revealed"], false);

    let revealed = shadertoy(&["--json", "blind", "reveal", &session_arg]);
    assert!(revealed.status.success(), "{revealed:?}");
    let reveal: serde_json::Value =
        serde_json::from_slice(&revealed.stdout).expect("parse reveal JSON");
    assert_eq!(reveal["result"]["judgment"]["selected"], label);
    assert_eq!(reveal["result"]["mapping"].as_array().unwrap().len(), 2);
    assert!(
        reveal["result"]["selected_variant"]["set"][0]
            .as_str()
            .unwrap()
            .starts_with("gain=")
    );
    assert!(output_dir.join("blind-reveal.json").exists());

    // Reusing the output directory starts a fresh blind lifecycle instead of
    // accidentally inheriting a previous judgment/reveal.
    let swept_again = shadertoy(&[
        "sweep",
        "--project",
        &project_arg,
        "--frame",
        "0",
        "--blind",
        "--set",
        "gain=0.20,0.80",
        "--output-dir",
        &output_dir_arg,
    ]);
    assert!(swept_again.status.success(), "{swept_again:?}");
    assert!(!output_dir.join("blind-judgment.json").exists());
    assert!(!output_dir.join("blind-reveal.json").exists());

    let output_dir_session_arg = output_dir.to_string_lossy().into_owned();
    let early_reveal_again = shadertoy(&["blind", "reveal", &output_dir_session_arg]);
    assert!(!early_reveal_again.status.success());
    assert!(String::from_utf8_lossy(&early_reveal_again.stderr).contains("no judgment yet"));
}

fn write_solid_png(path: &Path, rgb: [u8; 3]) {
    let pixels = [rgb, rgb, rgb, rgb]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    image::save_buffer(path, &pixels, 2, 2, image::ColorType::Rgb8).expect("write test PNG");
}

#[test]
fn blind_create_anonymizes_existing_image_sets_until_reveal() {
    let temp = TempRoot::new("blind-existing-images");
    let old = temp.path().join("old");
    let new = temp.path().join("new");
    std::fs::create_dir_all(&old).expect("create old image set");
    std::fs::create_dir_all(&new).expect("create new image set");
    write_solid_png(&old.join("frame-000.png"), [255, 0, 0]);
    write_solid_png(&old.join("frame-001.png"), [192, 0, 0]);
    write_solid_png(&new.join("frame-000.png"), [0, 255, 0]);
    write_solid_png(&new.join("frame-001.png"), [0, 192, 0]);

    let output_dir = temp.path().join("blind");
    std::fs::create_dir_all(output_dir.join("variants/C")).expect("create stale blind output");
    write_solid_png(&output_dir.join("variants/C/stale.png"), [0, 0, 255]);
    let old_arg = old.to_string_lossy().into_owned();
    let new_arg = new.to_string_lossy().into_owned();
    let output_arg = output_dir.to_string_lossy().into_owned();
    let created = shadertoy(&[
        "--json",
        "blind",
        "create",
        &old_arg,
        &new_arg,
        "--output-dir",
        &output_arg,
    ]);
    assert!(created.status.success(), "{created:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&created.stdout).expect("parse blind-create JSON");
    assert_eq!(report["variant_count"], 2);
    assert_eq!(report["images_per_variant"], 2);
    assert_eq!(report["width"], 2);
    assert_eq!(report["height"], 2);
    assert!(output_dir.join("variants/A/image-000.png").exists());
    assert!(output_dir.join("variants/A/image-001.png").exists());
    assert!(output_dir.join("variants/B/image-000.png").exists());
    assert!(!output_dir.join("variants/C").exists());
    assert!(output_dir.join("blind-contact-sheet.png").exists());

    let public_json = serde_json::to_string(&report).expect("serialize public report");
    let public_session =
        std::fs::read_to_string(output_dir.join("blind-session.json")).expect("read session");
    let sealed_mapping =
        std::fs::read(output_dir.join(".blind-mapping.bin")).expect("read sealed mapping");
    for source in [&old_arg, &new_arg] {
        assert!(
            !public_json.contains(source),
            "public report leaked {source}"
        );
        assert!(
            !public_session.contains(source),
            "public session leaked {source}"
        );
        assert!(
            !sealed_mapping
                .windows(source.len())
                .any(|window| window == source.as_bytes()),
            "sealed mapping leaked plaintext source {source}"
        );
    }

    let session = output_dir.join("blind-session.json");
    let session_arg = session.to_string_lossy().into_owned();
    let label = report["variants"][0]["label"]
        .as_str()
        .expect("blind label")
        .to_string();
    let judged = shadertoy(&[
        "blind",
        "judge",
        &session_arg,
        "--pick",
        &label,
        "--reason",
        "Preferred the blinded result.",
    ]);
    assert!(judged.status.success(), "{judged:?}");

    let revealed = shadertoy(&["--json", "blind", "reveal", &session_arg]);
    assert!(revealed.status.success(), "{revealed:?}");
    let reveal: serde_json::Value =
        serde_json::from_slice(&revealed.stdout).expect("parse reveal JSON");
    let mapping = reveal["result"]["mapping"]
        .as_array()
        .expect("mapping array");
    assert_eq!(mapping.len(), 2);
    let identities = mapping
        .iter()
        .map(|entry| entry["set"][0].as_str().expect("source identity"))
        .collect::<Vec<_>>();
    assert!(
        identities
            .iter()
            .any(|value| value == &format!("source={old_arg}"))
    );
    assert!(
        identities
            .iter()
            .any(|value| value == &format!("source={new_arg}"))
    );
}

#[cfg(target_os = "linux")]
#[test]
fn blind_create_renders_projects_and_sttf_builds() {
    let temp = TempRoot::new("blind-render-sources");
    let project_a = temp.path().join("project-a");
    let project_b = temp.path().join("project-b");
    write_uniform_project(&project_a);
    write_uniform_project(&project_b);

    let project_a_arg = project_a.to_string_lossy().into_owned();
    let project_b_arg = project_b.to_string_lossy().into_owned();
    let project_output = temp.path().join("project-blind");
    let project_output_arg = project_output.to_string_lossy().into_owned();
    let projects = shadertoy(&[
        "--json",
        "blind",
        "create",
        &project_a_arg,
        &project_b_arg,
        "--frames",
        "0,1",
        "--output-dir",
        &project_output_arg,
    ]);
    assert!(projects.status.success(), "{projects:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&projects.stdout).expect("parse project blind JSON");
    assert_eq!(report["images_per_variant"], 2);
    assert!(project_output.join("blind-contact-sheet.png").exists());

    let build_a = temp.path().join("a.sttf");
    let build_b = temp.path().join("b.sttf");
    let build_a_arg = build_a.to_string_lossy().into_owned();
    let build_b_arg = build_b.to_string_lossy().into_owned();
    assert!(
        shadertoy(&["build", "--project", &project_a_arg, "-o", &build_a_arg])
            .status
            .success()
    );
    assert!(
        shadertoy(&["build", "--project", &project_b_arg, "-o", &build_b_arg])
            .status
            .success()
    );

    let build_output = temp.path().join("build-blind");
    let build_output_arg = build_output.to_string_lossy().into_owned();
    let builds = shadertoy(&[
        "--json",
        "blind",
        "create",
        &build_a_arg,
        &build_b_arg,
        "--frames",
        "0,1",
        "--width",
        "1",
        "--height",
        "1",
        "--output-dir",
        &build_output_arg,
    ]);
    assert!(builds.status.success(), "{builds:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&builds.stdout).expect("parse STTF blind JSON");
    assert_eq!(report["images_per_variant"], 2);
    assert_eq!(report["width"], 1);
    assert_eq!(report["height"], 1);
}

#[cfg(target_os = "linux")]
#[test]
fn blind_create_sttf_preserves_distinct_uniform_defaults() {
    let temp = TempRoot::new("blind-sttf-uniform-defaults");
    let open = temp.path().join("open");
    let storm = temp.path().join("storm");
    write_uniform_project(&open);
    write_uniform_project(&storm);

    let storm_manifest = std::fs::read_to_string(storm.join("ShaderToy.toml"))
        .expect("read storm manifest")
        .replace("default = 0.25", "default = 0.75");
    std::fs::write(storm.join("ShaderToy.toml"), storm_manifest).expect("write storm manifest");

    let open_arg = open.to_string_lossy().into_owned();
    let storm_arg = storm.to_string_lossy().into_owned();
    let open_sttf = temp.path().join("open.sttf");
    let storm_sttf = temp.path().join("storm.sttf");
    let open_sttf_arg = open_sttf.to_string_lossy().into_owned();
    let storm_sttf_arg = storm_sttf.to_string_lossy().into_owned();

    assert!(
        shadertoy(&["build", "--project", &open_arg, "-o", &open_sttf_arg])
            .status
            .success()
    );
    assert!(
        shadertoy(&["build", "--project", &storm_arg, "-o", &storm_sttf_arg])
            .status
            .success()
    );

    let output_dir = temp.path().join("blind");
    let output_arg = output_dir.to_string_lossy().into_owned();
    let created = shadertoy(&[
        "--json",
        "blind",
        "create",
        &open_sttf_arg,
        &storm_sttf_arg,
        "--frames",
        "0",
        "--width",
        "1",
        "--height",
        "1",
        "--output-dir",
        &output_arg,
    ]);
    assert!(created.status.success(), "{created:?}");

    let mut reds = ["A", "B"]
        .map(|label| {
            image::open(output_dir.join(format!("variants/{label}/image-000.png")))
                .expect("open blinded STTF render")
                .to_rgb8()
                .get_pixel(0, 0)[0]
        })
        .to_vec();
    reds.sort_unstable();
    assert!(
        (60..=70).contains(&reds[0]) && (185..=195).contains(&reds[1]),
        "STTF uniform defaults were not restored: {reds:?}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn blind_create_bare_git_refs_follow_current_nested_project_before_images() {
    let temp = TempRoot::new("blind-git-nested-project");
    let repo = temp.path().join("repo");
    let project = repo.join("fable");
    write_uniform_project(&project);
    std::fs::create_dir_all(project.join("assets")).expect("create project assets");
    write_solid_png(&project.join("assets/foam-a.png"), [255, 255, 255]);
    write_solid_png(&project.join("assets/foam-b.png"), [192, 192, 192]);
    write_solid_png(&project.join("assets/foam-c.png"), [128, 128, 128]);

    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(args)
            .output()
            .expect("run git");
        assert!(output.status.success(), "{output:?}");
        output
    };
    git(&["init", "--quiet"]);
    git(&["config", "user.email", "tests@example.invalid"]);
    git(&["config", "user.name", "ShaderToy Tests"]);
    git(&["add", "."]);
    git(&["commit", "--quiet", "-m", "old"]);
    let old_ref = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
        .expect("old ref UTF-8")
        .trim()
        .to_string();

    std::fs::write(project.join("revision.txt"), "new").expect("write second revision marker");
    git(&["add", "fable/revision.txt"]);
    git(&["commit", "--quiet", "-m", "new"]);
    let new_ref = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
        .expect("new ref UTF-8")
        .trim()
        .to_string();

    let old_source = format!("git:{old_ref}");
    let new_source = format!("git:{new_ref}");
    let repo_arg = repo.to_string_lossy().into_owned();
    let output_dir = temp.path().join("blind");
    let output_arg = output_dir.to_string_lossy().into_owned();
    let created = Command::new(env!("CARGO_BIN_EXE_shadertoy"))
        .current_dir(&project)
        .args([
            "--json",
            "blind",
            "create",
            &old_source,
            &new_source,
            "--git-root",
            &repo_arg,
            "--frames",
            "0",
            "--width",
            "3",
            "--height",
            "2",
            "--output-dir",
            &output_arg,
        ])
        .output()
        .expect("run nested git blind comparison");
    assert!(created.status.success(), "{created:?}");

    let report: serde_json::Value =
        serde_json::from_slice(&created.stdout).expect("parse nested git blind JSON");
    assert_eq!(report["images_per_variant"], 1);
    assert_eq!(report["width"], 3);
    assert_eq!(report["height"], 2);
}

#[test]
fn blind_create_materializes_git_revisions_without_leaking_refs() {
    let temp = TempRoot::new("blind-git-refs");
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join("renders")).expect("create git fixture");
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(args)
            .output()
            .expect("run git");
        assert!(output.status.success(), "{output:?}");
        output
    };
    git(&["init", "--quiet"]);
    git(&["config", "user.email", "tests@example.invalid"]);
    git(&["config", "user.name", "ShaderToy Tests"]);

    write_solid_png(&repo.join("renders/frame.png"), [255, 0, 0]);
    git(&["add", "renders/frame.png"]);
    git(&["commit", "--quiet", "-m", "old"]);
    let old_ref = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
        .expect("old ref UTF-8")
        .trim()
        .to_string();

    write_solid_png(&repo.join("renders/frame.png"), [0, 255, 0]);
    git(&["add", "renders/frame.png"]);
    git(&["commit", "--quiet", "-m", "new"]);
    let new_ref = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
        .expect("new ref UTF-8")
        .trim()
        .to_string();

    let old_source = format!("git:{old_ref}::renders");
    let new_source = format!("git:{new_ref}::renders");
    let repo_arg = repo.to_string_lossy().into_owned();
    let output_dir = temp.path().join("blind");
    let output_arg = output_dir.to_string_lossy().into_owned();
    let created = shadertoy(&[
        "--json",
        "blind",
        "create",
        &old_source,
        &new_source,
        "--git-root",
        &repo_arg,
        "--output-dir",
        &output_arg,
    ]);
    assert!(created.status.success(), "{created:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&created.stdout).expect("parse git blind JSON");
    assert_eq!(report["variant_count"], 2);
    assert_eq!(report["images_per_variant"], 1);
    let public = serde_json::to_string(&report).expect("serialize public report");
    assert!(!public.contains(&old_ref));
    assert!(!public.contains(&new_ref));
    assert!(output_dir.join("variants/A/image-000.png").exists());
    assert!(output_dir.join("variants/B/image-000.png").exists());
}

#[cfg(target_os = "linux")]
#[test]
fn quality_presets_drive_render_build_profile_and_blind_sources() {
    let temp = TempRoot::new("quality-presets");
    let project = temp.path().join("project");
    write_uniform_project(&project);
    let manifest_path = project.join("ShaderToy.toml");
    let mut manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
    manifest.push_str(
        r#"
[preset.high]
render_scale = 4.0

[preset.high.pass.buffer-a]
width = 4
height = 4

[preset.low]
render_scale = 2.0

[preset.low.pass.buffer-a]
width = 2
height = 2
"#,
    );
    std::fs::write(&manifest_path, manifest).expect("write preset manifest");

    let project_arg = project.to_string_lossy().into_owned();

    let checked = shadertoy(&["check", "--project", &project_arg, "--preset", "low"]);
    assert!(checked.status.success(), "{checked:?}");

    let low_png = temp.path().join("low.png");
    let low_png_arg = low_png.to_string_lossy().into_owned();
    let rendered = shadertoy(&[
        "render",
        "--project",
        &project_arg,
        "--preset",
        "low",
        "--frame",
        "0",
        "-o",
        &low_png_arg,
    ]);
    assert!(rendered.status.success(), "{rendered:?}");
    let low = image::open(&low_png).expect("open low preset render");
    assert_eq!((low.width(), low.height()), (2, 2));

    let artifact = temp.path().join("low.sttf");
    let artifact_arg = artifact.to_string_lossy().into_owned();
    let built = shadertoy(&[
        "build",
        "--project",
        &project_arg,
        "--preset",
        "low",
        "-o",
        &artifact_arg,
    ]);
    assert!(built.status.success(), "{built:?}");
    assert!(artifact.is_file());

    let profile = shadertoy(&[
        "--json",
        "profile",
        "--project",
        &project_arg,
        "--preset",
        "low",
        "--frame",
        "0",
        "--warmup",
        "0",
        "--samples",
        "2",
    ]);
    assert!(profile.status.success(), "{profile:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&profile.stdout).expect("parse profile report");
    assert_eq!(report["width"], 2);
    assert_eq!(report["height"], 2);
    assert_eq!(report["timing_mode"], "completion_synchronized");
    assert_eq!(report["gpu_frame_mode"], "attributed_pass_sum");
    assert!(
        report["gpu_frame"]["median_ms"]
            .as_f64()
            .is_some_and(|value| value > 0.0)
    );
    assert!(report["passes"][0]["p95_ms"].as_f64().is_some());

    let high_source = format!("project:{project_arg}@preset=high");
    let low_source = format!("project:{project_arg}@preset=low");
    let blind_dir = temp.path().join("blind");
    let blind_dir_arg = blind_dir.to_string_lossy().into_owned();
    let blinded = shadertoy(&[
        "--json",
        "blind",
        "create",
        &high_source,
        &low_source,
        "--frames",
        "0",
        "--output-dir",
        &blind_dir_arg,
    ]);
    assert!(blinded.status.success(), "{blinded:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&blinded.stdout).expect("parse blind report");
    assert_eq!(report["width"], 4);
    assert_eq!(report["height"], 4);
    assert_eq!(report["images_per_variant"], 1);

    let sync_profile = shadertoy(&[
        "--json",
        "profile",
        "--project",
        &project_arg,
        "--preset",
        "low",
        "--frame",
        "0",
        "--warmup",
        "0",
        "--samples",
        "1",
        "--sync-per-pass",
    ]);
    assert!(sync_profile.status.success(), "{sync_profile:?}");
    let sync_report: serde_json::Value =
        serde_json::from_slice(&sync_profile.stdout).expect("parse sync profile report");
    assert_eq!(sync_report["timing_mode"], "sync_per_pass");
    assert_eq!(sync_report["sync_per_pass"], true);
}
