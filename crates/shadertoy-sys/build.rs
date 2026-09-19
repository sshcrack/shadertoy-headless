use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `std::fs::canonicalize` yields `\\?\`-prefixed extended-length paths on
/// Windows. CMake forwards them verbatim into compile commands, which MSVC's
/// `cl.exe` cannot open (fatal error C1083). Strip the prefix; the path stays
/// absolute and equivalent.
#[cfg(windows)]
fn without_verbatim_prefix(path: PathBuf) -> PathBuf {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    // `\\?\UNC\server\share\...` -> `\\server\share\...`
    const UNC: [u16; 8] = [0x5C, 0x5C, 0x3F, 0x5C, 0x55, 0x4E, 0x43, 0x5C];
    if wide.starts_with(&UNC) {
        let mut fixed = vec![0x5Cu16, 0x5C];
        fixed.extend_from_slice(&wide[UNC.len()..]);
        return PathBuf::from(OsString::from_wide(&fixed));
    }
    // `\\?\D:\...` -> `D:\...`
    const VERBATIM: [u16; 4] = [0x5C, 0x5C, 0x3F, 0x5C];
    if wide.starts_with(&VERBATIM) {
        return PathBuf::from(OsString::from_wide(&wide[VERBATIM.len()..]));
    }
    path
}

#[cfg(not(windows))]
fn without_verbatim_prefix(path: PathBuf) -> PathBuf {
    path
}

/// Stable 64-bit FNV-1a hash rendered as hex. Used to key the shared native
/// build directory by source root (std SipHash is randomly seeded per process
/// and would defeat directory reuse across cargo invocations).
fn fnv1a_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn main() {
    println!("cargo:rerun-if-changed=shadertoy-c/include/shadertoy/shadertoy.h");
    println!("cargo:rerun-if-changed=shadertoy-c/src/shadertoy.cpp");
    println!("cargo:rerun-if-changed=shadertoy-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=shadertoy");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=vcpkg.json");

    let repo_root = without_verbatim_prefix(
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .expect("failed to locate packaged ShaderToy source root"),
    );
    let header = repo_root.join("shadertoy-c/include/shadertoy/shadertoy.h");

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("st_.*")
        .allowlist_type("st_.*")
        .allowlist_var("ST_.*")
        .derive_default(true)
        .generate_comments(true)
        .generate()
        .expect("failed to generate ShaderToy C bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write generated bindings");

    // Cargo hands each command (build/clippy/test, dev/test profiles) its own
    // OUT_DIR, but every invocation configures the exact same native build
    // (CMAKE_BUILD_TYPE=Release, same defines). Share one native build tree
    // per target dir + source root so repeat invocations are a no-op ninja run
    // instead of a full CMake+vcpkg rebuild. OUT_DIR has the form
    // <target>/<profile>/build/<pkg>-<hash>/out, so three ancestors up is the
    // profile dir (debug/ or release/), shared by all in-profile invocations.
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .to_path_buf();
    let source_key = fnv1a_hex(&without_verbatim_prefix(repo_root.clone()).to_string_lossy());
    let shared_root =
        without_verbatim_prefix(profile_dir.join(format!("shadertoy-sys-native-{source_key}")));

    let mut config = cmake::Config::new(&repo_root);
    config.out_dir(shared_root.join("build"));
    if Command::new("ninja")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
    {
        config.generator("Ninja");
    }
    config
        .define("SHADERTOY_BUILD_GUI", "OFF")
        .define("SHADERTOY_BUILD_PREVIEW_TOOL", "OFF")
        .define("SHADERTOY_BUILD_C_API", "OFF")
        .define("SHADERTOY_BUILD_C_API_STATIC", "ON")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    if let Ok(vcpkg_root) = env::var("VCPKG_ROOT").or_else(|_| env::var("VCPKG_INSTALLATION_ROOT"))
    {
        let toolchain = Path::new(&vcpkg_root)
            .join("scripts")
            .join("buildsystems")
            .join("vcpkg.cmake");
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain);

        let vcpkg_work_root = shared_root.join("vcpkg");
        let install_options = format!(
            "--x-buildtrees-root={};--x-packages-root={};--downloads-root={}",
            vcpkg_work_root.join("buildtrees").display(),
            vcpkg_work_root.join("packages").display(),
            vcpkg_work_root.join("downloads").display()
        );
        config.define("VCPKG_INSTALL_OPTIONS", install_options);
    }

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    config.build_target("shadertoy-c-static");
    let destination = config.build();
    emit_static_link_manifest(&destination.join("build"));
}

fn emit_static_link_manifest(build_dir: &Path) {
    let manifest = fs::read_dir(build_dir)
        .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", build_dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("shadertoy-rust-link-") && name.ends_with(".txt")
                })
        })
        .unwrap_or_else(|| {
            panic!(
                "CMake did not generate a ShaderToy Rust static-link manifest in {}",
                build_dir.display()
            )
        });

    let body = fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest.display()));
    let mut search_paths = BTreeSet::new();
    let mut static_libraries = Vec::new();
    let mut system_libraries = Vec::new();
    let mut frameworks = Vec::new();

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(path) = line.strip_prefix("static=") {
            let path = PathBuf::from(path);
            let parent = path
                .parent()
                .unwrap_or_else(|| panic!("static library path has no parent: {}", path.display()));
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_else(|| panic!("invalid static library filename: {}", path.display()));
            search_paths.insert(parent.to_path_buf());
            static_libraries.push(filename.to_string());
        } else if let Some(name) = line.strip_prefix("system=") {
            if !name.is_empty() {
                system_libraries.push(name.to_string());
            }
        } else if let Some(name) = line.strip_prefix("framework=") {
            if !name.is_empty() {
                frameworks.push(name.to_string());
            }
        } else {
            panic!("unsupported ShaderToy Rust link-manifest entry: {line}");
        }
    }

    for path in search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for library in static_libraries {
        println!("cargo:rustc-link-lib=static:+verbatim={library}");
    }
    for library in system_libraries {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
    for framework in frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
