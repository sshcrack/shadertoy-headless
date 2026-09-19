use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../shadertoy-c/include/shadertoy/shadertoy.h");
    println!("cargo:rerun-if-changed=../../shadertoy-c/src/shadertoy.cpp");
    println!("cargo:rerun-if-changed=../../shadertoy");
    println!("cargo:rerun-if-changed=../../CMakeLists.txt");
    println!("cargo:rerun-if-changed=../../vcpkg.json");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .join("../..")
        .canonicalize()
        .expect("failed to locate repository root");
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

    let mut config = cmake::Config::new(&repo_root);
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
        .define("SHADERTOY_BUILD_C_API", "ON")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    if let Ok(vcpkg_root) = env::var("VCPKG_ROOT").or_else(|_| env::var("VCPKG_INSTALLATION_ROOT"))
    {
        let toolchain = Path::new(&vcpkg_root)
            .join("scripts")
            .join("buildsystems")
            .join("vcpkg.cmake");
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain);

        let vcpkg_root = out_dir.join("vcpkg");
        let install_options = format!(
            "--x-buildtrees-root={};--x-packages-root={};--downloads-root={}",
            vcpkg_root.join("buildtrees").display(),
            vcpkg_root.join("packages").display(),
            vcpkg_root.join("downloads").display()
        );
        config.define("VCPKG_INSTALL_OPTIONS", install_options);
    }

    let destination = config.build();
    let lib_dir = destination.join("lib");
    let bin_dir = destination.join("bin");

    // Cargo makes native build-script search paths available while it launches
    // binaries itself, but a directly executed target/{profile}/shadertoy also
    // needs a stable relative location. Stage the C ABI next to the Cargo target
    // tree and embed only a relative loader path (never an OUT_DIR path).
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected Cargo OUT_DIR layout");
    if cfg!(target_os = "windows") {
        let source = bin_dir.join("shadertoy_c.dll");
        let destination = profile_dir.join("shadertoy_c.dll");
        std::fs::copy(&source, &destination).unwrap_or_else(|error| {
            panic!(
                "failed to stage {} at {}: {error}",
                source.display(),
                destination.display()
            )
        });
    } else {
        let target_root = profile_dir
            .parent()
            .expect("Cargo profile directory should have a target root");
        let staged_lib_dir = target_root.join("lib");
        std::fs::create_dir_all(&staged_lib_dir)
            .expect("failed to create staged native library directory");
        let library_name = if cfg!(target_os = "macos") {
            "libshadertoy_c.dylib"
        } else {
            "libshadertoy_c.so"
        };
        let source = lib_dir.join(library_name);
        let staged = staged_lib_dir.join(library_name);
        std::fs::copy(&source, &staged).unwrap_or_else(|error| {
            panic!(
                "failed to stage {} at {}: {error}",
                source.display(),
                staged.display()
            )
        });
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
        }
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", bin_dir.display());
    println!("cargo:rustc-link-lib=dylib=shadertoy_c");
    println!("cargo:libdir={}", lib_dir.display());
    println!("cargo:bindir={}", bin_dir.display());
}
