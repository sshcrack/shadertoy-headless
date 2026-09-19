fn main() {
    // shadertoy-sys stages the native C ABI in the Cargo target root's lib/
    // directory. Keep the executable relocatable within the conventional
    // bin/../lib release-bundle layout instead of embedding an OUT_DIR path.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
    } else if cfg!(all(unix, not(target_os = "macos"))) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    }
}
