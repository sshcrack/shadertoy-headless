# Changelog

All notable changes to the Rust library and CLI are recorded here.

## Unreleased

## 2.1.7

- Added RGBA32F buffer inspection with per-channel statistics, NaN/Inf detection, pixel probing, and diagnostic visualizations.
- Added per-pass GPU timer-query profiling, CPU render-call timing, and persistent-buffer VRAM estimates.
- Added project-local GLSL include support with dependency-aware transactional hot reload that preserves feedback buffers.
- Added deterministic manifest-driven visual/numeric regression tests with baseline updates and diff artifacts.
- Added preview input recording and deterministic replay with exact ShaderToy timing state, SHA-256 project fingerprints, and invalidation when project files change.

## 2.1.6

- Added fixed per-buffer render resolutions for stable simulation/FFT grids while preserving output-sized behavior by default.
- Added per-channel sampler controls for nearest/linear/mipmap filtering and clamp/repeat wrapping, including independent sampler state when the same source is bound multiple times.
- Hardened fixed-buffer state restore, mixed-resolution channel reporting, native dimension validation, and raw shader channel-slot validation.

## 2.1.5

- Fixed the Windows release triplet selection: the static triplet must be passed via `VCPKG_DEFAULT_TRIPLET` (the env var the vcpkg tool honors in manifest mode), not `VCPKG_TARGET_TRIPLET` (only read as a CMake variable, so 2.1.3/2.1.4 silently built the dynamic triplet again).
- Installed EGL runtime (`libegl1`, Mesa DRI drivers) on Linux CI so headless render tests can create a context.

## 2.1.4

- Fixed the Windows release link (`LNK2019` on `__imp_*` CRT symbols): when Rust links the static CRT, the native code is now compiled `/MT` too (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` in `shadertoy-sys`).
- Fixed `clippy -D warnings` on current stable (`chunks_exact_to_as_chunks` in `state.rs`).

## 2.1.3

- Fixed the Windows CLI dying on startup with a missing-DLL error (`STATUS_DLL_NOT_FOUND`): the release build now links vcpkg ports (notably `glfw3`) and the CRT statically (`x64-windows-static`, `+crt-static`), so the shipped `shadertoy.exe` needs no sidecar DLLs or VC redist.

## 2.1.2

- Fixed `cargo binstall` on Windows: use an exact-target override (`x86_64-pc-windows-msvc`, `zip`) instead of a `cfg(target_os = "windows")` override, which `cargo-binstall` versions before ~1.17 silently ignore (they then probe for a nonexistent `.tgz` instead of the shipped `.zip`).

## 2.1.1

- Fixed heap corruption when rendering or previewing RGB output whose width is not 4-byte aligned.
- Hardened related OpenGL pixel-transfer paths, including tightly packed single-channel volumes.
- Made the live WebSocket/canvas preview responsive and touch-friendly on mobile.

## 2.1.0

- Made Linux offscreen rendering display-less (surfaceless EGL) so headless rendering needs no X/Wayland display.
- Added deterministic multi-frame rendering with frame streaming.
- Streamed live preview frames over the WebSocket canvas.
- Added Camoufox-backed ShaderToy import for materializing remote shaders into local projects.

## 2.0.1

- Hardened CLI project mutations and manifest paths against traversal, symlink escapes, reserved input-name collisions, and partial failed writes.
- Fixed live preview lifecycle so GLFW/OpenGL context creation, rendering, and destruction remain on the process main thread; improved auth URL handling and shutdown behavior.
- Made JSON mode cover command-line parse failures and added project-path option aliases for check and build.
- Hardened state, STTF, C ABI, and safe Rust buffer/dimension handling against malformed sizes, overflow, unsupported volume layouts, and allocation failures.
- Expanded CLI/native regression coverage for filesystem safety, malformed state/STTF data, and invalid C API dimensions.

## 2.0.0

- Breaking release relative to the previous 1.x C++ line: separated the reusable renderer/library from the desktop editor and made the native API suitable for external consumers.
- Added the agent-first `shadertoy` CLI, safe Rust wrapper, and generated sys bindings.
- Added schema-backed directory projects, deterministic rendering, resumable buffer state, and native live preview.
- Added static native linkage for the CLI and crates.io/cargo-binstall release infrastructure.
