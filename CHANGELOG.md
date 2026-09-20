# Changelog

All notable changes to the Rust library and CLI are recorded here.

## Unreleased

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
