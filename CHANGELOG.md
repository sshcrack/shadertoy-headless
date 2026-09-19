# Changelog

All notable changes to the Rust library and CLI are recorded here.

## Unreleased

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
