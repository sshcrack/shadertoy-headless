# Changelog

All notable changes to the Rust library and CLI are recorded here.

## Unreleased

## 2.2.7

- Fixed per-pass GPU profiling attribution for deferred compute work by isolating pass completion while profiling, so compute-heavy passes no longer report near-zero time while their cost is charged to a later consumer.
- Added `shadertoy blind create` for source-agnostic A/B/C comparisons of individual images, image/render directories, ShaderToy project directories, STTF builds, and Git revisions, with whole render sets kept together under one anonymous label.
- Added direct STTF loading to the Rust/C runtime bridge so built artifacts can participate in deterministic render comparisons.


## 2.2.6

- Added bias-resistant blind parameter sweeps with randomized A/B/C variant labels and a public contact sheet/session report that omits parameter assignments.
- Added shadertoy blind judge to commit a selected anonymous variant plus rationale before the parameter mapping can be revealed.
- Added shadertoy blind reveal to unlock the mapping only after judgment and write a combined blind-reveal.json report containing the decision, selected settings, and full mapping.

## 2.2.5

- Added `--set NAME=VALUE` custom-uniform overrides to `state capture`, `inspect buffer`, and `inspect storage`, completing override support across deterministic runtime inspection/state-capture workflows.
- Added `shadertoy sweep` for bounded Cartesian custom-uniform sweeps with deterministic variant PNGs, named-pass support, JSON metadata, and contact-sheet generation.
- Expanded profiler statistics with median and p95 for each GPU pass, aggregate GPU timing, and CPU render-call timing while preserving the existing mean field for compatibility.

## 2.2.2

- Re-verified the complete v2.2 advanced GPU pipeline on the applied release branch, including compute passes, typed render targets, MRT routing, persistent/shared SSBOs, fixed-size passes, state capture/restore, sampler controls, and headless EGL rendering.
- Added native and CLI regression matrices covering all four typed compute formats (`r32f`, `rg32f`, `rgba16f`, and `rgba32f`).

## 2.2.1

- Automatically refresh an existing project-local .shadertoy/shadertoy.schema.json when the installed CLI embeds a newer schema, while leaving projects without a local schema untouched.
- Fixed fragment/image/buffer/cubemap SSBO binding and shader-storage barriers so named storage is actually shared across fragment and compute passes and remains visible across frames.
- Reject compute local workgroup Z sizes other than 1 because the public compute entrypoint is two-dimensional (`mainCompute(ivec2)`) and larger Z sizes duplicated writes/atomics to the same pixel.
- Harden raw STTF loading/compilation against duplicate SSBO binding points and document SSBO pass-ordering and state-capture semantics.
- State format 3 records each persistent pass's render-target format and rejects incompatible typed-buffer restores instead of silently dropping or reinterpreting channels; state formats 1 and 2 remain readable.

## 2.2.0

- Added first-class OpenGL 4.3 compute passes with fixed dispatch dimensions, configurable local workgroups, writable typed image outputs, and repeated per-frame dispatch through iIteration.
- Added persistent zero-initialized named shader-storage buffers shared across passes, enabling structured GPU state and GLSL atomic operations without packing data into texture channels.
- Added explicit r32f, rg32f, rgba16f, and rgba32f formats for 2D buffer/compute outputs; inspection, state capture, regression tests, preview, and VRAM profiling understand compute outputs and their actual storage cost.
- Added up to eight render targets per buffer/compute pass, with fragment location outputs, compute iOutput1..iOutput7 images, and per-channel output selection for downstream passes.
- Kept ordinary shader compatibility by preferring OpenGL 4.3 contexts and falling back to OpenGL 4.1 when advanced compute/SSBO features are not used.

## 2.1.8

- Synchronized CLI help, repository/crate READMEs, built-in documentation references, and generated-project guidance with the current import, fixed-buffer/sampler, inspection/profiling, regression-test, and record/replay workflows.

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
