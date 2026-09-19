# shadertoy

[![build-windows](https://github.com/sshcrack/shadertoy/actions/workflows/build-windows.yml/badge.svg)](https://github.com/sshcrack/shadertoy/actions/workflows/build-windows.yml)
[![build-linux](https://github.com/sshcrack/shadertoy/actions/workflows/build-linux.yml/badge.svg)](https://github.com/sshcrack/shadertoy/actions/workflows/build-linux.yml)
[![build-macos](https://github.com/sshcrack/shadertoy/actions/workflows/build-macos.yml/badge.svg)](https://github.com/sshcrack/shadertoy/actions/workflows/build-macos.yml)

An unofficial ShaderToy renderer and live editor.

The repository has several frontends built on one rendering implementation:

- **shadertoy::shadertoy** — the standalone C++23 renderer/library.
- **shadertoy-c** — a deliberately small stable C ABI over project construction and runtime rendering.
- **Rust shadertoy-sys / shadertoy-native crates** — generated raw bindings plus a safe, idiomatic wrapper.
- **shadertoy CLI** — the agent-first project/check/render/debug/live-preview workflow.
- **desktop editor** — the existing ImGui editor. It remains a C++ library client; direct directory-project editing will be adapted later.

The core C++ library does not depend on ImGui, HelloImGui, the node editor, GLFW, or native file dialogs. The embedding application owns the OpenGL context and makes it current. The optional C ABI includes a hidden GLFW context helper for command-line/off-screen hosts.

## Architecture

~~~
 ShaderToy.toml + GLSL/assets
             |
             v
      project semantics
             |
             v
      ShaderDocument
             |
             v
       C++ Runtime  -----------------> desktop GUI
             |
             v
       stable C ABI
             |
       +-----+------+
       |            |
 shadertoy-sys   other FFI
       |
 safe Rust shadertoy-native (crate name: shadertoy)
       |
 shadertoy CLI
  check / render / inspect / state / preview
~~~

ShaderDocument remains GUI-independent. Directory-project graph semantics live below the frontends, while the Rust CLI owns TOML/schema/file-watching/web-server concerns. shadertoy-sys generates its bindings directly from shadertoy-c/shadertoy.h; CLI code never handles raw C pointers.

## Using the library

For Rust, the safe wrapper is published as `shadertoy-native` while its library target remains `shadertoy`:

~~~bash
cargo add shadertoy-native
~~~

~~~rust
use shadertoy::Runtime;
~~~

For C++, after installing the project:

~~~cmake
find_package(shadertoy CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE shadertoy::shadertoy)
~~~

Create the graphics context in your application and make it current before compiling or rendering a document:

~~~cpp
#include <shadertoy/ShaderToy.hpp>

ShaderToy::Runtime runtime;

// A compatible OpenGL context must be current here.
auto loaded = runtime.loadImageShader(
    "example",
    R"(
        void mainImage(out vec4 color, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            color = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
        }
    )"
);

if (!loaded)
    throw loaded.error();

runtime.tickFixed(1.0f / 60.0f, 60.0f);
auto rgb = runtime.renderToBuffer({128.0f, 128.0f});
~~~

The host may instead construct/load a ShaderDocument, import a ShaderToy URL/response, or load an STTF file and pass the resulting document to Runtime::setDocument().

Runtime can be constructed before an OpenGL context exists. A current context is required when a document is compiled and whenever it is rendered. The library initializes its own OpenGL loader; the host does not need to use GLEW directly.

## Agent-first CLI

Install the prebuilt CLI with cargo-binstall:

~~~bash
cargo binstall shadertoy-cli
shadertoy --version
~~~

The binary package is `shadertoy-cli`; the installed executable is `shadertoy`. Source installation with `cargo install shadertoy-cli` is also supported when the required native build dependencies are available, but cargo-binstall is the preferred fast path.

A CLI project is an editable directory rather than a monolithic STTF file:

~~~text
my-shader/
├── ShaderToy.toml
├── README.md
├── shaders/
│   ├── image.frag
│   └── buffer-a.frag
├── assets/
└── target/          # generated, gitignored
~~~

Create a minimal project, a working feedback-buffer example, or import a public ShaderToy:

~~~bash
shadertoy new hello
shadertoy new feedback --template multipass
shadertoy import https://www.shadertoy.com/view/lsX3W4 -o mandelbrot
~~~

ShaderToy import uses a Camoufox browser session rather than direct HTTP, then materializes supported shader passes, pass/feedback wiring, sampler settings, textures, cubemaps, and volumes into the normal editable directory format. Imported projects retain source metadata plus the exact browser response in .shadertoy/import-response.json. On first use the CLI creates a private cached Python environment for its pinned Camoufox adapter and fetches the matching browser; Python 3.10+ is required.

The multipass template intentionally demonstrates Buffer A previous-frame feedback and wiring the current Buffer A result into Image through iChannel0.

The common agent loop is deliberately small:

~~~bash
cd feedback
shadertoy inspect --json
shadertoy check --json
shadertoy render -o target/check.png
shadertoy preview
~~~

When something is wrong, the same interface drills down instead of requiring renderer internals:

~~~bash
shadertoy inspect graph --json
shadertoy inspect pass buffer-a --json
shadertoy inspect channels image --json
shadertoy render --pass buffer-a -o target/buffer-a.png
~~~

shadertoy state captures lossless RGBA32F feedback-buffer state together with deterministic time/frame metadata. That makes multipass bugs resumable and lets an agent replace one buffer with a known exact-size image:

~~~bash
shadertoy state capture --frame 300 -o target/frame300.ststate
shadertoy state inspect target/frame300.ststate --json

shadertoy render   --state target/frame300.ststate   --set-buffer buffer-a=fixtures/known.png   -o target/debug.png
~~~

Use shadertoy docs agent for the concise workflow embedded in the executable. Other topics include project, manifest, passes, buffers, channels, state, and preview.

### Schema-backed ShaderToy.toml

The canonical JSON Schema is checked in at crates/shadertoy-cli/assets/schema/shadertoy.schema.json and generated from the same Rust types that parse the manifest. A test prevents the checked-in schema from drifting from those types.

New projects receive a local .shadertoy/shadertoy.schema.json, a #:schema directive in ShaderToy.toml, and a .taplo.toml association. Editors with Taplo / compatible TOML schema support can therefore validate keys and types and provide completion. Agents can print the exact same schema with:

~~~bash
shadertoy docs manifest --schema
~~~

shadertoy check adds semantic validation that JSON Schema cannot express, including graph references/cycles and real GLSL compilation through the native renderer.

### Live native preview

shadertoy preview starts a local web server, but the browser is only a viewer/controller: rendering remains in the native C++ renderer. It watches the manifest, shader sources, and assets, keeps the last successful frame when a new edit fails compilation, and hot-reloads automatically after the error is fixed.

The preview exposes final Image and named 2D buffers plus pause/resume, reset, frame step, time scale, resolution, mouse, and keyboard controls. It binds to 127.0.0.1 by default; non-loopback binds require --token.

ShaderToy-page import is available through shadertoy import URL_OR_ID. It is isolated behind the Camoufox browser adapter; normal project checking, rendering, building, and previewing remain browser-independent.

## Interactive editor

The desktop application keeps the original editing workflow on top of the library:

- visual render-pass/pipeline editor;
- GLSL source editor;
- ShaderToy URL import;
- STTF open/save;
- live canvas with mouse and keyboard channels;
- pause/reset/time-scale controls;
- screenshots and application logging.

Run it with an optional STTF file or ShaderToy URL:

~~~bash
./build/app/shadertoy [shader.sttf|https://www.shadertoy.com/view/...]
~~~

## Host audio / music visualization

Embedders can feed live music analysis through Runtime::setAudioInput().

Shaders that declare a ShaderToy music, musicstream, mic, or audio input receive the conventional 512x2 audio texture in their assigned iChannelN:

- row 0: normalized spectrum, low frequencies on the left;
- row 1: waveform encoded as 0..1 (sampleAudioWaveform() returns -1..1).

The renderer also exposes semantic aliases including iAudioLoudness, iAudioBass, iAudioMid, iAudioTreble, iAudioKick, iAudioSnare, iAudioHihat, iAudioOnset, iAudioBpm, iAudioBeatPhase, iAudioBeatConfidence, iAudioBeatStrength, iAudioDrop, and iAudioSectionChange. iAudioAvailable is 1 when the host supplied a current frame.

A minimal audio-reactive shader can use both the ShaderToy-style audio texture and semantic uniforms:

~~~glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float spectrum = sampleAudioSpectrum(iChannel0, clamp(abs(uv.x), 0.0, 1.0));
    float glow = exp(-10.0 * abs(length(uv) - (0.25 + 0.12 * iAudioBass)));
    vec3 color = (0.35 + 0.65 * cos(iTime + vec3(0.0, 2.0, 4.0)))
               * (glow + spectrum * 0.6);
    color *= 0.7 + 0.45 * iAudioBeatStrength;
    fragColor = vec4(color, 1.0);
}
~~~

## Features

Render passes:

- [x] Image
- [x] Cubemap
- [x] Buffer
- [x] Common
- [ ] Sound

Channels:

- [x] Textures
- [x] Volumes
- [x] Cubemaps
- [x] Buffer / previous frame
- [x] Keyboard
- [x] Music and microphone (host-provided)
- [ ] Video
- [ ] Webcam

Editor/runtime utilities:

- [x] Import from shadertoy.com
- [x] Import directly from a ShaderToy response body
- [x] Render pass editor
- [x] GLSL shader editor
- [x] STTF import/export
- [x] Off-screen RGB rendering
- [x] Deterministic fixed-step rendering
- [x] Screenshots
- [ ] Video recording
- [ ] Custom uniforms
- [ ] Custom meshes
- [ ] Anti-aliasing controls

## Build

### Prerequisites

- Windows, Linux, or macOS
- CMake 3.20+
- a C++23 compiler
- [vcpkg](https://github.com/microsoft/vcpkg)
- OpenGL
- for the Rust CLI: stable Rust with Edition 2024 support plus Clang/libclang for bindgen

Clone with the text-editor submodule:

~~~bash
git clone --recursive https://github.com/sshcrack/shadertoy.git
cd shadertoy
~~~

Set VCPKG_ROOT, then use one of the checked-in presets:

~~~bash
# GUI + library
cmake --preset default
cmake --build --preset default

# Standalone library only: no GUI dependencies are selected
cmake --preset library
cmake --build --preset library
ctest --preset library

# Library + legacy deterministic preview fixture
cmake --preset preview
cmake --build --preset preview

# Library + stable C ABI
cmake --preset c-api
cmake --build --preset c-api

# Agent-first Rust CLI. shadertoy-sys builds the C ABI and generates bindings.
cargo build -p shadertoy-cli
./target/debug/shadertoy --help
~~~

The Rust CLI statically links the ShaderToy C ABI, C++ renderer, and vcpkg-provided native dependencies into the executable. Normal operating-system runtime/graphics libraries (for example libc, libstdc++, OpenGL/GLX, and X11 on Linux) remain dynamically linked, but no adjacent `libshadertoy_c`/GLFW/GLEW/fmt/OpenSSL/Brotli shared libraries are required.

The equivalent CMake switches are `SHADERTOY_BUILD_GUI`, `SHADERTOY_BUILD_PREVIEW_TOOL`, `SHADERTOY_BUILD_C_API`, `SHADERTOY_BUILD_C_API_STATIC`, and `BUILD_TESTING`.

### Laptop MCP

Current Laptop MCP managed images install the pinned `shadertoy-cli` release globally, so fresh sandboxes can invoke `shadertoy` without repository configuration or a source build. This repository therefore does not declare a ShaderToy-specific workspace bootstrap contract.

The generic `[image].cargo_binstall` and per-session `sandbox_config_update(cargo_binstall=[...])` mechanisms remain available for other exact-version prebuilt Rust tools.

### Install the standalone package

~~~bash
cmake --preset library -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build --preset library
cmake --install build-lib
~~~

This installs the public headers, libshadertoy, and a CMake package exporting shadertoy::shadertoy.

## Deterministic preview tool

The optional shadertoy-preview executable creates a hidden GLFW context, advances fixed frame time, injects repeatable synthetic music analysis, renders through the same library pipeline, and writes a PNG:

~~~bash
./build-preview/shadertoy-preview examples/music_neon_orbit.frag preview.png 256 128 120
~~~

On Linux without a display server, run it under a virtual/headless display that provides OpenGL.

## Gallery

[![goo](https://user-images.githubusercontent.com/15650457/236786522-80c10c46-f3b0-46f3-88ef-abbe39c3cd5f.png)](https://www.shadertoy.com/view/lllBDM)

[![expansive reaction-diffusion](https://user-images.githubusercontent.com/15650457/236787527-b26fa835-1d36-4dc6-be59-6d508e898e04.png)](https://www.shadertoy.com/view/4dcGW2)

[![mandelbrot](https://user-images.githubusercontent.com/15650457/236788040-2411c757-7c51-407a-869f-5c6709bf5e5d.png)](https://www.shadertoy.com/view/lsX3W4)

[![MultiscaleMIPFluid](https://user-images.githubusercontent.com/15650457/236790106-5ebeb8a2-0c16-4cbd-a7cf-d8bbb21ad613.png)](https://www.shadertoy.com/view/tsKXR3)

[![RainbowSand](https://user-images.githubusercontent.com/15650457/236790355-c20303e1-7abd-4d42-9088-2133a0e756fa.png)](https://www.shadertoy.com/view/stdyRr)

[![subsurface](https://user-images.githubusercontent.com/15650457/236790664-3defcade-c5b4-4f9c-9f21-0a1b67b72536.png)](https://www.shadertoy.com/view/dltGWl)

[![noise-contour](https://user-images.githubusercontent.com/15650457/236791146-b3b9cdff-6754-42ae-83c3-d69ef2ea9387.png)](https://www.shadertoy.com/view/MscSzf)

[![cubemaps](https://github.com/dtcxzyw/shadertoy/assets/15650457/e1719ed0-2748-47e0-a7eb-db6d35cd8dec)](https://www.shadertoy.com/view/MsXGz4)

See [examples](examples) for more shaders.

## Releases

See the [Releases page](https://github.com/sshcrack/shadertoy/releases) for pre-built binaries.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
