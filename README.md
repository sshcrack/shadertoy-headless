# shadertoy

[![build-windows](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-windows.yml/badge.svg)](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-windows.yml)
[![build-linux](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-linux.yml/badge.svg)](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-linux.yml)
[![build-macos](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-macos.yml/badge.svg)](https://github.com/dtcxzyw/shadertoy/actions/workflows/build-macos.yml)

An unofficial ShaderToy renderer and live editor.

The repository has two layers built on the same rendering implementation:

- **shadertoy::shadertoy** — a standalone C++23 library for importing, compiling, advancing, and rendering ShaderToy pipelines.
- **shadertoy** — the interactive editor. It is a client of the library and adds the ImGui pipeline editor, GLSL editor, dialogs, screenshots, and desktop interaction.

The library does not depend on ImGui, HelloImGui, the node editor, GLFW, or native file dialogs. The embedding application owns the OpenGL context and makes it current; the library owns the renderer and OpenGL loader details.

## Architecture

The library seam is the shader document/runtime interface:

~~~
                 +-----------------------+
                 |     ShaderDocument    |
                 | passes, inputs, links |
                 +-----------+-----------+
                             |
              import/STTF    |    compile
                             v
+-------------+       +------+-------+       +----------------+
| ShaderToy   | ----> |    Runtime   | ----> | OpenGL renderer|
| importer    |       | time + input |       | passes/buffers |
+-------------+       +------+-------+       +----------------+
                             ^
                             |
                    +--------+---------+
                    | GUI / other host |
                    | owns GL context  |
                    +------------------+
~~~

ShaderDocument is GUI-independent, so the desktop editor and embedded/headless users compile exactly the same pipeline representation. Runtime owns playback state, ShaderToy uniforms, host audio/keyboard/mouse state, and the compiled renderer.

## Using the library

After installing the project:

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

Clone with the text-editor submodule:

~~~bash
git clone --recursive https://github.com/dtcxzyw/shadertoy.git
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

# Library + deterministic preview CLI
cmake --preset preview
cmake --build --preset preview
~~~

The equivalent configuration switches are SHADERTOY_BUILD_GUI, SHADERTOY_BUILD_PREVIEW_TOOL, and BUILD_TESTING.

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

See the [Releases page](https://github.com/dtcxzyw/shadertoy/releases) for pre-built binaries.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
