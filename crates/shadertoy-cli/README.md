# shadertoy-cli

Agent-friendly ShaderToy project, rendering, debugging, and live-preview CLI.

The installed binary is named `shadertoy`.

```bash
cargo binstall shadertoy-cli
shadertoy new demo --template multipass
cd demo
shadertoy check
shadertoy render -o target/frame.png
shadertoy render-frames --range 0:180:60 --contact-sheet target/contact.png
shadertoy render-video --frames 180 -o target/clip.mp4
shadertoy inspect buffer buffer-a --frame 120 --pixel 8,8
shadertoy inspect storage particle-state --frame 120 --type f32 --count 16
shadertoy profile --frame 120 --samples 30
shadertoy test
shadertoy preview --record target/session.strec
shadertoy replay target/session.strec -o target/replay.png
```

The CLI embeds its project templates, JSON Schema, agent documentation, preview web UI, and Camoufox import helper, so the installed executable does not need adjacent data files. Built-in topics are available through `shadertoy docs` for agent, project, import, manifest, passes, glsl, assets, buffers, channels, state, and preview guidance. Use `shadertoy docs glsl` for the generated prelude/entry-point contract and `shadertoy docs assets` for cubemap and volume file layouts.

Import a public ShaderToy into a fully local editable project:

    shadertoy import https://www.shadertoy.com/view/lsX3W4 -o mandelbrot
    cd mandelbrot
    shadertoy check

Import uses a Camoufox browser session to handle ShaderToy's browser/Cloudflare path. On first use it creates a private cached Python environment, installs the pinned Camoufox adapter, and fetches its browser. Python 3.10+ is required. On Linux, the Camoufox browser also needs the usual Firefox GTK runtime (for Debian/Ubuntu, `libgtk-3-0` or its distro equivalent). Supported textures, cubemaps, volumes, file-backed video inputs, and Sound passes are materialized into the local project; the original ShaderToy response is retained under .shadertoy/import-response.json. Webcam inputs remain live-preview-only.

See the repository README for project format, state/debugging workflows, and the underlying C++/Rust library architecture.

The underlying C++ renderer was originally written by Yingwei Zheng ([dtcxzyw/shadertoy](https://github.com/dtcxzyw/shadertoy)), whose groundwork this CLI builds on.

On Linux, GL-backed CLI operations including `check`, `render`, `render-frames`, `render-video`, `render-audio`, `inspect buffer`, `inspect storage`, state capture, `profile`, `test`, `replay`, and native preview use a surfaceless EGL context and do not need `DISPLAY` or `WAYLAND_DISPLAY`. `render-frames` and `render-video` reuse one deterministic runtime across their requested timeline. Encoded video and file-backed video channels require `ffmpeg`/`ffprobe`.


Additional development tooling
------------------------------

GLSL sources support project-local quoted `#include` directives. Configure
shared include roots with `[shader] include_dirs = ["shaders/lib"]`; live
preview tracks the include dependency graph and recompiles only affected passes,
preserving feedback buffers when possible.

Projects can declare typed custom uniforms with `[[uniform]]`; defaults are applied
everywhere, `render`/`render-frames`/`render-video`/`profile` accept `--set name=value`,
preview exposes matching controls, and `[[test]]` cases can override them independently.

Buffer passes can opt into fixed `width`/`height` render targets for stable
simulation grids, while each iChannel can independently choose
`filter = "nearest" | "linear" | "mipmap"` and
`wrap = "clamp" | "repeat"`.

For heavier pipelines, `kind = "compute"` adds fixed-size OpenGL 4.3 compute
dispatch, typed `r32f`/`rg32f`/`rgba16f`/`rgba32f` outputs, repeated dispatch
iterations via `iIteration`, writable `iOutput` image load/store, and persistent
named SSBOs shared across passes. Buffer/compute passes can also expose up to eight render targets with extra_outputs, and consumers select an attachment with output = N. `inspect buffer --output-index N --raw ...` reads any MRT attachment, while `inspect storage` can decode/dump a named SSBO. `.ststate` format 4 can optionally include SSBO bytes with `state capture --include-storage`. See `shadertoy docs passes` and
`shadertoy docs channels` for the exact semantics.

For automated visual/numeric validation, add `[[test]]` cases to
`ShaderToy.toml` and run `shadertoy test`. Visual cases compare
deterministic PNGs with an RMSE tolerance and emit actual/expected/diff artifacts
on failure; buffer cases can also assert no NaN/Inf values and finite-value mean
ranges. Tests can also expand across `frames = [...]` and `resolutions = [[w,h], ...]`,
repeat each variant with `assert_deterministic = true`, and prove fixed GPU
simulation grids are independent of output size with
`assert_resolution_independent = true`.

ShaderToy Sound passes (`kind = "sound"`) are compiled by `check` and render to
deterministic stereo PCM WAV with `shadertoy render-audio`. File-backed video
assets update at deterministic `iTime` and expose `iChannelTime`; webcam channels
are available only in live preview and are intentionally rejected by headless
render/replay/recording.
