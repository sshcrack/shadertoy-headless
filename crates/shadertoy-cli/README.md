# shadertoy-cli

Agent-friendly ShaderToy project, rendering, debugging, and live-preview CLI.

The installed binary is named `shadertoy`.

```bash
cargo binstall shadertoy-cli
shadertoy new demo --template multipass
cd demo
shadertoy check
shadertoy render -o target/frame.png
shadertoy render-frames --frames 0,60,120,180 --contact-sheet target/contact.png
shadertoy inspect buffer buffer-a --frame 120 --pixel 8,8
shadertoy profile --frame 120 --samples 30
shadertoy test
shadertoy preview --record target/session.strec
shadertoy replay target/session.strec -o target/replay.png
```

The CLI embeds its project templates, JSON Schema, agent documentation, preview web UI, and Camoufox import helper, so the installed executable does not need adjacent data files. Built-in topics are available through `shadertoy docs` for agent, project, import, manifest, passes, buffers, channels, state, and preview guidance.

Import a public ShaderToy into a fully local editable project:

    shadertoy import https://www.shadertoy.com/view/lsX3W4 -o mandelbrot
    cd mandelbrot
    shadertoy check

Import uses a Camoufox browser session to handle ShaderToy's browser/Cloudflare path. On first use it creates a private cached Python environment, installs the pinned Camoufox adapter, and fetches its browser. Python 3.10+ is required. On Linux, the Camoufox browser also needs the usual Firefox GTK runtime (for Debian/Ubuntu, `libgtk-3-0` or its distro equivalent). Supported textures, cubemaps, and volumes are downloaded into the project; the original ShaderToy response is retained under .shadertoy/import-response.json.

See the repository README for project format, state/debugging workflows, and the underlying C++/Rust library architecture.

The underlying C++ renderer was originally written by Yingwei Zheng ([dtcxzyw/shadertoy](https://github.com/dtcxzyw/shadertoy)), whose groundwork this CLI builds on.

On Linux, GL-backed CLI operations including `check`, `render`, `render-frames`, `inspect buffer`, state capture, `profile`, `test`, `replay`, and native preview use a surfaceless EGL context and do not need `DISPLAY` or `WAYLAND_DISPLAY`. `render-frames` reuses one deterministic runtime across all requested frames and can emit a contact sheet for visual iteration.


Additional development tooling
------------------------------

GLSL sources support project-local quoted `#include` directives. Configure
shared include roots with `[shader] include_dirs = ["shaders/lib"]`; live
preview tracks the include dependency graph and recompiles only affected passes,
preserving feedback buffers when possible.

Buffer passes can opt into fixed `width`/`height` render targets for stable
simulation grids, while each iChannel can independently choose
`filter = "nearest" | "linear" | "mipmap"` and
`wrap = "clamp" | "repeat"`. See `shadertoy docs passes` and
`shadertoy docs channels` for the exact semantics.

For automated visual/numeric validation, add `[[test]]` cases to
`ShaderToy.toml` and run `shadertoy test`. Visual cases compare
deterministic PNGs with an RMSE tolerance and emit actual/expected/diff artifacts
on failure; buffer cases can also assert no NaN/Inf values and finite-value mean
ranges.
