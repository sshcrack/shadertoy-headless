# shadertoy-cli

Agent-friendly ShaderToy project, rendering, debugging, and live-preview CLI.

The installed binary is named `shadertoy`.

```bash
cargo binstall shadertoy-cli
shadertoy new demo --template multipass
cd demo
shadertoy check
shadertoy check --pedantic
shadertoy graph --dot target/graph.dot
shadertoy render -o target/frame.png
shadertoy render-frames --range 0:180:60 --contact-sheet target/contact.png
shadertoy render-video --frames 180 -o target/clip.mp4
shadertoy sweep --frame 120 --set foam_gain=0.8,1.0,1.2
shadertoy sweep --blind --frame 120 --set foam_gain=0.8,1.0,1.2
shadertoy blind create target/old-renders target/new-renders --output-dir target/old-vs-new
shadertoy blind create 'project:.@preset=high' 'project:.@preset=medium' 'project:.@preset=low' --frames 60,180,300
shadertoy blind judge target/old-vs-new/blind-session.json --pick B --reason "preferred breakup"
shadertoy blind reveal target/old-vs-new/blind-session.json
shadertoy experiment --baseline git:main --candidate git:HEAD --frames 0,60,120 --blind
shadertoy inspect buffer buffer-a --frame 120 --pixel 8,8 --set foam_gain=1.0
shadertoy inspect storage particle-state --frame 120 --type f32 --count 16
shadertoy profile --preset medium --frame 120 --samples 30
shadertoy profile --preset medium --frame 120 --samples 30 --sync-per-pass
shadertoy test --ci
shadertoy trace capture --frame 120 --include-intermediates -o target/bug.sttrace
shadertoy trace inspect target/bug.sttrace
shadertoy trace replay target/bug.sttrace
shadertoy preview --record target/session.strec
shadertoy replay target/session.strec -o target/replay.png
```

The CLI embeds its project templates, JSON Schema, agent documentation, preview web UI, and Camoufox import helper, so the installed executable does not need adjacent data files. Built-in topics are available through `shadertoy docs` for agent, project, import, manifest, passes, glsl, assets, buffers, channels, state, sweep, blind, experiment, test, trace, graph, and preview guidance. Use `shadertoy docs glsl` for the generated prelude/entry-point contract and `shadertoy docs assets` for cubemap and volume file layouts.

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
everywhere, deterministic rendering/state-capture/runtime-inspection commands accept
`--set name=value`, preview exposes matching controls, and `[[test]]` cases can override
them independently. Named `[preset.NAME]` sections keep quality tiers in the same
project root; `render_scale` adjusts default output size while
`[preset.NAME.pass.PASS]` can override fixed pass dimensions and compute
iterations/local size. `check`, `build`, `preview`, `render`, `render-frames`,
`render-video`, `sweep`, and `profile` accept `--preset NAME`.

Preview and profile deliberately do not count a preset's smaller final output as
an optimization. Preview holds the final Image at the base project output size
across preset switches (or at a human-selected review size), and
`profile --preset NAME` benchmarks at base project output size by default while
reporting the preset-requested scale separately. Internal pass reductions still
apply and remain measurable.

`shadertoy sweep --set gain=0.8,1.0,1.2` renders parameter variants and a contact
sheet without temporary project copies. `profile` reports mean, median, p95, min, and max per-pass GPU timestamp
statistics plus a total GPU-work duration computed from those same attributed pass
intervals. Completion-synchronized boundaries keep deferred compute from migrating into
a consumer; this pass-sum definition is used because portable whole-frame timer queries
can undercount asynchronous compute. `--sync-per-pass` additionally completes each query
boundary before continuing for maximum-isolation driver diagnostics. Blind sweeps anonymize parameter variants,
while `blind create` accepts existing images/render directories, project
directories, STTF builds, Git revisions, and `project:PATH@preset=NAME` sources.
`blind judge` records the preference and rationale before `blind reveal` exposes
the sealed mapping and writes a combined report. `experiment` turns the same
source forms into a reproducible baseline/candidate or N-way run with
deterministic frame sets, normalized RMSE, windowed SSIM, contact sheets,
provenance, project/git GPU profiles, and optional sealed blind labels.

`graph` renders the resolved pass/resource graph as text/JSON or Graphviz DOT.
`check --pedantic` additionally treats advisory graph/resource diagnostics as
failures, including unreachable passes, unused assets/uniforms, viewport-sized
feedback, and unordered shared SSBO users.

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
`assert_resolution_independent = true`. Tests can also compare
`uniforms` against `reference_uniforms` with minimum/maximum RMSE bounds,
enforce total/per-pass GPU budgets, compare exact SSBO binary fixtures, and set
`assert_state_roundtrip = true` to verify persistent pass/SSBO state survives
serialization into a fresh runtime. Use `shadertoy test --ci` for a non-mutating
automation path.

ShaderToy Sound passes (`kind = "sound"`) are compiled by `check` and render to
deterministic stereo PCM WAV with `shadertoy render-audio`. File-backed video
assets update at deterministic `iTime` and expose `iChannelTime`; webcam channels
are available only in live preview and are intentionally rejected by headless
render/replay/recording.


Trace bundles
-------------

`shadertoy trace capture -o target/bug.sttrace` freezes a deterministic render
into a self-contained directory containing the STTF, final PNG, persistent
`.ststate`, resolved configuration, source/asset hashes, graph/synchronization
metadata, GPU pass timings, and optional per-pass PNG/RGBA32F readbacks.
`trace inspect` verifies artifact SHA-256 values, while `trace replay` renders
only from the bundled STTF and requires a bit-exact RGB match.
