ShaderToy CLI agent workflow

To start from an existing public ShaderToy:

     shadertoy import https://www.shadertoy.com/view/XXXXXX -o project

This creates the same editable directory format used by shadertoy new and keeps
supported remote assets local.

1. Start by inspecting the project:
     shadertoy inspect --json

2. Edit ShaderToy.toml and files under shaders/ or assets/.
   Put quality tiers in `[preset.NAME]` / `[preset.NAME.pass.PASS]` and select
   them with `--preset NAME` instead of copying the project.
   For GPU simulations, FFTs, particles, and large structured state, prefer
   compute passes + typed targets + named SSBOs; see shadertoy docs passes.
   The manifest is schema-backed. To print the exact schema:
     shadertoy docs manifest --schema

3. Validate after edits:
     shadertoy check --json
   For advisory graph/resource linting too:
     shadertoy check --pedantic --json
     shadertoy graph --dot target/graph.dot --json
   Or validate an effective quality tier:
     shadertoy check --preset medium --json

4. Render deterministic evidence:
     shadertoy render -o target/check.png
   For temporal comparison/contact sheets, reuse one runtime:
     shadertoy render-frames --range 0:180:60 --contact-sheet target/contact.png
   For an encoded deterministic clip:
     shadertoy render-video --frames 180 -o target/clip.mp4
   For parameter/art-direction comparisons:
     shadertoy sweep --frame 120 --set foam_gain=0.8,1.0,1.2
   To avoid value/expectation bias, prefer a blind comparison when choosing a visual variant:
     shadertoy sweep --blind --frame 120 --set foam_gain=0.8,1.0,1.2
   To blind old-vs-new images, projects, builds, or git revisions:
     shadertoy blind create target/old-renders target/new-renders --output-dir target/old-vs-new
   To compare quality presets without temporary project copies:
     shadertoy blind create 'project:.@preset=high' 'project:.@preset=medium' 'project:.@preset=low' --frames 60,180,300
     shadertoy blind judge target/old-vs-new/blind-session.json --pick B --reason "concise visual rationale"
     shadertoy blind reveal target/old-vs-new/blind-session.json
   For a complete baseline/candidate experiment with metrics, provenance,
   profiling, contact sheets, and optional blind labels:
     shadertoy experiment --baseline git:main --candidate git:HEAD --frames 0,60,120 --blind

5. Debug multipass projects from the outside in:
     shadertoy inspect graph --json
     shadertoy inspect pass buffer-a --json
     shadertoy inspect buffer buffer-a --frame 120 --pixel 8,8 --json
     shadertoy inspect buffer gbuffer --output-index 1 --raw target/attachment.rgba32f
     shadertoy inspect storage particle-state --frame 120 --type f32 --count 16 --json
     shadertoy render --pass buffer-a -o target/buffer-a.png

6. Freeze a feedback state when a bug appears:
     shadertoy state capture --frame 300 --include-storage --set storm=1.0 -o target/frame300.ststate
     shadertoy state inspect target/frame300.ststate --json

7. Replace a buffer with a known image to isolate a pass:
     shadertoy render --state target/frame300.ststate \
       --set-buffer buffer-a=fixtures/a.png \
       -o target/debug.png

8. Profile expensive passes on the real GPU path:
     shadertoy profile --frame 120 --warmup 5 --samples 30 --json
   Reports mean, median, p95, min, and max per-pass GPU timestamp durations plus
   a total GPU-work duration and CPU render-call timing. Completion-synchronized
   boundaries keep asynchronous compute attached to the issuing pass. The total is the
   sum of those attributed intervals because portable whole-frame timer queries can
   undercount async compute on affected drivers. For maximum-isolation diagnostics,
   rerun with:
     shadertoy profile --frame 120 --samples 30 --sync-per-pass --json

9. Define deterministic [[test]] cases in ShaderToy.toml and run:
     shadertoy test --ci
   Tests can cover frame/resolution matrices, uniform-vs-uniform RMSE bounds,
   per-pass/total GPU budgets, exact SSBO fixtures, deterministic raw output,
   output-resolution independence, and .ststate round-trips. Use --update
   deliberately to write visual baselines. See shadertoy docs test.

10. Freeze difficult deterministic renderer bugs into a self-contained bundle:
      shadertoy trace capture --frame 120 --include-intermediates -o target/bug.sttrace
      shadertoy trace inspect target/bug.sttrace --json
      shadertoy trace replay target/bug.sttrace --json
    The trace carries STTF, persistent state/SSBOs, timings, hashes, graph
    diagnostics, and optional pass readbacks. See shadertoy docs trace.

11. Use live native-rendered review when iterating:
     shadertoy preview
    Or launch a named quality tier directly:
     shadertoy preview --preset medium
    When presets exist, the browser also exposes a Quality preset selector for live switching.
    Declared custom uniforms become live controls. A `kind = "webcam"` channel
    exposes a Start webcam button; webcam input is intentionally not recordable
    or usable by headless commands.
    Shared GLSL can use quoted #include directives; preview recompiles only passes
    affected by a changed source/include and keeps existing feedback targets alive.

12. Record an input-sensitive preview bug, then reproduce it without the browser:
     shadertoy preview --record target/repro.strec
     shadertoy replay target/repro.strec -o target/replayed.png

On Linux, check/render/render-frames/state capture/preview/profile/test/experiment/trace/replay use surfaceless EGL and do not require DISPLAY or WAYLAND_DISPLAY. The context prefers OpenGL 4.3 and falls back to 4.1 for projects that do not use compute/SSBO features.

Do not edit target/. It is disposable generated output.
