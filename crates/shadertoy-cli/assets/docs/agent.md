ShaderToy CLI agent workflow

To start from an existing public ShaderToy:

     shadertoy import https://www.shadertoy.com/view/XXXXXX -o project

This creates the same editable directory format used by shadertoy new and keeps
supported remote assets local.

1. Start by inspecting the project:
     shadertoy inspect --json

2. Edit ShaderToy.toml and files under shaders/ or assets/.
   For GPU simulations, FFTs, particles, and large structured state, prefer
   compute passes + typed targets + named SSBOs; see shadertoy docs passes.
   The manifest is schema-backed. To print the exact schema:
     shadertoy docs manifest --schema

3. Validate after edits:
     shadertoy check --json

4. Render deterministic evidence:
     shadertoy render -o target/check.png
   For temporal comparison/contact sheets, reuse one runtime:
     shadertoy render-frames --frames 0,60,120,180 --contact-sheet target/contact.png

5. Debug multipass projects from the outside in:
     shadertoy inspect graph --json
     shadertoy inspect pass buffer-a --json
     shadertoy inspect buffer buffer-a --frame 120 --pixel 8,8 --json
     shadertoy render --pass buffer-a -o target/buffer-a.png

6. Freeze a feedback state when a bug appears:
     shadertoy state capture --frame 300 -o target/frame300.ststate
     shadertoy state inspect target/frame300.ststate --json

7. Replace a buffer with a known image to isolate a pass:
     shadertoy render --state target/frame300.ststate \
       --set-buffer buffer-a=fixtures/a.png \
       -o target/debug.png

8. Profile expensive passes on the real GPU path:
     shadertoy profile --frame 120 --warmup 5 --samples 30 --json

9. Define deterministic [[test]] cases in ShaderToy.toml and run:
     shadertoy test
   Use --update deliberately to write visual baselines.

10. Use live native-rendered review when iterating:
     shadertoy preview
    Shared GLSL can use quoted #include directives; preview recompiles only passes
    affected by a changed source/include and keeps existing feedback targets alive.

11. Record an input-sensitive preview bug, then reproduce it without the browser:
     shadertoy preview --record target/repro.strec
     shadertoy replay target/repro.strec -o target/replayed.png

On Linux, check/render/render-frames/state capture/preview/profile/test/replay use surfaceless EGL and do not require DISPLAY or WAYLAND_DISPLAY. The context prefers OpenGL 4.3 and falls back to 4.1 for projects that do not use compute/SSBO features.

Do not edit target/. It is disposable generated output.
