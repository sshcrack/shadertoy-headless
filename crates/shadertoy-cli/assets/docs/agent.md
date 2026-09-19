ShaderToy CLI agent workflow

1. Start by inspecting the project:
     shadertoy inspect --json

2. Edit ShaderToy.toml and files under shaders/ or assets/.
   The manifest is schema-backed. To print the exact schema:
     shadertoy docs manifest --schema

3. Validate after edits:
     shadertoy check --json

4. Render deterministic evidence:
     shadertoy render -o target/check.png

5. Debug multipass projects from the outside in:
     shadertoy inspect graph --json
     shadertoy inspect pass buffer-a --json
     shadertoy render --pass buffer-a -o target/buffer-a.png

6. Freeze a feedback state when a bug appears:
     shadertoy state capture --frame 300 -o target/frame300.ststate
     shadertoy state inspect target/frame300.ststate --json

7. Replace a buffer with a known image to isolate a pass:
     shadertoy render --state target/frame300.ststate \
       --set-buffer buffer-a=fixtures/a.png \
       -o target/debug.png

8. Use live native-rendered review when iterating:
     shadertoy preview

Do not edit target/. It is disposable generated output.
