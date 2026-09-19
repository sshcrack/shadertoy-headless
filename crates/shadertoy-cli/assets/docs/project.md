A ShaderToy CLI project is a directory containing ShaderToy.toml.

Typical layout:

  ShaderToy.toml
  shaders/
    image.frag
    buffer-a.frag
  assets/
  target/        # generated and gitignored

The manifest describes passes and iChannel wiring. Shader source stays in separate
files so agents, editors, diffs, and hot reload can operate on it directly.

Create examples with:

  shadertoy new hello
  shadertoy new feedback --template multipass
