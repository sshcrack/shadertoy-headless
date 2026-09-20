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

For CLI-created projects, the generated .shadertoy/shadertoy.schema.json is
refreshed automatically on project load when a newer installed CLI embeds a
different schema.

Create examples with:

  shadertoy new hello
  shadertoy new feedback --template multipass


Import a public ShaderToy into the same directory format with:

  shadertoy import https://www.shadertoy.com/view/XXXXXX
  shadertoy docs import

Imported projects keep their source provenance under the project section and preserve
the original browser response under .shadertoy/import-response.json.

STTF export limits
------------------

`shadertoy build` serializes the native static render graph. Directory projects
that contain Sound passes or dynamic video/webcam channels must stay in directory
form; build rejects them rather than silently dropping Sound or freezing media to
one frame.
