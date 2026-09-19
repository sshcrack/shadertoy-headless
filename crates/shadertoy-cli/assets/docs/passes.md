Pass kinds:

  image    Exactly one final Image pass is required.
  buffer   Offscreen 2D pass. May be consumed by later passes and/or itself.
  cubemap  Cubemap pass.

Each pass points at one GLSL file with a ShaderToy-style mainImage/mainCubemap
entry point.

Inspect one pass:

  shadertoy inspect pass buffer-a
  shadertoy render --pass buffer-a -o target/buffer-a.png
