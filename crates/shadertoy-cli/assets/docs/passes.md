Pass kinds:

  image    Exactly one final Image pass is required.
  buffer   Offscreen 2D pass. May be consumed by later passes and/or itself.
  cubemap  Cubemap pass.

Each pass points at one GLSL file with a ShaderToy-style mainImage/mainCubemap
entry point.

Inspect one pass:

  shadertoy inspect pass buffer-a
  shadertoy render --pass buffer-a -o target/buffer-a.png


Buffer passes may optionally pin their render target to a fixed resolution:

  [[pass]]
  name = "spectrum"
  kind = "buffer"
  source = "shaders/spectrum.frag"
  width = 256
  height = 256

Both width and height must be specified together. Without them, the buffer keeps
the normal output-sized behavior. For fixed buffers, iResolution uses the fixed
size and iChannelResolution reports the actual dimensions of every input.
Changing the preview or output resolution does not resize or clear fixed buffer
feedback.


Each pass input has independent sampler state. For an FFT/simulation buffer,
bind the fixed grid with exact texel sampling:

  [[pass.input]]
  channel = 0
  source = "spectrum"
  filter = "nearest"
  wrap = "repeat"

Available filters are nearest, linear, and mipmap; wrap modes are clamp and
repeat. The settings are applied independently per iChannel, even if two
channels reference the same source.
