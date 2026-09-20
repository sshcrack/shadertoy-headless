Every pass may bind up to four inputs: iChannel0 through iChannel3.

An input selects:
  channel = 0..3
  source  = another pass, an asset, "keyboard", or "music"
  frame   = "current" or "previous" (pass inputs only)
  filter  = "nearest", "linear", or "mipmap"
  wrap    = "clamp" or "repeat"

Use:

  shadertoy inspect channels image --json

to see exactly what a pass consumes.

Sampler settings are per input, including when the same source is connected to
multiple channels with different sampling behavior:

  [[pass.input]]
  channel = 0
  source = "spectrum"
  filter = "nearest"
  wrap = "repeat"

nearest is the right choice for exact numerical grids such as FFT stages.
linear enables interpolated sampling, while mipmap uses trilinear
minification and linear magnification. clamp maps out-of-range coordinates to
the edge texel; repeat wraps them. Omitting these fields keeps the existing
linear/repeat defaults.

The same controls are available when editing a project from the CLI:

  shadertoy channel set image 0 spectrum --filter nearest --wrap repeat
