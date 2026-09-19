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
