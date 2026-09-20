Buffer and compute passes are persistent 2D offscreen render targets. Compute
passes additionally expose the target as the writable image2D iOutput.

Current-frame dependency:

  source = "buffer-a"
  frame = "current"

Previous-frame feedback:

  source = "buffer-a"
  frame = "previous"

Previous-frame semantics are represented at the project level. Callers never
need to construct the renderer's internal LastFrame nodes.

For debugging you can capture and resume lossless RGBA32F snapshots of persistent 2D pass state:

  shadertoy state capture --frame 300 -o target/debug.ststate
  shadertoy render --state target/debug.ststate -o target/resumed.png

Or replace one persistent 2D pass with an exact RGBA image before the next rendered frame:

  shadertoy render --state target/debug.ststate \
    --set-buffer buffer-a=fixtures/known.png


Inspecting float data
---------------------

Inspect the actual floating-point contents (read back as RGBA32F) rather than the clamped display image:

  shadertoy inspect buffer spectrum --frame 120 --pixel 10,12

The command reports per-channel min/max/mean and NaN/Inf counts. Use
--visualization signed, rgb, or magnitude with --output to write a diagnostic
PNG while preserving the raw statistics in JSON output.
