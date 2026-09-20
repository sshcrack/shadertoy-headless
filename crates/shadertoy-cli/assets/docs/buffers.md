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
  shadertoy inspect buffer gbuffer --output-index 1 --raw target/gbuffer1.rgba32f

The command reports per-channel min/max/mean and NaN/Inf counts. `--output-index`
selects an MRT attachment; `--raw` writes little-endian RGBA32F bytes. Use
--visualization signed, rgb, or magnitude with --output to write a diagnostic
PNG while preserving raw statistics in JSON output.

Named SSBOs can be inspected directly:

  shadertoy inspect storage particles --frame 120 --type f32 --offset 0 --count 32
  shadertoy inspect storage particles --frame 120 -o target/particles.bin

Visualization modes
-------------------

shadertoy inspect buffer ... --output diagnostic.png accepts:

  --visualization auto
      Use rgb when RGB values stay within 0..1; otherwise use signed.

  --visualization rgb
      Write RGB directly with display clamping to 0..1.

  --visualization signed
      Center zero at 0.5 and scale all RGB channels by the largest absolute
      finite RGB value in the inspected buffer.

  --visualization magnitude
      Write length(rgb) as grayscale, normalized by the largest magnitude in
      the inspected buffer.

The visualization affects only the diagnostic PNG. Raw statistics and --raw
RGBA32F output retain the original floating-point values.
