.ststate is ShaderToy CLI's resumable debugging artifact.

It stores:
  - render dimensions
  - fixed frame rate
  - shader time and iFrame
  - lossless RGBA32F snapshots of persistent 2D buffer/compute passes
  - each captured buffer's actual dimensions (including fixed-size passes)
  - each captured buffer's render-target format (format 3)

Named SSBO byte contents are not stored in .ststate. An SSBO-driven simulation
that must be resumable should mirror the required state into a captured 2D pass
texture or rebuild the SSBO deterministically after resume.

The captured texture/timing state is deterministic and independent of wall-clock playback.

Capture:
  shadertoy state capture --frame 300 -o target/frame300.ststate

Inspect:
  shadertoy state inspect target/frame300.ststate --json

Modify one buffer:
  shadertoy state set target/frame300.ststate \
    buffer-a=fixtures/known.png \
    -o target/modified.ststate

Resume:
  shadertoy render --state target/modified.ststate -o target/next.png


State format 3 stores per-buffer dimensions and render-target formats so typed
buffer state cannot be silently restored into an incompatible format. Formats 1
and 2 remain readable; their older files do not contain render-format metadata. A
state containing only fixed-size persistent passes can be resumed at a different
output resolution; output-sized feedback buffers still require the captured
resolution so their state is not silently discarded.
