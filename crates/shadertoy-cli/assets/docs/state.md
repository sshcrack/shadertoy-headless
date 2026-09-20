.ststate is ShaderToy CLI's resumable debugging artifact.

It stores:
  - render dimensions
  - fixed frame rate
  - shader time and iFrame
  - lossless RGBA32F contents of persistent 2D buffer passes
  - each captured buffer's actual dimensions (including fixed-size passes)

The state is deterministic and independent of wall-clock playback.

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


State format 2 stores per-buffer dimensions. Format 1 state files remain
readable. A state containing only fixed-size buffers can be resumed at a
different output resolution; output-sized feedback buffers still require the
captured resolution so their state is not silently discarded.
