Pass kinds:

  image    Exactly one final Image pass is required.
  buffer   Offscreen 2D fragment pass. May be consumed by later passes and/or itself.
  cubemap  Cubemap fragment pass.
  compute  Fixed-size OpenGL 4.3 compute pass with a writable image output.

Fragment passes use ShaderToy-style mainImage/mainCubemap entry points. Compute
passes implement:

  void mainCompute(ivec2 coord)

The host provides iResolution, iTime, iTimeDelta, iFrameRate, iFrame, iMouse,
iDate, iChannelResolution, configured iChannel samplers, iIteration, and a
writable image2D named iOutput. The wrapper bounds-checks global invocation
coordinates, so local workgroups do not need to divide the pass dimensions.

Inspect one pass:

  shadertoy inspect pass buffer-a
  shadertoy render --pass buffer-a -o target/buffer-a.png

Buffer passes may optionally pin their render target to a fixed resolution.
Compute passes always require a fixed resolution:

  [[pass]]
  name = "spectrum"
  kind = "compute"
  source = "shaders/spectrum.comp"
  width = 256
  height = 256
  format = "rg32f"
  local_size = [8, 8, 1]
  iterations = 8

Available 2D target formats are r32f, rg32f, rgba16f, and rgba32f. Buffer
passes default to rgba32f and may use the same format field.

For fixed passes, iResolution uses the pass dimensions and
iChannelResolution reports the actual dimensions of every input. Changing
preview/output resolution does not resize or clear fixed pass state.


Multiple render targets
-----------------------

Buffer and compute passes can expose up to eight 2D outputs. format describes
output 0; extra_outputs appends outputs 1 through 7:

  [[pass]]
  name = "gbuffer"
  kind = "buffer"
  source = "shaders/gbuffer.frag"
  format = "rgba16f"
  extra_outputs = ["rg32f", "rgba16f"]

A fragment pass writes output 0 through the normal mainImage result. Additional
fragment outputs use ordinary GLSL locations:

  layout(location = 1) out vec4 normalVelocity;
  layout(location = 2) out vec4 materialData;

Compute passes receive image2D uniforms iOutput, iOutput1, iOutput2, and so on,
with the declared format of each target. This makes one dispatch suitable for
producing related simulation fields without repeating the work.

A downstream channel selects an attachment with output:

  [[pass.input]]
  channel = 0
  source = "gbuffer"
  output = 1
  filter = "nearest"
  wrap = "clamp"

Output 0 remains the default when output is omitted. Nonzero outputs are
current-frame dependencies in 2.2.0; previous-frame feedback is intentionally
limited to output 0 so .ststate capture/restore remains complete and
deterministic.

Compute iterations
------------------

iterations dispatches the same compute pass multiple times in one rendered
frame. iIteration is zero-based and a memory barrier is inserted between
dispatches. The same iOutput image remains bound, so imageLoad/imageStore can
carry per-pixel state from one dispatch to the next. Cross-pixel algorithms
that require strict Jacobi-style ping-pong should use separate storage regions
or ordinary previous-frame pass feedback.

Shader storage buffers
----------------------

Compute and fragment passes may bind persistent SSBOs by name:

  [[pass.storage]]
  binding = 3
  name = "particles"
  size = 4194304

Declare the matching block in GLSL:

  layout(std430, binding = 3) buffer Particles {
      vec4 particleData[];
  };

Storage with the same name and size is shared across passes even when each pass
uses a different binding index. New storage is zero-initialized. OpenGL 4.3
image load/store, SSBO operations, and atomics are therefore available to
advanced pipelines. A pass using compute or storage buffers requires OpenGL
4.3; ordinary projects continue to work on the existing OpenGL 4.1 path.

Each pass input has independent sampler state. For an FFT/simulation grid,
bind the fixed grid with exact texel sampling:

  [[pass.input]]
  channel = 0
  source = "spectrum"
  filter = "nearest"
  wrap = "repeat"

Available filters are nearest, linear, and mipmap; wrap modes are clamp and
repeat. The settings are applied independently per iChannel, even if two
channels reference the same source.
