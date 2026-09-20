Project assets are declared with [[asset]] and referenced by name from pass inputs.

Example:

  [[asset]]
  name = "noise"
  kind = "texture"
  path = "assets/noise.png"

  [[asset]]
  name = "sky"
  kind = "cubemap"
  path = "assets/sky-strip.png"

  [[asset]]
  name = "density"
  kind = "volume"
  path = "assets/density.bin"

Supported asset kinds are texture, cubemap, volume, and video.

Cubemap files
-------------

A cubemap asset is one ordinary image containing six square faces in one
horizontal strip. If each face is N x N, the source image must be exactly
(6*N) x N. Faces are consumed left-to-right in strip order.

For example, a 256-pixel cubemap uses a 1536x256 image.

The CLI converts the strip into the renderer's six cubemap faces when the
project is loaded. A non-strip image is rejected with its actual and expected
dimensions.

Volume files
------------

Volume assets use ShaderToy's compact binary volume layout. The file is:

  bytes  0..3   ignored u32, little-endian
  bytes  4..7   X dimension, u32 little-endian
  bytes  8..11  Y dimension, u32 little-endian
  bytes 12..15  Z dimension, u32 little-endian
  bytes 16..19  metadata, u32 little-endian
  bytes 20..    tightly packed voxel bytes

The current CLI accepts only positive cubic volumes, so X == Y == Z > 0.

metadata is packed as:

  bits  0..7    channel count
  bits  8..15   layout
  bits 16..31   format

Supported values are:
  channel count = 1 or 4
  layout        = 0
  format        = 0

The payload size must be exactly X*Y*Z*channels bytes. Samples are unsigned
8-bit normalized values. Volume channels are exposed to GLSL as sampler3D.

Texture/cubemap decoding uses the image formats supported by the embedded image
decoder. Video assets are file-backed media decoded by ffmpeg/ffprobe for
headless rendering.

For input wiring and sampler controls, see:

  shadertoy docs channels
