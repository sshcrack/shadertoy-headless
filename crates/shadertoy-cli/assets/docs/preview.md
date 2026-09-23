shadertoy preview runs the native C++ renderer behind a small local web server.

The browser is a viewer/controller; it does not execute a second shader renderer.
Native RGB frames are handed to a bounded latest-frame transport worker. Choose
the transport with --preview-transport auto|raw|mjpeg|png. The default auto mode
uses raw RGB when preview is bound to localhost/127.0.0.1/::1, avoiding image
encode/decode entirely for local review. Any externally reachable bind, including
0.0.0.0 and ::, defaults to MJPEG so a proxied or LAN preview does not accidentally
send roughly 3 bytes per pixel per frame. Use --preview-transport raw to force the
uncompressed fast path, --preview-transport png for lossless compressed frames,
or --preview-transport mjpeg to force the low-bandwidth live path.

MJPEG keeps one persistent low-latency ffmpeg encoder alive and pushes independent
JPEG frames as binary WebSocket messages; if ffmpeg is unavailable, an in-process
JPEG encoder is used as a compatibility fallback. PNG uses a low-compression
preview encoder off the graphics thread. Raw frames carry an 8-byte little-endian
width/height header followed by RGB24 pixels and are uploaded directly to a WebGL2
texture in the browser. Compressed frames use native browser image decode and
BitmapRenderer when available. There is no per-frame HTTP polling and a slow
encoder/browser cannot build an unbounded stale-frame queue. Set SHADERTOY_FFMPEG
to override the ffmpeg executable. The preview reports a rolling end-to-end actual
FPS based on frames successfully drawn by the browser, alongside the configured
target FPS. ShaderToy.toml, shader sources, and assets are watched for changes.
shadertoy preview --preset NAME applies a named manifest quality preset and keeps
that preset active across full hot reloads. When presets exist, the browser exposes
a Quality preset selector (including the base manifest) and can switch tiers live
without restarting preview.

Preview is deliberately a human-review surface. Its final Image output stays at
the base project's render width/height while switching presets, even when a
preset declares render_scale. This prevents a lower-resolution final image from
looking like a shader-performance improvement. Preset pass overrides (fixed
buffer/compute dimensions, iterations, local size) still apply normally.

The browser exposes common review resolutions plus custom width/height. A
manually selected review resolution stays fixed while switching presets. Choose
Project output to return to the base manifest's render dimensions.

Shader/source edits use an include-aware dependency graph. A successful source
reload recompiles only affected passes and leaves existing render targets and
feedback history intact. Manifest/asset changes take the conservative full
project reload path. Any failed reload keeps the last successful render alive
and exposes the compile/load error.

The preview supports final Image and intermediate 2D buffer views, pause/reset,
frame stepping, time scale, explicit review-resolution changes,
mouse/keyboard forwarding, and
live controls for declared custom uniforms. File-backed video channels are
updated from shader time. If the manifest declares a `kind = "webcam"` channel,
the browser exposes a Start webcam control and forwards 320x240 RGBA frames to
the native renderer.

On Linux the native preview uses the same surfaceless EGL path as check/render, so it can run without DISPLAY or WAYLAND_DISPLAY.


Input recording and replay
--------------------------

Live webcam input cannot be recorded because camera frames are not reproducible;
use a local video asset when a media-dependent bug needs deterministic replay.

Record shader-affecting preview controls together with exact rendered
iFrame/iTime/iTimeDelta/iFrameRate markers:

  shadertoy preview --record target/repro.strec

Reproduce the recorded timeline headlessly:

  shadertoy replay target/repro.strec -o target/replay.png

Recordings contain a SHA-256 fingerprint of the manifest, expanded shader
sources, and assets. Replay refuses a changed project by default; use
--allow-project-changes only when comparing behavior intentionally. Any watched
project-file edit while recording invalidates that recording, because a single
replay cannot reproduce multiple source revisions; restart preview recording
after the edit settles.
