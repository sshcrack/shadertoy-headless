shadertoy preview runs the native C++ renderer behind a small local web server.

The browser is a viewer/controller; it does not execute a second WebGL renderer.
Native PNG frames are pushed as binary WebSocket messages and drawn to a canvas;
there is no per-frame HTTP image polling. ShaderToy.toml, shader sources, and
assets are watched for changes.

Shader/source edits use an include-aware dependency graph. A successful source
reload recompiles only affected passes and leaves existing render targets and
feedback history intact. Manifest/asset changes take the conservative full
project reload path. Any failed reload keeps the last successful render alive
and exposes the compile/load error.

The preview supports final Image and intermediate 2D buffer views, pause/reset,
frame stepping, time scale, resolution changes, mouse/keyboard forwarding, and
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
