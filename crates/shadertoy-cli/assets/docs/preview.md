shadertoy preview runs the native C++ renderer behind a small local web server.

The browser is a viewer/controller; it does not execute a second WebGL renderer.
Native PNG frames are pushed as binary WebSocket messages and drawn to a canvas;
there is no per-frame HTTP image polling. ShaderToy.toml, shader sources, and
assets are watched for changes.

A successful reload replaces the current native project. A failed reload keeps
the last successful render alive and exposes the compile/load error.

The preview supports final Image and intermediate 2D buffer views, pause/reset,
frame stepping, time scale, resolution changes, and mouse forwarding.

On Linux the native preview uses the same surfaceless EGL path as check/render, so it can run without DISPLAY or WAYLAND_DISPLAY.
