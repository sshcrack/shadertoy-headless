shadertoy preview runs the native C++ renderer behind a small local web server.

The browser is a viewer/controller; it does not execute a second WebGL renderer.
ShaderToy.toml, shader sources, and assets are watched for changes.

A successful reload replaces the current native project. A failed reload keeps
the last successful render alive and exposes the compile/load error.

The preview supports final Image and intermediate 2D buffer views, pause/reset,
frame stepping, time scale, resolution changes, and mouse forwarding.
