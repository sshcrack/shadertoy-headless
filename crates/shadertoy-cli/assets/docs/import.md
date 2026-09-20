Import a public ShaderToy page into a fully local editable project:

  shadertoy import https://www.shadertoy.com/view/lsX3W4
  shadertoy import lsX3W4 -o my-shader

The import path uses Camoufox instead of direct HTTP so ShaderToy's browser/Cloudflare
path is handled by a real browser session. On first use the CLI creates a private
cached Python environment, installs its pinned Camoufox adapter, and fetches the
matching Camoufox browser. Python 3.10+ must be available. On Linux, the Camoufox browser also needs the usual Firefox GTK runtime (for Debian/Ubuntu, `libgtk-3-0` or its distro equivalent).

Camoufox handles ShaderToy's managed browser verification in the same browser
session used for the import and reuses a cached profile for later imports. If a
verification step still requires manual interaction, run from a graphical desktop;
the visible Camoufox window remains available for completing it.

Imported projects keep supported shader passes, Common code, pass/feedback wiring,
sampler filter/wrap settings, keyboard/audio inputs, textures, cubemaps, volumes,
file-backed video inputs, and Sound passes. Static resources are copied into
assets/ so rendering no longer depends on ShaderToy.

Project metadata records the source URL/id, author, and description. The exact
ShaderToy response used for the import is preserved at:

  .shadertoy/import-response.json

Webcam inputs are preserved as live-preview channels. Unsupported ShaderToy input
types are reported as warnings and left unbound instead of being silently replaced.

Environment overrides for advanced/packaged setups:

  SHADERTOY_CAMOUFOX_PYTHON=/path/to/python
      Use an existing Python interpreter that already has camoufox installed.

  SHADERTOY_CACHE_DIR=/path/to/cache
      Change where the CLI stores its managed Camoufox Python environment.

  SHADERTOY_CAMOUFOX_HEADLESS=virtual|true|false
      Override browser display mode. The default is a visible browser when a graphical
      display exists and Camoufox virtual-display mode otherwise.
