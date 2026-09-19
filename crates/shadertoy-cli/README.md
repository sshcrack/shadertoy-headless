# shadertoy-cli

Agent-friendly ShaderToy project, rendering, debugging, and live-preview CLI.

The installed binary is named `shadertoy`.

```bash
cargo binstall shadertoy-cli
shadertoy new demo --template multipass
cd demo
shadertoy check
shadertoy render -o target/frame.png
shadertoy render-frames --frames 0,60,120,180 --contact-sheet target/contact.png
shadertoy preview
```

The CLI embeds its project templates, JSON Schema, agent documentation, preview web UI, and Camoufox import helper, so the installed executable does not need adjacent data files.

Import a public ShaderToy into a fully local editable project:

    shadertoy import https://www.shadertoy.com/view/lsX3W4 -o mandelbrot
    cd mandelbrot
    shadertoy check

Import uses a Camoufox browser session to handle ShaderToy's browser/Cloudflare path. On first use it creates a private cached Python environment, installs the pinned Camoufox adapter, and fetches its browser. Python 3.10+ is required. On Linux, the Camoufox browser also needs the usual Firefox GTK runtime (for Debian/Ubuntu, `libgtk-3-0` or its distro equivalent). Supported textures, cubemaps, and volumes are downloaded into the project; the original ShaderToy response is retained under .shadertoy/import-response.json.

See the repository README for project format, state/debugging workflows, and the underlying C++/Rust library architecture.

The underlying C++ renderer was originally written by Yingwei Zheng ([dtcxzyw/shadertoy](https://github.com/dtcxzyw/shadertoy)), whose groundwork this CLI builds on.

On Linux, `check`, `render`, `render-frames`, state capture, and native preview use a surfaceless EGL context and do not need `DISPLAY` or `WAYLAND_DISPLAY`. `render-frames` reuses one deterministic runtime across all requested frames and can emit a contact sheet for visual iteration.
