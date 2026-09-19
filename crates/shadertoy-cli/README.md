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

The CLI embeds its project templates, JSON Schema, agent documentation, and preview web UI, so the installed executable does not need adjacent data files.

See the repository README for project format, state/debugging workflows, and the underlying C++/Rust library architecture.

The underlying C++ renderer was originally written by Yingwei Zheng ([dtcxzyw/shadertoy](https://github.com/dtcxzyw/shadertoy)), whose groundwork this CLI builds on.

On Linux, `check`, `render`, `render-frames`, state capture, and native preview use a surfaceless EGL context and do not need `DISPLAY` or `WAYLAND_DISPLAY`. `render-frames` reuses one deterministic runtime across all requested frames and can emit a contact sheet for visual iteration.
