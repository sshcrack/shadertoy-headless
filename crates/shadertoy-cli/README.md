# shadertoy-cli

Agent-friendly ShaderToy project, rendering, debugging, and live-preview CLI.

The installed binary is named `shadertoy`.

```bash
cargo binstall shadertoy-cli
shadertoy new demo --template multipass
cd demo
shadertoy check
shadertoy render -o target/frame.png
shadertoy preview
```

The CLI embeds its project templates, JSON Schema, agent documentation, and preview web UI, so the installed executable does not need adjacent data files.

See the repository README for project format, state/debugging workflows, and the underlying C++/Rust library architecture.
