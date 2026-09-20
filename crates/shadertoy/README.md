# shadertoy-native

Safe Rust bindings for the native ShaderToy renderer in this repository. The crates.io package is `shadertoy-native`, while the Rust library target remains `shadertoy`.

The crate wraps `shadertoy-sys` and exposes native project construction, display-less/offscreen OpenGL context creation, deterministic stepping, per-pass rendering, lossless feedback-state snapshot/restore, and input injection without exposing raw C pointers.
 It also exposes compute passes, typed floating-point render targets, multiple render targets, repeated compute dispatch, and persistent named shader-storage buffers for OpenGL 4.3-capable pipelines.

For the directory-project format and agent-oriented workflow, see the repository README and the `shadertoy-cli` crate.

The underlying C++ renderer was originally written by Yingwei Zheng ([dtcxzyw/shadertoy](https://github.com/dtcxzyw/shadertoy)), whose groundwork this crate builds on.

```rust,no_run
use shadertoy::{HeadlessContext, PassKind, Project, Runtime};

let context = HeadlessContext::new(640, 360)?;
let mut project = Project::new("demo")?;
project.add_pass(
    "image",
    PassKind::Image,
    r#"
        void mainImage(out vec4 color, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            color = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
        }
    "#,
)?;

let mut runtime = Runtime::new(&context)?;
runtime.load_project(&project)?;
runtime.tick_fixed(1.0 / 60.0, 60.0)?;
let image = runtime.render(640, 360)?;
assert_eq!(image.pixels.len(), 640 * 360 * 3);

# Ok::<(), shadertoy::Error>(())
```
