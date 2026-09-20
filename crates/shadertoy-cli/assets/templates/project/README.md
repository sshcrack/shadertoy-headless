# {{name}}

{{template_note}}

## Agent / CLI workflow

```sh
shadertoy inspect --json
shadertoy check --json
shadertoy render -o target/check.png
shadertoy render-frames --range 0:120:30 --contact-sheet target/contact.png
shadertoy preview
```

For the concise agent guide:

```sh
shadertoy docs agent
```

For project and runtime semantics:

```sh
shadertoy docs project
shadertoy docs passes
shadertoy docs buffers
shadertoy docs channels
shadertoy docs state
shadertoy docs preview
```

For the exact manifest schema:

```sh
shadertoy docs manifest --schema
```

`ShaderToy.toml` is associated with `.shadertoy/shadertoy.schema.json`. Taplo / Even Better TOML can use that schema for validation and completion.

Generated build/render outputs belong under `target/` and are ignored by Git.
