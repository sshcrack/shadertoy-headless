# {{name}}

{{template_note}}

## Agent / CLI workflow

```sh
shadertoy inspect --json
shadertoy check --json
shadertoy render -o target/check.png
shadertoy preview
```

For the concise agent guide:

```sh
shadertoy docs agent
```

For project and channel semantics:

```sh
shadertoy docs project
shadertoy docs buffers
shadertoy docs channels
```

For the exact manifest schema:

```sh
shadertoy docs manifest --schema
```

`ShaderToy.toml` is associated with `.shadertoy/shadertoy.schema.json`. Taplo / Even Better TOML can use that schema for validation and completion.

Generated build/render outputs belong under `target/` and are ignored by Git.
