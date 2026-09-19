# Agent development guide

## Source layout

Keep implementation files focused and easy to inspect.

- Prefer source files below roughly 500 lines.
- Consider splitting a file once it grows past roughly 700 lines.
- Files above 1,000 lines should be exceptional.
- Prefer a small module tree in a named folder over one large catch-all source file.
- Vendored third-party code is not subject to these limits.
- Avoid growing legacy large files when a focused new module is practical.

For the Rust CLI, group substantial features under folders such as `src/ops/`,
`src/preview/`, and `src/cli/`.

## Embedded artifacts

Do not put substantial HTML, Markdown, shader templates, schemas, or other static
payloads inline in Rust source.

CLI artifacts belong under:

`crates/shadertoy-cli/assets/`

Load text artifacts through the crate's `include_file!` macro. Add a focused
subfolder such as `docs/`, `preview/`, `schema/`, or `templates/` rather
than flattening unrelated assets together.

## Validation

Before handing off CLI/native changes, run the relevant subset of:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

cmake --preset library
cmake --build --preset library
ctest --preset library

cmake --preset c-api
cmake --build --preset c-api
ctest --test-dir build-c-api --output-on-failure
```

Rendering and preview behavior should also be exercised with a hardware-backed
OpenGL display when those paths changed.
