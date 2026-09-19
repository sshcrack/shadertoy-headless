# Releasing the Rust crates and CLI

The Rust workspace publishes three crates in dependency order:

1. `shadertoy-sys` — raw generated bindings plus the packaged native C/C++ source.
2. `shadertoy-native` — safe Rust wrapper (library target `shadertoy`).
3. `shadertoy-cli` — installs the `shadertoy` executable.

All three inherit one version from `[workspace.package]` in the repository root `Cargo.toml`.

The cargo-binstall URL is derived from that same workspace `repository` value. The checked-in metadata targets `sshcrack/shadertoy`, the release repository for these crates and binaries. The release and crates-publish workflows deliberately fail when `GITHUB_REPOSITORY` does not match this metadata, because publishing a tag on a fork while leaving the upstream URL would make cargo-binstall look for binaries in the wrong repository. If the release host changes later, update `repository` (and `homepage`) before publishing a new crates.io version.

## Before the first crates.io release

Crates.io requires the first version of a crate to be published before a Trusted Publisher can be registered for it. The first release is therefore intentionally manual.

Verify the release locally or in CI first:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo package -p shadertoy-sys --locked
```

Make sure the crate names are still available on crates.io. Then authenticate locally with a scoped crates.io token and publish in dependency order:

```bash
cargo publish --locked -p shadertoy-sys
# Wait until shadertoy-sys@VERSION is visible in the registry.
cargo publish --locked -p shadertoy-native
# Wait until shadertoy-native@VERSION is visible in the registry.
cargo publish --locked -p shadertoy-cli
```

The helper `scripts/release/publish_crates.sh` performs the same ordered publishing and waits for registry visibility, but it is primarily intended for the Trusted Publishing workflow after the first release.

## Configure crates.io Trusted Publishing

After the first version exists, configure the same GitHub Actions Trusted Publisher for each of the three crates:

- repository owner: the GitHub owner of this repository;
- repository: `shadertoy`;
- workflow filename: `publish-crates.yml`;
- environment: `release`.

Create a GitHub Actions environment named `release`. Requiring manual approval for that environment is recommended.

Once the OIDC workflow has been proven, crates.io can optionally be switched to Trusted-Publishing-only mode for each crate so long-lived API tokens can no longer publish them.

## Normal release flow

1. Update `[workspace.package].version` in the root `Cargo.toml`.
2. Update `CHANGELOG.md` and run `cargo check` so `Cargo.lock` is current.
3. Commit the release.
4. Create and push an exact matching tag such as `v0.1.1`.
5. Wait for `.github/workflows/release-cli.yml` to finish. It creates the GitHub Release and uploads prebuilt CLI archives plus SHA-256 files.
6. Run the `publish-crates` workflow manually with that same tag. The protected job obtains a short-lived crates.io token through OIDC and publishes missing workspace crates in dependency order.
7. Verify binary installation from the official release:

```bash
cargo binstall shadertoy-cli@0.1.1 --no-confirm
shadertoy --version
```

Publishing the GitHub binary release before the crate is useful: as soon as `shadertoy-cli` appears on crates.io, cargo-binstall can resolve its official prebuilt artifact immediately.

## Binary artifact contract

`shadertoy-cli` declares explicit cargo-binstall metadata. Release artifacts are named:

```text
shadertoy-cli-<rust-target>-v<version>.tgz
shadertoy-cli-<rust-target>-v<version>.zip   # Windows
```

The archive contains only the executable at its root:

```text
shadertoy
# or
shadertoy.exe
```

The release workflow currently builds:

- `x86_64-unknown-linux-gnu`;
- `aarch64-unknown-linux-gnu`;
- `x86_64-apple-darwin`;
- `aarch64-apple-darwin`;
- `x86_64-pc-windows-msvc`.

The native ShaderToy core and vcpkg dependencies are statically linked into the CLI. Normal operating-system graphics/runtime libraries remain dynamic where appropriate.

## Package-layout invariant

`shadertoy-sys` is the root Cargo package even though its Rust source lives under `crates/shadertoy-sys/`. This is deliberate: crates.io only packages files beneath a package root, and the sys crate must contain the real top-level native C++ and C-ABI sources without maintaining a duplicated vendored copy.

`cargo package -p shadertoy-sys` is therefore a release gate. It must build successfully from the extracted `.crate` archive.
