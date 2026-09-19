# Releasing the Rust crates and CLI

The Rust workspace publishes three crates in dependency order:

1. `shadertoy-sys` — raw generated bindings plus the packaged native C/C++ source.
2. `shadertoy-native` — safe Rust wrapper (library target `shadertoy`).
3. `shadertoy-cli` — installs the `shadertoy` executable.

All three inherit one version from `[workspace.package]` in the repository root `Cargo.toml`.

`2.0.0` is the first Rust-crate publication, but it intentionally follows the existing ShaderToy project's historical `v0.1.x` and `v1.x` C++ release tags. Do not reuse those historical tag names for the Rust/CLI release line; the breaking library/editor split starts at project version `2.0.0`.

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

## vcpkg binary cache

GitHub Actions shares vcpkg build artifacts through `https://nuget.sshcrack.me/v3/index.json`. The local composite action `.github/actions/configure-vcpkg-cache` configures the feed for every workflow that can build the native C++ dependencies.

When the repository secret `NUGET_API_KEY` is available, the cache is `readwrite` and newly built ABI packages are uploaded. When GitHub withholds the secret, such as for an untrusted fork pull request, the same feed is configured read-only so existing packages can still be restored. On Unix, the cache action runs a `NuGet.exe` client through Mono when vcpkg returns one, but executes native/wrapper NuGet clients directly.

The same composite action also points `VCPKG_DOWNLOADS` at a GitHub Actions cache. That cache stores vcpkg's downloaded source archives and tools, while the NuGet feed remains the authoritative cache for compiled vcpkg ABI packages.

## CI dependency caches

The workflows cache repeatable dependency/download work at the package-manager boundary rather than preserving whole build directories:

- Linux APT packages are restored with `cache-apt-pkgs-action`;
- Cargo registry/git data and reusable `target` artifacts are restored with `rust-cache`;
- vcpkg compiled ABI packages use the shared NuGet feed;
- vcpkg source/tool downloads use `VCPKG_DOWNLOADS` plus the GitHub Actions cache;
- macOS Homebrew bottle downloads for CMake, Ninja, and Mono use a formula-version-derived GitHub Actions cache.

CMake build directories and installed system/Homebrew trees are intentionally not cached. They contain machine- and configuration-specific generated state; rebuilding those from cached dependencies is safer than restoring stale configured build trees.

## Normal release flow

1. Update the release version in the root `Cargo.toml`, `CMakeLists.txt`, `vcpkg.json`, and `shadertoy/Config.hpp`. `scripts/release/version.py` rejects releases when these surfaces disagree.
2. Update `CHANGELOG.md` and run `cargo check` so `Cargo.lock` is current.
3. Commit the release.
4. Create and push an exact matching tag such as `v2.0.1`.
5. Wait for `.github/workflows/release-cli.yml` to finish. It creates the GitHub Release and uploads prebuilt CLI archives plus SHA-256 files.
6. Run the `publish-crates` workflow manually with that same tag. The protected job obtains a short-lived crates.io token through OIDC and publishes missing workspace crates in dependency order.
7. Verify binary installation from the official release:

```bash
cargo binstall shadertoy-cli@2.0.1 --no-confirm
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
- `x86_64-apple-darwin`;
- `aarch64-apple-darwin`;
- `x86_64-pc-windows-msvc`.

(Linux ARM64 was dropped: the `ubuntu-24.04-arm` partner image ships no
usable vcpkg installation.)

The native ShaderToy core and vcpkg dependencies are statically linked into the CLI. Normal operating-system graphics/runtime libraries remain dynamic where appropriate.

## Package-layout invariant

`shadertoy-sys` is the root Cargo package even though its Rust source lives under `crates/shadertoy-sys/`. This is deliberate: crates.io only packages files beneath a package root, and the sys crate must contain the real top-level native C++ and C-ABI sources without maintaining a duplicated vendored copy.

`cargo package -p shadertoy-sys` is therefore a release gate. It must build successfully from the extracted `.crate` archive.
