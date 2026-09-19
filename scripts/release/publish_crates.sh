#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

version="$(python3 scripts/release/version.py --print)"
crates=(shadertoy-sys shadertoy-native shadertoy-cli)

wait_until_visible() {
  local crate="$1"
  local attempt
  for attempt in $(seq 1 36); do
    if cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "${crate}@${version} did not become visible on crates.io within 3 minutes" >&2
  return 1
}

for crate in "${crates[@]}"; do
  if cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1; then
    echo "${crate}@${version} is already published; skipping"
    continue
  fi

  cargo publish --locked -p "$crate"
  wait_until_visible "$crate"
done
