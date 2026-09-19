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

  if ! cargo publish --locked -p "$crate"; then
    # A rerun can race crates.io index propagation after a previous upload.
    # Treat a failed publish as successful only if this exact version becomes
    # visible during the normal propagation window; otherwise preserve failure.
    echo "publish command for ${crate}@${version} failed; checking whether the version is already propagating" >&2
    if wait_until_visible "$crate"; then
      echo "${crate}@${version} became visible after the failed publish; continuing"
      continue
    fi
    exit 1
  fi
  wait_until_visible "$crate"
done
