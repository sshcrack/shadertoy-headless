#!/usr/bin/env python3
"""Read and validate the workspace release version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import urlparse


def workspace_metadata(repo: Path) -> dict[str, object]:
    data = tomllib.loads((repo / "Cargo.toml").read_text(encoding="utf-8"))
    return data["workspace"]["package"]


def workspace_version(repo: Path) -> str:
    return str(workspace_metadata(repo)["version"])


def github_repository(repo: Path) -> str:
    repository = str(workspace_metadata(repo)["repository"])
    parsed = urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError(f"workspace repository is not a GitHub HTTPS URL: {repository!r}")
    slug = parsed.path.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    if slug.count("/") != 1 or not all(slug.split("/")):
        raise ValueError(f"workspace repository does not identify owner/repo: {repository!r}")
    return slug


def native_versions(repo: Path) -> dict[str, str]:
    cmake = (repo / "CMakeLists.txt").read_text(encoding="utf-8")
    cmake_match = re.search(r"project\(shadertoy\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", cmake)
    if cmake_match is None:
        raise ValueError("could not read ShaderToy version from CMakeLists.txt")

    vcpkg = json.loads((repo / "vcpkg.json").read_text(encoding="utf-8"))
    vcpkg_version = str(vcpkg.get("version", ""))
    if not vcpkg_version:
        raise ValueError("could not read ShaderToy version from vcpkg.json")

    config = (repo / "shadertoy" / "Config.hpp").read_text(encoding="utf-8")
    config_match = re.search(
        r"SHADERTOY_VERSION\s+SHADERTOY_MAKE_VERSION\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        config,
    )
    if config_match is None:
        raise ValueError("could not read ShaderToy version from shadertoy/Config.hpp")

    return {
        "CMakeLists.txt": cmake_match.group(1),
        "vcpkg.json": vcpkg_version,
        "shadertoy/Config.hpp": ".".join(config_match.groups()),
    }


def validate_version_surfaces(repo: Path, expected: str) -> None:
    mismatches = [
        f"{path}={version}"
        for path, version in native_versions(repo).items()
        if version != expected
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"native release versions do not match Cargo workspace version {expected}: {joined}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tag", help="Expected git tag, e.g. v2.0.0")
    parser.add_argument(
        "--github-repository",
        help="Expected GitHub owner/repo hosting release artifacts",
    )
    parser.add_argument("--print", action="store_true", dest="print_version")
    args = parser.parse_args()

    version = workspace_version(args.repo)
    try:
        validate_version_surfaces(args.repo, version)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.tag is not None and args.tag != f"v{version}":
        print(
            f"release tag {args.tag!r} does not match workspace version v{version}",
            file=sys.stderr,
        )
        return 2
    if args.github_repository is not None:
        try:
            expected_repository = github_repository(args.repo)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
        if args.github_repository != expected_repository:
            print(
                f"GitHub repository {args.github_repository!r} does not match workspace repository {expected_repository!r}; cargo-binstall release URLs would be wrong",
                file=sys.stderr,
            )
            return 2
    if args.print_version:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
