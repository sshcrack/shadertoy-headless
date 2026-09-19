#!/usr/bin/env python3
"""Read and validate the workspace release version."""

from __future__ import annotations

import argparse
from pathlib import Path
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tag", help="Expected git tag, e.g. v0.1.0")
    parser.add_argument(
        "--github-repository",
        help="Expected GitHub owner/repo hosting release artifacts",
    )
    parser.add_argument("--print", action="store_true", dest="print_version")
    args = parser.parse_args()

    version = workspace_version(args.repo)
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
