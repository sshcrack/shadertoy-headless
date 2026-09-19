#!/usr/bin/env python3
"""Create the release archive layout consumed by cargo-binstall."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import tarfile
import zipfile

from version import workspace_version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_tgz(binary: Path, output: Path) -> None:
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                info = archive.gettarinfo(str(binary), arcname="shadertoy")
                info.mode = 0o755
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = mtime
                with binary.open("rb") as handle:
                    archive.addfile(info, handle)


def make_zip(binary: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("shadertoy.exe")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.create_system = 3
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, binary.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    version = workspace_version(args.repo)
    if args.tag is not None and args.tag != f"v{version}":
        parser.error(f"tag {args.tag!r} does not match workspace version v{version}")

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    windows = "windows" in args.target
    expected_name = "shadertoy.exe" if windows else "shadertoy"
    if binary.name != expected_name:
        parser.error(f"expected binary named {expected_name!r}, got {binary.name!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if windows else ".tgz"
    archive = args.out / f"shadertoy-cli-{args.target}-v{version}{suffix}"
    if windows:
        make_zip(binary, archive)
    else:
        make_tgz(binary, archive)

    checksum = sha256(archive)
    checksum_file = archive.with_name(f"{archive.name}.sha256")
    checksum_file.write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
