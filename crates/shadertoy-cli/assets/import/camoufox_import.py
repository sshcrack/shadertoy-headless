from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from camoufox.sync_api import Camoufox


DYNAMIC_CUBEMAP_ID = "4dX3Rr"
ALLOWED_RESOURCE_HOSTS = {"shadertoy.com", "www.shadertoy.com"}


def resource_url(path: str) -> str:
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError(f"unsupported ShaderToy resource scheme: {parsed.scheme}")
        if parsed.hostname is None or parsed.hostname.lower() not in ALLOWED_RESOURCE_HOSTS:
            raise RuntimeError(f"refusing non-ShaderToy resource URL: {path}")
    resolved = urljoin("https://www.shadertoy.com/", path)
    resolved_parsed = urlsplit(resolved)
    if resolved_parsed.hostname is None or resolved_parsed.hostname.lower() not in ALLOWED_RESOURCE_HOSTS:
        raise RuntimeError(f"refusing non-ShaderToy resource URL: {path}")
    return resolved


def cubemap_faces(path: str) -> list[str]:
    parsed = urlsplit(path)
    stem_path = Path(parsed.path)
    suffix = stem_path.suffix
    if not suffix:
        raise RuntimeError(f"cubemap resource has no extension: {path}")
    base = parsed.path[: -len(suffix)]
    result = []
    for face_suffix in ("", "_1", "_2", "_3", "_4", "_5"):
        face_path = f"{base}{face_suffix}{suffix}"
        if parsed.query:
            face_path += f"?{parsed.query}"
        result.append(face_path)
    return result


def required_resources(payload: object) -> list[str]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("ShaderToy returned an unexpected response shape")

    resources: list[str] = []
    seen: set[str] = set()
    for render_pass in payload[0].get("renderpass", []):
        if not isinstance(render_pass, dict):
            continue
        for shader_input in render_pass.get("inputs", []):
            if not isinstance(shader_input, dict):
                continue
            kind = shader_input.get("type")
            path = shader_input.get("filepath")
            if kind not in {"texture", "cubemap", "volume"} or not isinstance(path, str):
                continue
            if kind == "cubemap" and shader_input.get("id") == DYNAMIC_CUBEMAP_ID:
                continue
            paths = cubemap_faces(path) if kind == "cubemap" else [path]
            for resource in paths:
                if resource not in seen:
                    seen.add(resource)
                    resources.append(resource)
    return resources


def fetch_shader(page, shader_id: str) -> tuple[int, str]:
    return page.evaluate(
        """async (id) => {
            const body = new URLSearchParams();
            body.set("s", JSON.stringify({shaders: [id]}));
            body.set("nt", "1");
            body.set("nl", "1");
            body.set("np", "1");
            const response = await fetch("/shadertoy", {
                method: "POST",
                credentials: "same-origin",
                headers: {"content-type": "application/x-www-form-urlencoded; charset=UTF-8"},
                body: body.toString(),
            });
            return [response.status, await response.text()];
        }""",
        shader_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shader-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    shader_id = args.shader_id
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resources_dir = output / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    canonical_url = f"https://www.shadertoy.com/view/{shader_id}"
    profile = Path(args.profile)
    profile.mkdir(parents=True, exist_ok=True)

    override = os.environ.get("SHADERTOY_CAMOUFOX_HEADLESS", "").strip().lower()
    if override in {"1", "true", "headless"}:
        headless_mode = True
    elif override == "virtual":
        headless_mode = "virtual"
    elif override in {"0", "false", "headed", "headful"}:
        headless_mode = False
    elif sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        headless_mode = "virtual"
    else:
        headless_mode = False

    with Camoufox(
        headless=headless_mode,
        persistent_context=True,
        user_data_dir=str(profile),
        humanize=True,
        disable_coop=True,
        i_know_what_im_doing=True,
        window=(1280, 720),
    ) as context:
        page = context.new_page()
        page.goto(canonical_url, wait_until="domcontentloaded", timeout=90_000)
        # Let automatic verification finish. If Cloudflare presents its visible
        # browser control, interact with that control inside the Camoufox page and
        # retain the resulting clearance in the persistent profile.
        status = 0
        body = ""
        interactive = headless_mode is False
        attempts = 180 if interactive else 20
        announced = False
        for _ in range(attempts):
            try:
                status, body = fetch_shader(page, shader_id)
            except Exception:
                # Completing an interactive challenge can navigate/reload the page
                # while evaluate() is in flight. Treat that as progress and retry.
                page.wait_for_timeout(1000)
                continue
            if status == 200:
                break
            if status not in {403, 429, 503}:
                raise RuntimeError(f"ShaderToy browser request returned HTTP {status}")

            # Managed Cloudflare verification sometimes exposes a Turnstile frame.
            # Camoufox's COOP override makes its iframe geometry available; clicking
            # the visible verification control is equivalent to the normal browser
            # interaction and lets the same profile retain any clearance cookie.
            challenge = next(
                (frame for frame in page.frames if "challenges.cloudflare.com" in frame.url),
                None,
            )
            if challenge is not None:
                try:
                    box = challenge.locator("body").bounding_box()
                    if box is not None and box["width"] > 0 and box["height"] > 0:
                        page.mouse.click(
                            box["x"] + min(30, box["width"] / 2),
                            box["y"] + box["height"] / 2,
                        )
                except Exception:
                    pass

            if interactive and not announced:
                print(
                    "ShaderToy requested browser verification. Camoufox is attempting the browser check; use the visible window if manual interaction is still requested.",
                    file=sys.stderr,
                    flush=True,
                )
                announced = True
            page.wait_for_timeout(1000)
        if status != 200:
            if not interactive:
                raise RuntimeError(
                    "ShaderToy requires interactive browser verification. Re-run import from a graphical desktop, or set SHADERTOY_CAMOUFOX_HEADLESS=false when a display is available. The Camoufox profile is reused for later imports."
                )
            raise RuntimeError(
                f"ShaderToy browser verification did not complete (HTTP {status})"
            )

        payload = json.loads(body)
        response_path = output / "response.json"
        response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        captured: dict[str, str] = {}
        for index, resource in enumerate(required_resources(payload)):
            suffix = Path(urlsplit(resource).path).suffix or ".bin"
            filename = f"{index:04d}{suffix}"
            resolved_url = resource_url(resource)
            resource_page = context.new_page()
            try:
                fetched = resource_page.goto(
                    resolved_url,
                    wait_until="commit",
                    timeout=90_000,
                    referer=canonical_url,
                )
                if fetched is None or not fetched.ok:
                    status = fetched.status if fetched is not None else "no response"
                    raise RuntimeError(
                        f"ShaderToy resource {resource!r} returned HTTP {status}"
                    )
                (resources_dir / filename).write_bytes(fetched.body())
            finally:
                resource_page.close()
            captured[resource] = f"resources/{filename}"

    metadata = {
        "shader_id": shader_id,
        "source_url": canonical_url,
        "response": "response.json",
        "resources": captured,
    }
    (output / "browser-import.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Camoufox import error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
