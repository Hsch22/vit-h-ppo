#!/usr/bin/env python3
"""Download or verify the public, revision-pinned release artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("base", "tracker", "optional", "all")


def load_manifest():
    manifest = json.loads((REPO_ROOT / "config/checkpoints.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("repo_type") != "dataset":
        raise ValueError("Unsupported checkpoint manifest schema or repository type")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Checkpoint revision must be an immutable commit")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", manifest["repo_id"]):
        raise ValueError("Invalid Hugging Face repository ID")
    destinations = set()
    for item in manifest["artifacts"]:
        for key in ("path", "local_path"):
            path = PurePosixPath(item[key])
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Invalid artifact {key}")
        if item["local_path"] in destinations:
            raise ValueError("Duplicate checkpoint destination")
        destinations.add(item["local_path"])
        if item["profile"] not in PROFILES[:-1]:
            raise ValueError("Unsupported checkpoint profile")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid checkpoint SHA-256")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] <= 0:
            raise ValueError("Invalid checkpoint size")
    if not destinations:
        raise ValueError("Empty checkpoint manifest")
    return manifest


def verify(path, item):
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    if path.stat().st_size != item["size_bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def download(root, manifest, item):
    destination = root / item["local_path"]
    if destination.exists():
        verify(destination, item)
        print(f"Already verified: {item['local_path']}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/"
        f"{manifest['revision']}/{quote(item['path'], safe='/')}"
    )
    temporary = None
    try:
        print(f"Downloading {item['path']} ({item['size_bytes'] / 1e6:.1f} MB)", flush=True)
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".", suffix=".part",
            dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    output.write(block)
        verify(temporary, item)
        # An exclusive hard link publishes the verified file without replacing
        # an existing checkpoint, including one created by a concurrent job.
        os.link(temporary, destination)
        print(f"Verified: {item['local_path']}", flush=True)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} downloading {item['path']}") from None
    except URLError:
        raise RuntimeError(f"Network error downloading {item['path']}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "download", "verify"))
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="Artifact root, default: this repository")
    args = parser.parse_args()
    if args.command != "list" and args.profile is None:
        parser.error("download/verify requires an explicit --profile")
    manifest = load_manifest()
    profile = args.profile or "all"
    items = [item for item in manifest["artifacts"]
             if profile == "all" or item["profile"] == profile]
    if not items:
        raise ValueError("No artifacts match this profile")
    print(f"{manifest['repo_id']} @ {manifest['revision']}", flush=True)
    root = args.root.expanduser().resolve()
    for item in items:
        if args.command == "list":
            print(f"{item['profile']:8} {item['size_bytes'] / 1e6:8.1f} MB  {item['local_path']}")
        elif args.command == "verify":
            verify(root / item["local_path"], item)
            print(f"Verified: {item['local_path']}", flush=True)
        else:
            download(root, manifest, item)
    print(f"Selected total: {sum(item['size_bytes'] for item in items) / 1e9:.3f} GB")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
