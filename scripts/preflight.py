#!/usr/bin/env python3
"""Preflight check for ViT-H inference, tracker, and training environments.

Inspects Python isolation, GPU/CUDA readiness, filesystem free space (including
TMPDIR and cache locations), release checkpoints, and external SAM3D/MHR/SMPL-X
assets. Exits non-zero when required checks fail.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import site
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disk_report(path: Path) -> dict[str, object]:
    probe = path.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(path),
        "probe_path": str(probe),
        "total_gb": round(usage.total / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
        "used_pct": round((usage.used / usage.total) * 100.0, 1) if usage.total else 0.0,
    }


def _check_python_isolation() -> dict[str, object]:
    user_site_enabled = bool(site.ENABLE_USER_SITE)
    user_site_dir = Path(site.getusersitepackages())
    leaked_packages = []
    if user_site_enabled and user_site_dir.is_dir():
        leaked_packages = sorted(
            entry.name for entry in user_site_dir.iterdir()
            if not entry.name.startswith(".")
        )[:10]
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "user_site_enabled": user_site_enabled,
        "user_site_dir": str(user_site_dir),
        "user_site_entries_sample": leaked_packages,
        "isolated": (not user_site_enabled) or (len(leaked_packages) == 0),
    }


def _check_gpu() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "bf16_supported": False,
    }
    try:
        import torch
    except Exception as exc:
        info["import_error"] = str(exc)
        return info

    info["torch_installed"] = True
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["device_count"] = int(torch.cuda.device_count())
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": f"sm_{props.major}{props.minor}",
                    "total_gib": round(total_bytes / (1024**3), 2),
                    "free_gib": round(free_bytes / (1024**3), 2),
                }
            )
        info["devices"] = devices
    return info


def _check_manifest_checkpoints(
    root: Path,
    profiles: tuple[str, ...],
    verify_hash: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    from scripts.checkpoints import load_manifest, verify

    manifest = load_manifest()
    records = []
    errors = []
    for item in manifest["artifacts"]:
        if item["profile"] not in profiles:
            continue
        path = root / item["local_path"]
        rec: dict[str, object] = {
            "profile": item["profile"],
            "local_path": item["local_path"],
            "expected_size_bytes": item["size_bytes"],
            "exists": path.is_file(),
            "ok": False,
        }
        if not path.is_file():
            rec["error"] = "missing"
            errors.append(f"Missing checkpoint ({item['profile']}): {item['local_path']}")
        else:
            actual_size = path.stat().st_size
            rec["actual_size_bytes"] = actual_size
            if actual_size != item["size_bytes"]:
                rec["error"] = f"size mismatch ({actual_size} != {item['size_bytes']})"
                errors.append(f"Checkpoint size mismatch: {item['local_path']}")
            elif verify_hash:
                try:
                    verify(path, item)
                    rec["ok"] = True
                    rec["sha256_verified"] = True
                except Exception as exc:
                    rec["error"] = str(exc)
                    errors.append(str(exc))
            else:
                rec["ok"] = True
                rec["sha256_verified"] = False
        records.append(rec)
    return records, errors


def _check_external_assets(
    root: Path,
    verify_hash: bool,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    fast_sam_root = root / "third-party/Fast-SAM-3D-Body"
    required_files = [
        ("sam3d_checkpoint", fast_sam_root / "checkpoints/sam-3d-body-dinov3/model.ckpt"),
        ("sam3d_config", fast_sam_root / "checkpoints/sam-3d-body-dinov3/model_config.yaml"),
        ("sam3d_mhr", fast_sam_root / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"),
        ("smplx_neutral", root / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"),
    ]
    records = []
    errors = []
    warnings = []

    repo_ok = (fast_sam_root / "sam_3d_body/__init__.py").is_file()
    records.append(
        {
            "name": "fast_sam_repo",
            "path": str(fast_sam_root),
            "ok": repo_ok,
        }
    )
    if not repo_ok:
        errors.append(f"Missing Fast-SAM-3D-Body source: {fast_sam_root}")

    for name, path in required_files:
        exists = path.is_file()
        rec: dict[str, object] = {
            "name": name,
            "path": str(path),
            "exists": exists,
            "ok": exists,
        }
        if exists:
            rec["size_bytes"] = path.stat().st_size
            if verify_hash:
                rec["sha256"] = _sha256_file(path)
        else:
            errors.append(f"Missing external asset ({name}): {path}")
        records.append(rec)

    dinov3_cache = Path("~/.cache/torch/hub/facebookresearch_dinov3_main").expanduser()
    dinov3_present = (dinov3_cache / "hubconf.py").is_file()
    records.append(
        {
            "name": "dinov3_offline_hub_cache",
            "path": str(dinov3_cache),
            "ok": dinov3_present,
            "optional_offline_cache": True,
        }
    )
    if not dinov3_present:
        warnings.append(
            f"Offline DINOv3 torch.hub cache not found at {dinov3_cache}; "
            "SAM3D will attempt online GitHub torch.hub download at runtime."
        )
    return records, errors, warnings


def _check_imports(root: Path, mode: str) -> tuple[dict[str, object], list[str]]:
    modules = []
    if mode in ("base", "all"):
        modules.extend(["torch", "torchvision", "pytorch3d", "lightning", "hydra", "smplx", "ultralytics", "yacs", "easydict"])
    if mode in ("tracker", "all"):
        modules.extend(["mujoco", "onnxruntime", "scipy", "imageio"])
    results = {}
    errors = []
    for mod in dict.fromkeys(modules):
        try:
            loaded = importlib.import_module(mod)
            results[mod] = {"ok": True, "version": getattr(loaded, "__version__", None)}
        except Exception as exc:
            results[mod] = {"ok": False, "error": str(exc)}
            errors.append(f"Cannot import {mod}: {exc}")

    if mode in ("base", "all"):
        fast_sam_root = root / "third-party/Fast-SAM-3D-Body"
        if fast_sam_root.is_dir() and str(fast_sam_root) not in sys.path:
            sys.path.insert(0, str(fast_sam_root))
        try:
            importlib.import_module("sam_3d_body")
            results["sam_3d_body"] = {"ok": True}
        except Exception as exc:
            results["sam_3d_body"] = {"ok": False, "error": str(exc)}
            errors.append(f"Cannot import sam_3d_body: {exc}")
    return results, errors


def run_preflight(
    root: Path,
    mode: str,
    require_gpu: bool,
    verify_hash: bool,
    min_free_gb: float,
    output_root: Path | None,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    isolation = _check_python_isolation()
    if not isolation["isolated"]:
        warnings.append(
            "User site-packages is enabled and non-empty; set PYTHONNOUSERSITE=1 "
            "to prevent host /root/.local packages from leaking into the environment."
        )

    tmp_dir = Path(tempfile.gettempdir())
    disk_targets = [("repo_root", root), ("tmpdir", tmp_dir)]
    if output_root is not None:
        disk_targets.append(("output_root", output_root))
    for env_name in ("TORCH_HOME", "HF_HOME", "WANDB_DIR"):
        if os.environ.get(env_name):
            disk_targets.append((env_name.lower(), Path(os.environ[env_name])))

    disks = {}
    for label, path in disk_targets:
        report = _disk_report(path)
        disks[label] = report
        if float(report["free_gb"]) < min_free_gb:
            errors.append(
                f"Low free space on {label} ({report['probe_path']}): "
                f"{report['free_gb']} GiB free < {min_free_gb} GiB threshold"
            )

    gpu = _check_gpu()
    if require_gpu and not gpu["cuda_available"]:
        errors.append("CUDA GPU required (--require-gpu) but torch.cuda.is_available() is False")

    profiles: tuple[str, ...] = (
        ("base",) if mode == "base" else ("tracker",) if mode == "tracker" else ("base", "tracker")
    )
    checkpoints, ckpt_errors = _check_manifest_checkpoints(root, profiles, verify_hash)
    errors.extend(ckpt_errors)

    external_assets: list[dict[str, object]] = []
    if mode in ("base", "all"):
        external_assets, ext_errors, ext_warnings = _check_external_assets(root, verify_hash)
        errors.extend(ext_errors)
        warnings.extend(ext_warnings)

    imports, import_errors = _check_imports(root, mode)
    errors.extend(import_errors)

    return {
        "ok": len(errors) == 0,
        "mode": mode,
        "repo_root": str(root),
        "python": isolation,
        "gpu": gpu,
        "disks": disks,
        "checkpoints": checkpoints,
        "external_assets": external_assets,
        "imports": imports,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("base", "tracker", "all"),
        default="base",
        help="Runtime profile to validate (default: base).",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Verify full SHA-256 digests of checkpoints and hash external assets.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Minimum required free GiB for repo/TMPDIR/output filesystems (default: 5.0).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    report = run_preflight(
        root=args.root.expanduser().resolve(),
        mode=args.mode,
        require_gpu=args.require_gpu,
        verify_hash=args.verify_hash,
        min_free_gb=args.min_free_gb,
        output_root=args.output_root.expanduser().resolve() if args.output_root else None,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"Preflight [{args.mode}]: {status}")
        print(f"  Python: {report['python']['python_version']} ({report['python']['python_executable']})")
        print(f"  Isolated user-site: {report['python']['isolated']}")
        gpu = report["gpu"]
        if gpu["cuda_available"]:
            dev_desc = ", ".join(
                f"gpu{d['index']}={d['name']}({d['capability']}, {d['free_gib']}/{d['total_gib']}GiB free)"
                for d in gpu["devices"]
            )
            print(f"  CUDA: {gpu['cuda_version']} | torch {gpu['torch_version']} | bf16={gpu['bf16_supported']} | {dev_desc}")
        else:
            print(f"  CUDA: unavailable (torch={gpu['torch_version']})")
        for name, disk in report["disks"].items():
            print(f"  Disk [{name}]: {disk['free_gb']} GiB free / {disk['total_gb']} GiB ({disk['probe_path']})")
        if report["warnings"]:
            for warn in report["warnings"]:
                print(f"  WARNING: {warn}")
        if report["errors"]:
            for err in report["errors"]:
                print(f"  ERROR: {err}", file=sys.stderr)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
