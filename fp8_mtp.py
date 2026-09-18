#!/usr/bin/env python3
"""Rewrite the Qwen MTP head to FP8 for native MXFP4 vLLM loading.

This module is executed by setup-mxfp4.sh inside the pinned container. It does
not require Python, ROCm, or Hugging Face tooling on the host.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

MTP_KEY = re.compile(r"(^|\.)mtp(\.|$)", re.IGNORECASE)


def _set_quantization_metadata(config: dict[str, Any]) -> None:
    """Declare an MXFP4 body and an FP8 module-name override for ``mtp``."""
    quant = config.setdefault("quantization_config", {})
    if not isinstance(quant, dict):
        raise ValueError("quantization_config must be an object")
    quant["quant_method"] = "mxfp4"
    ignored = quant.setdefault("modules_to_not_convert", [])
    if not isinstance(ignored, list):
        raise ValueError("quantization_config.modules_to_not_convert must be a list")
    if "mtp" not in ignored:
        ignored.append("mtp")
    overrides = quant.setdefault("module_quantization", {})
    if not isinstance(overrides, dict):
        raise ValueError("quantization_config.module_quantization must be an object")
    overrides["mtp"] = {"quant_method": "fp8", "module_name": "mtp"}
    # Keep this top-level declaration as well. It makes the intended module
    # boundary inspectable by tooling that does not understand vLLM overrides.
    config["mtp_quantization_config"] = {"module_name": "mtp", "quant_method": "fp8"}


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rewrite_config(source: Path, output: Path) -> None:
    config = _load_config(source / "config.json")
    _set_quantization_metadata(config)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mtp_tensor(name: str) -> bool:
    return bool(MTP_KEY.search(name))


def _rewrite_safetensors(output: Path) -> int:
    """Convert floating-point tensors belonging to mtp modules to FP8.

    The conversion is deliberately restricted to module-qualified keys. Body
    MXFP4 tensors and unrelated model weights are copied byte-for-byte. A
    per-file temporary is used so an interrupted rewrite leaves the copied
    checkpoint intact rather than producing a partially-written shard.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "the pinned image needs torch and safetensors to run fp8_mtp.py"
        ) from exc

    converted = 0
    for shard in sorted(output.glob("*.safetensors")):
        tensors = load_file(str(shard), device="cpu")
        changed = False
        for name, tensor in list(tensors.items()):
            if _mtp_tensor(name) and tensor.is_floating_point():
                if tensor.dtype != torch.float8_e4m3fn:
                    tensors[name] = tensor.to(dtype=torch.float8_e4m3fn)
                    converted += 1
                    changed = True
        if changed:
            with tempfile.NamedTemporaryFile(
                dir=shard.parent, prefix=f".{shard.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                save_file(tensors, str(temporary))
                temporary.replace(shard)
            finally:
                temporary.unlink(missing_ok=True)
    return converted


def rewrite(source: Path, output: Path, metadata_only: bool = False) -> int:
    """Create the rewritten checkpoint and return the number of FP8 tensors."""
    source = source.resolve()
    output = output.resolve()
    if not (source / "config.json").is_file():
        raise ValueError(f"source checkpoint has no config.json: {source}")
    if source == output:
        raise ValueError("source and output checkpoints must be different directories")
    if output.exists():
        raise FileExistsError(f"output already exists; remove it only if rewrite is incomplete: {output}")
    shutil.copytree(source, output, symlinks=True)
    try:
        _rewrite_config(source, output)
        converted = 0 if metadata_only else _rewrite_safetensors(output)
        if not metadata_only and converted == 0:
            raise RuntimeError("no floating-point mtp.* tensors were found in safetensors shards")
        marker = {
            "source": source.name,
            "body_quant_method": "mxfp4",
            "mtp_module": "mtp",
            "mtp_quant_method": "fp8",
            "converted_tensors": converted,
        }
        (output / ".fp8-mtp-rewrite.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return converted
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify(checkpoint: Path) -> None:
    """Verify the metadata contract required by the launcher."""
    config = _load_config(checkpoint / "config.json")
    quant = config.get("quantization_config")
    mtp_quant = config.get("mtp_quantization_config")
    body = quant.get("quant_method") if isinstance(quant, dict) else None
    mtp_method = mtp_quant.get("quant_method") if isinstance(mtp_quant, dict) else None
    mtp_module = mtp_quant.get("module_name") if isinstance(mtp_quant, dict) else None
    if body != "mxfp4":
        raise ValueError(f"body quant_method must be mxfp4, got {body!r}")
    if mtp_module != "mtp" or mtp_method != "fp8":
        raise ValueError("MTP metadata must declare module_name=mtp and quant_method=fp8")
    if not isinstance(quant, dict) or quant.get("module_quantization", {}).get("mtp", {}).get("quant_method") != "fp8":
        raise ValueError("quantization_config.module_quantization.mtp must declare fp8")
    print(f"verified mxfp4 body + fp8 mtp metadata: {checkpoint}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="copy and rewrite metadata without tensor conversion (test fixture mode)",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.verify)
        else:
            if args.output is None:
                parser.error("--output is required with --source")
            count = rewrite(args.source, args.output, metadata_only=args.metadata_only)
            print(f"rewrote {count} MTP tensor(s) to fp8: {args.output}")
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"fp8_mtp.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
