from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu-detect.sh"
SERVE = ROOT / "serve-mxfp4.sh"
SETUP = ROOT / "setup-mxfp4.sh"
CALIBRATE = ROOT / "calibrate-kv.sh"


def bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=merged,
        text=True, capture_output=True, check=False,
    )


def make_sysfs(tmp_path: Path, vram_and_ids: list[tuple[int, str]]) -> Path:
    root = tmp_path / "sys"
    for index, (vram_mib, device_id) in enumerate(vram_and_ids, start=128):
        device = root / "class" / "drm" / f"renderD{index}" / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text(str(vram_mib * 1024 * 1024))
        (device / "device").write_text(device_id)
    return root

def test_hf_cache_snapshot_reuse_and_mismatch_report(tmp_path: Path) -> None:
    """Exact cached repos are staged; the supplied mismatched repo is only reported."""
    cache = tmp_path / "hub"
    repo = "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8"
    snapshot = cache / "models--tcclaviger--Qwen3.8-Flash-Next-MXFP4-FP8" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    result = bash(
        f"source '{SETUP}'; snapshot=$(hf_cache_snapshot '{repo}'); printf '%s\\n' \"$snapshot\"; "
        f"report_hf_cache; destination='{tmp_path / 'staged'}'; "
        f"copy_hf_snapshot '{repo}' \"$destination\"; test -f \"$destination/config.json\"",
        env={"HF_HUB_CACHE": str(cache), "HF_CACHE_REPO": repo, "SETUP_LIB_ONLY": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert str(snapshot) in result.stdout
    assert "is not used" in result.stdout
    assert "fixed target" in result.stdout


def test_setup_container_scopes_gpus_and_skips_bandwidth_probe(tmp_path: Path) -> None:
    """Setup containers receive the detected GPUs and do not run the startup probe."""
    result = bash(
        "source 'setup-mxfp4.sh'; RUNTIME_KIND=podman; HIP_VISIBLE_DEVICES=0,3; "
        f"MODELS='{tmp_path / 'models'}'; R4D_CACHE='{tmp_path / 'cache'}'; "
        "container_args args; container_args python_args python; container_args bash_args bash; "
        "printf '%s\\n' \"${args[*]}\"; printf '%s\\n' \"${python_args[*]}\"; printf '%s\\n' \"${bash_args[*]}\"",
        env={"SETUP_LIB_ONLY": "1"},
    )
    assert result.returncode == 0
    assert "HIP_VISIBLE_DEVICES=0,3" in result.stdout
    assert "RADIANCE_RUN_BWTEST=0" in result.stdout
    assert "--entrypoint python" in result.stdout
    assert "--entrypoint bash" in result.stdout


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=-4, max_value=20))
def test_p1_tp_divides_head_counts(candidate: int) -> None:
    """Feature: vllm-mxfp4-r9700, Property 1: TP divides the head counts."""
    result = bash(f"source '{GPU}'; tp_is_valid {candidate}")
    assert (result.returncode == 0) is (candidate in {1, 2, 4, 8})


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=40000), min_size=1, max_size=12))
def test_p2_subthreshold_gpus_are_excluded(vram_values: list[int]) -> None:
    """Feature: vllm-mxfp4-r9700, Property 2: sub-threshold GPUs are always excluded."""
    values = " ".join(map(str, vram_values))
    result = bash(
        f"source '{GPU}'; for value in {values}; do "
        "if is_vram_usable \"$value\"; then printf '1 '; else printf '0 '; fi; done"
    )
    assert result.returncode == 0
    assert result.stdout.split() == ["1" if value >= 8192 else "0" for value in vram_values]


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=100000))
def test_p3_setup_profile_step_is_idempotent(kv_mem: int) -> None:
    """Feature: vllm-mxfp4-r9700, Property 3: setup is idempotent."""
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "profiles.tsv"
        env = {
            "HW_SIG": "2x7551-32624", "KV_PROFILE_FILE": str(profile),
            "KV_MEM_VALUE": str(kv_mem), "KV_TOKENS": "100",
        }
        first = subprocess.run([str(CALIBRATE)], cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True)
        second = subprocess.run([str(CALIBRATE)], cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True)
        assert first.returncode == second.returncode == 0
        rows = [line for line in profile.read_text().splitlines() if line and not line.startswith("hw_sig")]
        assert len(rows) == 1
        assert rows[0].split("\t")[4] == str(kv_mem)


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=32), st.integers(min_value=1, max_value=10_000))
def test_p4_served_checkpoint_is_rewrite(seqs: int, chunk: int) -> None:
    """Feature: vllm-mxfp4-r9700, Property 4: the served checkpoint is always the mtpfp8 rewrite."""
    script = (
        f"SERVE_LIB_ONLY=1 source '{SERVE}'; "
        f"TP=2 HARDWARE_SIG=2x7551-32624 MAXSEQS={seqs} CHUNK={chunk} "
        "assemble_command; printf '%s\\n' \"${VLLM_COMMAND[*]}\""
    )
    result = bash(script)
    assert result.returncode == 0
    assert "/models/Qwen3.8-27B-MXFP4-mtpfp8" in result.stdout
    assert "Quark-AWQ-MXFP4" not in result.stdout


@settings(max_examples=100, deadline=None)
@given(st.booleans())
def test_p5_pinned_libr4d_gate(ready: bool) -> None:
    """Feature: vllm-mxfp4-r9700, Property 5: the pinned libr4d is always used, never the stock kernels."""
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "r4d"
        cache.mkdir()
        if ready:
            (cache / ".pin").write_text("b9e42ab\n")
        result = bash(
            f"SERVE_LIB_ONLY=1 R4D_CACHE='{cache}' source '{SERVE}'; verify_r4d_cache",
            env={"R4D_CACHE": str(cache)},
        )
        assert (result.returncode == 0) is ready
        if not ready:
            assert "stock fallback" in result.stderr or "missing" in result.stderr


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=32), st.sampled_from(["dflash", "mtp"]))
def test_p6_native_mxfp4_command_flags(seqs: int, method: str) -> None:
    """Feature: vllm-mxfp4-r9700, Property 6: native MXFP4 means no --quantization flag."""
    result = bash(
        f"SERVE_LIB_ONLY=1 source '{SERVE}'; TP=2 HARDWARE_SIG=2x7551-32624 "
        f"MAXSEQS={seqs} SPEC_METHOD={method}; assemble_command; printf '%s\\n' \"${{VLLM_COMMAND[*]}}\""
    )
    assert result.returncode == 0
    assert "--quantization" not in result.stdout
    for flag in ("--enable-auto-tool-choice", "--tool-call-parser qwen3_coder", "--reasoning-parser qwen3", "qwen-fixed-v22.3.jinja"):
        assert flag in result.stdout


@settings(max_examples=100, deadline=None)
@given(st.sampled_from(["dflash", "mtp"]))
def test_p7_dry_run_has_no_execution(method: str) -> None:
    """Feature: vllm-mxfp4-r9700, Property 7: DRY_RUN prints exactly what would run, and runs nothing."""
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        models = tmp_path / "models"
        cache = tmp_path / "cache"
        models.mkdir()
        cache.mkdir()
        (cache / ".pin").write_text("b9e42ab\n")
        sysfs = make_sysfs(tmp_path, [(32624, "0x7551"), (32624, "0x7551")])
        fakebin = tmp_path / "bin"
        fakebin.mkdir()
        marker = tmp_path / "called"
        fake = fakebin / "runtime"
        fake.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        fake.chmod(0o755)
        env = {
            "PATH": f"{fakebin}:{os.environ['PATH']}", "CONTAINER_RUNTIME": "runtime",
            "MODELS": str(models), "R4D_CACHE": str(cache), "SYSFS_ROOT": str(sysfs),
            "GPU_DETECT_SKIP_DEVICE_CHECK": "1", "DRY_RUN": "1", "SPEC_METHOD": method,
        }
        result = subprocess.run([str(SERVE)], cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "DRY_RUN=1: no container started" in result.stdout
        assert not marker.exists()
        assert "vllm serve" in result.stdout


spec = importlib.util.spec_from_file_location("fp8_mtp", ROOT / "fp8_mtp.py")
assert spec and spec.loader
fp8_mtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8_mtp)
smoke_spec = importlib.util.spec_from_file_location("smoke_validate", ROOT / "smoke_validate.py")
assert smoke_spec and smoke_spec.loader
smoke_validate = importlib.util.module_from_spec(smoke_spec)
smoke_spec.loader.exec_module(smoke_validate)


@settings(max_examples=100, deadline=None)
@given(st.text(min_size=1, max_size=30, alphabet=st.characters(blacklist_categories=("Cs",))))
def test_p8_completion_is_nonempty_and_nan_free(content: str) -> None:
    """Feature: vllm-mxfp4-r9700, Property 8: smoke-test content is non-empty and NaN-free."""
    body = {"choices": [{"message": {"content": content}}]}
    valid = bool(content.strip()) and "nan" not in content.lower()
    try:
        fp8_mtp  # keep module loading explicit for test collection environments
        observed = smoke_validate.completion_content(body) is content
    except ValueError:
        observed = False
    assert observed is valid


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=100000), st.integers(min_value=1, max_value=100000))
def test_p9_kv_profile_round_trips_and_is_unique(kv_mem: int, tokens: int) -> None:
    """Feature: vllm-mxfp4-r9700, Property 9: KV profile lookup round-trips and stays unique."""
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "profiles.tsv"
        env = {"HW_SIG": "2x7551-32624", "KV_PROFILE_FILE": str(profile), "KV_MEM_VALUE": str(kv_mem), "KV_TOKENS": str(tokens)}
        for _ in range(2):
            result = subprocess.run([str(CALIBRATE)], cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
        rows = [line for line in profile.read_text().splitlines() if line.startswith("2x7551-32624")]
        assert len(rows) == 1
        assert rows[0].split("\t")[4:6] == [str(kv_mem), str(tokens)]
