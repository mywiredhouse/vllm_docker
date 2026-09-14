# Requirements Document

## Introduction

A standalone containerised vLLM inference server that runs Qwen3.8-27B in native 4-bit MXFP4 on exactly two AMD Radeon AI PRO R9700 GPUs (gfx1201 / RDNA4) with tensor-parallel = 2. The stack is built around the upstream reference `radiance-vllm-mxfp4` (https://codeberg.org/ggz14/radiance-vllm-mxfp4), a launcher-style repo that pulls a prebuilt image and applies patches and kernels at container start rather than building an image from source.

This repo is **standalone**. It borrows hardware and operational conventions from `rocm_docker` (Fedora host, ROCm, dual-R9700 device selection, podman-preferred, interactive-before-automated discipline) but does **not** depend on it or share code.

The definition of done for this spec's scope: from a fresh clone, an operator can reach a first successful `vllm serve` that answers `/v1/chat/completions` on the two R9700 cards with the MXFP4 model.

### Scope

The single target is **MXFP4 Qwen3.8-27B on 2x R9700 at TP=2** via the fast image path (pull pinned upstream image, patch at start).

Explicitly **out of scope / future work**:
- TP other than 2 (TP=3, TP=4, TP=8) — TP must divide `num_attention_heads=24`, `linear_num_key_heads=16`, and `linear_num_value_heads=48`; only TP=2 fits two R9700
- ParoQuant or any quantisation other than native MXFP4
- The uncensored checkpoint variant (`just1moremodel/Qwen3.8-27B-Uncensored-MXFP4-awq`)
- Build-from-source via the 4-stage Dockerfile (Requirement 8 documents it as a fallback only)
- Multi-node / distributed serving beyond a single host

## Hardware Context

### Target System (2x R9700 in scope)
- GPU: 2x AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GiB each, wave32
- The dev box also contains an MI100 (gfx908) and a 7900 XTX (gfx1100) and a small iGPU — **none of these are in scope**; only the two R9700 are used
- Validated device selection: `HIP_VISIBLE_DEVICES=0,3`
- Hardware signature: `2x7551-32624` (device `0x7551`, 32624 MiB usable per card)
- Host: Fedora, amdgpu driver exposing `/dev/kfd` and `/dev/dri`
- ROCm userspace is **inside the image** — no host ROCm, Python, or HuggingFace CLI required
- Container runtime: **podman preferred** (rootless; container runs as a non-root user in the video/render groups, uses `--replace` and `--userns keep-id` / `keep-groups`), docker supported

### Software Stack (pinned)
- Base image: `stilldeadcode/vllm-radiance:0.9.3` (ROCm + PyTorch + Triton + AITER + vLLM compiled for gfx1201 only)
- Target checkpoint: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (~19 GiB, Quark OCP micro-scaling, Apache-2.0)
- MTP-rewritten checkpoint: `Qwen3.8-27B-MXFP4-mtpfp8` (produced locally by `fp8_mtp.py`, ~15 min)
- Drafter (speculative decoding): `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (~2 GiB)
- Kernels: `libr4d` (https://codeberg.org/StillDeadcode/libr4d), gfx1201 HIP attention/GEMM/all-reduce, built inside the image on first run, cached at `~/.cache/radiance-libr4d`, pinned at `R4D_PIN=b9e42ab`

### The MTP Load Problem
The target checkpoint does **not** load as-is. Its `mtp.*` layers are bf16 but are excluded by tensor-name rather than module-name, so vLLM tries to apply mxfp4 to a bf16 head and the load dies. The `fp8_mtp.py` rewrite requantizes the MTP head to fp8 and declares it properly, producing `Qwen3.8-27B-MXFP4-mtpfp8`. This rewrite is mandatory for the default (dflash) and mtp speculative paths.

### The libr4d NaN Problem
The image's stock `libr4d` produces NaNs on this model's gated-delta-net. Building the pinned `libr4d` (`AUTO_R4D=1`, `R4D_PIN=b9e42ab`) is **load-bearing** — without it the model produces garbage output.

### Measured Reference Result (2x R9700, TP=2, from upstream)
- Weights: 9.4 GiB per GPU
- KV cache: 943,581 tokens at 262,144 context
- Decode step: 22.7 ms
- Aggregate throughput: 573 t/s at 8 concurrent requests
- Quality: WikiText-2 perplexity 8.3708, GSM8K 97.8%

### Disk Budget
~60 GiB for full setup: 19 GiB source checkpoint + 19 GiB rewritten checkpoint + 2 GiB drafter + ~10 GiB image + margin.

## Glossary

- **Image**: The prebuilt container image `stilldeadcode/vllm-radiance:0.9.3` (ROCm + PyTorch + Triton + AITER + vLLM for gfx1201)
- **Container**: A running instance of the Image, launched by the serve launcher
- **Host**: The Fedora machine with the amdgpu driver, exposing `/dev/kfd` and `/dev/dri`
- **Launcher path**: The fast path — pull the pinned Image, apply patches/kernels at container start; never builds an image
- **MXFP4**: OCP micro-scaling 4-bit float quantisation. Native MXFP4 enabled by `RADIANCE_MXFP4=1` (reads `quant_method` from `config.json`; `--quantization` is dropped)
- **W4A8**: Optional hand-written fp8-WMMA GEMM path (`RADIANCE_MXFP4_W4A8=1`); more accurate than declared W4A4, 1.47–2.26x faster than aiter
- **TP / tensor-parallel**: Sharding a model across GPUs. TP=2 for two R9700 (must divide 24, 16, 48)
- **MTP**: Multi-Token Prediction head in the target checkpoint; used by `SPEC_METHOD=mtp` (no extra download)
- **dflash**: Default speculative decoding method (`SPEC_METHOD=dflash`) using the separate DFlash2 drafter model
- **Drafter**: The speculative decoding draft model `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (~2 GiB)
- **libr4d**: gfx1201 HIP kernel library (attention/GEMM/all-reduce). Pinned build required — stock NaNs the gated-delta-net
- **fp8_mtp.py**: The rewrite script that requantizes the MTP head to fp8, producing a loadable checkpoint
- **gpu-detect**: Logic that reads sysfs `mem_info_vram_total` per amdgpu render node, excludes cards < 8192 MiB (skips iGPU), and computes usable GPU count and TP
- **Hardware signature**: A key identifying the GPU config for KV calibration, e.g. `2x7551-32624`
- **KV calibration**: Pinning a safe KV cache size per hardware signature and batch shape via `calibrate-kv` + `kv-profiles.tsv`
- **DRY_RUN**: Launcher mode that prints the full `vllm serve` command without executing it

## Requirements

### Requirement 1: Host Preflight and GPU / TP Detection

**User Story:** As an operator on a fresh clone, I want the setup to verify the host can run the stack and to detect the two R9700 cards and derive TP=2, so that I never launch against wrong or insufficient hardware.

#### Acceptance Criteria

1. IF the amdgpu driver does not expose both `/dev/kfd` and `/dev/dri`, THEN THE setup script SHALL exit with a non-zero status and SHALL print which device node is missing.
2. WHEN the setup script runs, THE setup script SHALL verify a supported container runtime is present (podman preferred, docker supported) and SHALL print the name of the runtime it will use.
3. WHEN GPU detection runs, THE detection logic SHALL read sysfs `mem_info_vram_total` per amdgpu render node and SHALL exclude every device reporting less than 8192 MiB usable VRAM, so the iGPU is skipped.
4. WHEN GPU detection runs on the target system, THE detection logic SHALL identify exactly two R9700 cards matching signature `2x7551-32624` and SHALL derive TP = 2.
5. WHERE more than two usable cards are present, THE detection logic SHALL restrict device selection to the two R9700 cards via `HIP_VISIBLE_DEVICES=0,3` and SHALL set TP = 2.
6. IF the derived TP does not divide each of `num_attention_heads=24`, `linear_num_key_heads=16`, and `linear_num_value_heads=48`, THEN THE detection logic SHALL exit with a non-zero status and SHALL print the divisibility constraint that failed.
7. WHEN the setup script completes preflight, THE setup script SHALL print the free disk space in the models directory against the 60 GiB budget and SHALL print a warning WHEN free space is below 60 GiB.
8. WHEN the setup script is re-run after a partial or complete setup, THE setup script SHALL detect existing state and SHALL skip steps already completed.

### Requirement 2: Image Acquisition (Launcher Path)

**User Story:** As an operator, I want the pinned upstream image pulled so that I get a known-good ROCm/PyTorch/Triton/AITER/vLLM build for gfx1201 without a multi-hour compile.

#### Acceptance Criteria

1. WHEN image acquisition runs, THE setup script SHALL pull the pinned image `stilldeadcode/vllm-radiance:0.9.3` using the selected runtime.
2. WHEN image acquisition runs, THE setup script SHALL pull the exact pinned tag `0.9.3` and SHALL NOT pull `latest` or any other tag.
3. WHEN the pinned image is already present locally, THE setup script SHALL skip the pull.
4. WHEN image acquisition completes, THE setup script SHALL verify the pinned image is present locally and SHALL record its image digest.
5. THE launcher path SHALL reach a running MXFP4 stack without building an image.

### Requirement 3: Target Checkpoint Download and fp8 MTP Rewrite

**User Story:** As an operator, I want the target MXFP4 checkpoint downloaded and its MTP head rewritten to fp8 so that vLLM can actually load the model instead of dying on a bf16 head.

#### Acceptance Criteria

1. WHEN checkpoint download runs, THE setup script SHALL download `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (~19 GiB) into `${MODELS:-~/models}`.
2. WHEN the source checkpoint is present, THE setup script SHALL run the fp8 MTP rewrite (`fp8_mtp.py`) to requantize the `mtp.*` head to fp8 and declare it by module-name, producing `Qwen3.8-27B-MXFP4-mtpfp8`.
3. WHEN the rewrite completes, THE rewritten checkpoint SHALL declare the MTP head as fp8 (not bf16) so vLLM does not apply mxfp4 to a bf16 head.
4. IF the rewritten checkpoint `Qwen3.8-27B-MXFP4-mtpfp8` already exists, THEN THE setup script SHALL skip the rewrite (idempotent).
5. THE setup script SHALL run the rewrite inside the Container (no host Python or HF CLI required).
6. WHEN the rewrite completes, THE setup script SHALL verify that the rewritten checkpoint's `config.json` declares `mxfp4` as the body quantisation method and `fp8` for the MTP head.

### Requirement 4: Drafter Download (Speculative Decoding)

**User Story:** As an operator using the default dflash speculative method, I want the drafter model downloaded so that speculative decoding works out of the box.

#### Acceptance Criteria

1. WHEN `SPEC_METHOD=dflash` (default), THE setup script SHALL download the drafter `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (~2 GiB) into `${MODELS:-~/models}`.
2. WHERE `SPEC_METHOD=mtp` is selected, THE setup script SHALL use the in-target MTP head and SHALL NOT download the separate drafter.
3. IF the drafter is already present, THEN THE setup script SHALL skip the download (idempotent).
4. THE drafter path SHALL be resolvable by the launcher via the `DRAFTER` config knob.

### Requirement 5: libr4d Kernel Build

**User Story:** As an operator, I want the pinned libr4d kernels built inside the image so that the model produces correct output instead of NaNs.

#### Acceptance Criteria

1. WHEN kernel build runs with `AUTO_R4D=1` (default), THE setup or serve script SHALL build `libr4d` pinned at `R4D_PIN=b9e42ab` inside the Container.
2. THE kernel build SHALL cache its output at `~/.cache/radiance-libr4d` so subsequent runs skip the build.
3. WHEN a cached pinned build exists, THE serve script SHALL reuse it and SHALL NOT rebuild.
4. THE stack SHALL NOT rely on the image's stock `libr4d`, because it NaNs this model's gated-delta-net.
5. IF the pinned libr4d build fails, THEN THE serve script SHALL exit with a non-zero status, SHALL print the build failure, and SHALL NOT fall back to the stock kernels.

### Requirement 6: Launcher with TP=2 and Fixed Serving Flags

**User Story:** As an operator, I want a launcher that assembles the correct `vllm serve` command with detected TP=2, native MXFP4, and the fixed tool/reasoning flags, so that serving is reproducible and inspectable.

#### Acceptance Criteria

1. WHEN the serve launcher runs, THE launcher SHALL set `RADIANCE_MXFP4=1` (native MXFP4, reads `quant_method` from `config.json`) and SHALL NOT pass `--quantization`.
2. WHEN the serve launcher runs, THE launcher SHALL set `RADIANCE_MXFP4_W4A8=1` (fp8-WMMA W4A8 GEMM) by default and SHALL document in its help output that this path uses W4A8 numerics rather than the declared W4A4, is more accurate, and is 1.47–2.26x faster than aiter.
3. WHEN the serve launcher runs, THE launcher SHALL use the detected TP (2 on the target box) and SHALL restrict devices to the two R9700 via `HIP_VISIBLE_DEVICES=0,3`.
4. WHEN the serve launcher runs, THE launcher SHALL pass the fixed serving flags `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` and the repo chat template `qwen-fixed-v22.3.jinja`.
5. THE launcher SHALL expose config knobs as `${VAR:-default}`: `MODELS` (~/models), `SNAP` (target checkpoint), `DRAFTER`, `PORT` (8080), `SPEC_METHOD` (dflash), `SPEC` (7), `MAXSEQS` (8), `MAXLEN` (262144), `CHUNK` (8192), `GPU_UTIL` (0.98), `KV_MEM` (auto), `TP`/`GPUS` (auto).
6. WHEN invoked with `DRY_RUN=1`, THE launcher SHALL print the full `vllm serve` command and resolved environment WITHOUT executing it.
7. WHEN invoked with `--help`, THE launcher SHALL document every config knob, its default, and its effect.
8. THE launcher SHALL run the Container with the runtime's non-root, GPU-access conventions (podman `--replace`, keep-groups, video/render groups; `/dev/kfd` and `/dev/dri` passed through; `${MODELS}` and the libr4d cache mounted).

### Requirement 7: First Serve and Smoke Test

**User Story:** As an operator, I want to confirm the server is up and answers a chat completion on the two R9700 cards, so that I know the definition of done is met.

#### Acceptance Criteria

1. WHEN the launcher starts serving, THE server SHALL load `Qwen3.8-27B-MXFP4-mtpfp8` across both R9700 at TP=2 and SHALL begin listening on `${PORT:-8080}`.
2. WHEN `GET /health` is called after startup, THE server SHALL return HTTP 200.
3. WHEN `GET /v1/models` is called, THE server SHALL list the served MXFP4 model id.
4. WHEN a `POST /v1/chat/completions` request is sent, THE server SHALL return an OpenAI-compatible completion whose content is non-empty and contains no NaN token.
5. WHEN startup logs are inspected, THE logs SHALL report TP=2, native MXFP4 (quant_method read from config), the pinned libr4d in use, and per-GPU weight load in GiB.
6. IF `POST /v1/chat/completions` returns NaN or empty content, THEN THE smoke test SHALL report failure and SHALL reference the pinned libr4d build (Requirement 5) as the probable cause.
7. THE smoke test SHALL be a single documented `curl` command paired with its expected response shape, runnable without reference to any external document.

### Requirement 8: Image Strategy Decision (Launcher Path vs Build-From-Source)

**User Story:** As a maintainer, I want the image strategy recorded as a first-class decision so that operators know why the launcher path is default and when build-from-source applies.

#### Acceptance Criteria

1. THE design SHALL record two options: (A) pull the pinned `stilldeadcode/vllm-radiance:0.9.3` and patch at start (fast path, matches upstream), and (B) build from source via the 4-stage Dockerfile (builder / rocmprune / assemble / final; full control; hours-long compile).
2. THE design SHALL recommend option A for reaching first-serve and SHALL make it the default path in setup and serve scripts.
3. THE design SHALL document option B as a fallback / future work only (e.g. upstream image unavailable, need a different vLLM/ROCm pin), and SHALL keep it out of the first-serve critical path.
4. THE design SHALL state the tradeoffs (speed and reproducibility of A vs control and independence of B) so the choice is defensible.

### Requirement 9: KV Cache Calibration (Post-First-Serve Optimization)

**User Story:** As an operator who has reached first-serve, I want a calibrated KV cache size pinned for this hardware so that context and concurrency are maximised without OOM.

#### Acceptance Criteria

1. THE KV calibration SHALL be an optimization performed AFTER first-serve, and SHALL NOT be on the critical path to the definition of done.
2. WHEN calibration runs, THE calibrate step SHALL determine a safe KV cache size for the hardware signature `2x7551-32624` and the batch shape (`MAXSEQS`, `MAXLEN`, `CHUNK`).
3. WHEN calibration completes, THE result SHALL be written to `kv-profiles.tsv` keyed on hardware signature and batch shape.
4. WHEN the launcher runs with `KV_MEM=auto`, THE launcher SHALL look up `kv-profiles.tsv` for a profile matching the current hardware signature and batch shape and SHALL use that profile's KV cache size; IF no matching profile exists, THEN THE launcher SHALL delegate KV sizing to vLLM's default auto behaviour.
5. WHEN calibration is re-run, THE calibrate step SHALL update the matching profile in `kv-profiles.tsv` in place without requiring a fresh clone.

### Requirement 10: Standalone Repo Integrity

**User Story:** As a maintainer, I want this repo to stand alone so that it does not depend on `rocm_docker` or share its code.

#### Acceptance Criteria

1. THE repo SHALL contain its own copy of every script, config, and template needed to reach first-serve; it SHALL NOT source files from `rocm_docker` or any sibling repo.
2. THE repo SHALL borrow only hardware and operational conventions (device selection, podman-preferred, non-root, interactive-before-automated) as documented, not code.
3. THE repo SHALL be runnable from a fresh clone with no assumption that `rocm_docker` is present.
4. THE README SHALL document build, run, and test from a clean clone.
