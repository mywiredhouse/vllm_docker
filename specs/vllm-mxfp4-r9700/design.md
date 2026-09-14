# Design Document: vLLM MXFP4 on 2x R9700 (gfx1201)

## Overview

A standalone launcher repo that runs Qwen3.8-27B in native 4-bit MXFP4 on exactly two AMD Radeon AI PRO R9700 GPUs (gfx1201 / RDNA4) at tensor-parallel = 2. It is built around the upstream `radiance-vllm-mxfp4` reference stack: a prebuilt image is pulled, checkpoints are downloaded and rewritten, gfx1201 kernels are built inside the image on first run, and a launcher assembles the `vllm serve` command. No image build is required to reach first-serve.

Definition of done: from a fresh clone, an operator reaches a first successful `vllm serve` answering `/v1/chat/completions` on the two R9700 cards with the MXFP4 model.

The repo borrows hardware and operational conventions from `rocm_docker` (Fedora host, ROCm-in-image, dual-R9700 selection `HIP_VISIBLE_DEVICES=0,3`, podman-preferred rootless non-root user, interactive-before-automated discipline) but shares no code with it.

## Key Design Decision: Image Strategy

This is the primary architectural decision. Two options:

### Option A — Pull pinned upstream image, patch at start (RECOMMENDED, DEFAULT)

Pull `stilldeadcode/vllm-radiance:0.9.3` (ROCm + PyTorch + Triton + AITER + vLLM compiled for gfx1201 only). Apply patches and build the pinned `libr4d` kernels at container start. This matches upstream `radiance-vllm-mxfp4`, which is a launcher — not an image build.

- **Pros:** minutes to first pull vs hours to compile; known-good vLLM/ROCm/Triton/AITER pin; reproducible via pinned tag + recorded digest; matches upstream exactly, so upstream fixes and kernel pins apply directly.
- **Cons:** depends on the upstream registry being available; less control over the vLLM/ROCm versions than a from-source build.
- **Decision:** default path for setup and serve. This is the only path on the critical route to first-serve.

### Option B — Build from source via 4-stage Dockerfile (FALLBACK / FUTURE WORK)

Build the image locally with a 4-stage Dockerfile: `builder` (compile vLLM/Triton/AITER for gfx1201) → `rocmprune` (strip unused ROCm arches/artifacts) → `assemble` (lay out runtime) → `final` (non-root runtime image).

- **Pros:** full control over vLLM/ROCm/Triton pins; independence from the upstream registry; ability to change the gfx target or patch set.
- **Cons:** hours-long compile; larger maintenance burden; not needed to reach first-serve.
- **Decision:** documented as a fallback only (upstream image unavailable, or a different pin is required). Kept off the first-serve critical path. Out of scope for this spec's implementation.

### Tradeoff summary

| Dimension | A (pull + patch) | B (build from source) |
|-----------|------------------|-----------------------|
| Time to first-serve | minutes | hours |
| Reproducibility | pinned tag + digest | pinned Dockerfile + deps |
| Control over versions | low | high |
| Registry dependency | yes | no (after first build) |
| Matches upstream | exactly | diverges |
| First-serve critical path | yes | no |

## Two Fragile Facts the Design Must Respect

1. **MTP head must be rewritten to fp8.** The target checkpoint's `mtp.*` layers are bf16 but excluded by tensor-name, not module-name. vLLM applies mxfp4 to the bf16 head and load dies. `fp8_mtp.py` requantizes and re-declares the head, producing `Qwen3.8-27B-MXFP4-mtpfp8`. Mandatory before serve.
2. **libr4d must be the pinned build.** The image's stock `libr4d` NaNs this model's gated-delta-net. `AUTO_R4D=1` with `R4D_PIN=b9e42ab` builds the correct kernels, cached at `~/.cache/radiance-libr4d`. Load-bearing — a NaN/empty completion at smoke-test time points here first.

## Architecture

### File Structure

```
vllm_docker/
├── setup-mxfp4.sh          # Idempotent: host preflight, image pull, checkpoints, rewrite, drafter, kernels
├── serve-mxfp4.sh          # Launcher: detected TP, native MXFP4, fixed flags; --help, DRY_RUN=1
├── gpu-detect.sh           # sysfs VRAM read, iGPU exclusion, usable-count + TP derivation
├── calibrate-kv.sh         # Post-first-serve KV sizing for the hardware signature
├── kv-profiles.tsv         # KV cache profiles keyed on hardware sig + batch shape
├── fp8_mtp.py              # MTP head fp8 rewrite (runs inside the container)
├── qwen-fixed-v22.3.jinja  # Repo chat template (fixed serving flag)
├── docker-compose.yml      # Convenience wrapper for the launcher path
├── Dockerfile              # Option B build-from-source (fallback / future work, not on critical path)
├── TEST_PLAYBOOK.md        # Verification procedures (preflight → smoke test)
├── README.md               # Clean-clone build/run/test + image-strategy decision
└── LICENSE
```

### Setup Flow (Option A, idempotent)

```
Operator runs ./setup-mxfp4.sh
  → host preflight: /dev/kfd, /dev/dri, runtime (podman|docker), disk budget
  → gpu-detect.sh: read sysfs mem_info_vram_total, exclude <8192 MiB, find 2x R9700 → TP=2
  → pull stilldeadcode/vllm-radiance:0.9.3 (skip if present), record digest
  → download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 into ${MODELS} (skip if present)
  → run fp8_mtp.py in container → Qwen3.8-27B-MXFP4-mtpfp8 (skip if present)
  → if SPEC_METHOD=dflash: download tcclaviger/Qwen3.8-27B-DFlash2-FP8 (skip if present)
  → build pinned libr4d (R4D_PIN=b9e42ab) in container → cache ~/.cache/radiance-libr4d (skip if cached)
  → report ready
```

### Serve Flow

```
Operator runs ./serve-mxfp4.sh   (or DRY_RUN=1 ./serve-mxfp4.sh to inspect)
  → gpu-detect.sh → TP=2, HIP_VISIBLE_DEVICES=0,3
  → verify pinned libr4d cache present (else fail loudly)
  → resolve knobs: MODELS, SNAP, DRAFTER, PORT, SPEC_METHOD, SPEC, MAXSEQS, MAXLEN, CHUNK, GPU_UTIL, KV_MEM, TP
  → RADIANCE_MXFP4=1, RADIANCE_MXFP4_W4A8=1
  → run container (podman --replace, keep-groups, /dev/kfd + /dev/dri, mount ${MODELS} + libr4d cache)
  → exec: vllm serve <mtpfp8 checkpoint> --tensor-parallel-size 2 \
           --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
           --chat-template qwen-fixed-v22.3.jinja  [+ speculative + kv + length flags]
  → listens on ${PORT:-8080}
```

### Container Topology

```
Host                                   Container (stilldeadcode/vllm-radiance:0.9.3)
─────                                   ────────────────────────────────────────────
/dev/kfd, /dev/dri        ──────────→   GPU access (R9700 #0, #3 via HIP_VISIBLE_DEVICES=0,3)
${MODELS:-~/models}       ──────────→   model dir (rw): source, mtpfp8, drafter
~/.cache/radiance-libr4d  ──────────→   kernel cache (rw): pinned libr4d build
qwen-fixed-v22.3.jinja    ──────────→   chat template

                                        vLLM (gfx1201) + Triton + AITER + pinned libr4d
                                        RADIANCE_MXFP4=1, RADIANCE_MXFP4_W4A8=1
                                        User: non-root (video, render groups)
                                        Serving on 0.0.0.0:${PORT} inside; mapped to host
```

## Components and Interfaces

Each component below states its interface as a contract: what it consumes (inputs / env), what it produces (outputs / side effects), and how it signals failure.

### Component 1: gpu-detect.sh

**Interface** — Consumes: sysfs render nodes, `HIP_VISIBLE_DEVICES` (optional override). Produces (stdout / exported env): usable GPU count, `TP`, `HIP_VISIBLE_DEVICES=0,3`, hardware signature `2x7551-32624`. Fails: non-zero exit with the failing divisibility constraint, or when fewer than two R9700 are usable.

Reads `mem_info_vram_total` from each amdgpu render node under sysfs, excludes devices below 8192 MiB (skips iGPU), counts usable cards, and derives TP as the largest of 8/4/2/1 the usable cards fill.

TP must divide `num_attention_heads=24` AND `linear_num_key_heads=16` AND `linear_num_value_heads=48`. Two R9700 → TP=2 (only viable TP for this hardware). Emits the hardware signature (`2x7551-32624`) for KV lookup. Restricts to the two R9700 via `HIP_VISIBLE_DEVICES=0,3` on the mixed dev box.

Illustrative constraint check:

```bash
divides() { [ $(( $1 % $2 )) -eq 0 ]; }
if divides 24 "$TP" && divides 16 "$TP" && divides 48 "$TP"; then
  echo "TP=$TP valid"
else
  echo "TP=$TP does not divide 24/16/48 — refusing" >&2; exit 1
fi
```

### Component 2: setup-mxfp4.sh

**Interface** — Consumes: `MODELS`, `SPEC_METHOD`, `AUTO_R4D`, `R4D_PIN`, selected runtime. Produces: pulled image (digest recorded), source checkpoint, `Qwen3.8-27B-MXFP4-mtpfp8`, optional drafter, cached libr4d, a `ready` report. Idempotent — each step is a no-op when its output already exists.

Idempotent orchestrator. Each step checks for existing state before acting: image present → skip pull; checkpoint present → skip download; `Qwen3.8-27B-MXFP4-mtpfp8` present → skip rewrite; drafter present → skip download; libr4d cached → skip build. Runs the rewrite and kernel build **inside the container** so the host needs no Python, ROCm, or HF CLI. Records the pulled image digest.

### Component 3: fp8_mtp.py

**Interface** — Consumes: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` on disk. Produces: `Qwen3.8-27B-MXFP4-mtpfp8` whose `config.json` declares `mxfp4` for the body and `fp8` for the MTP head. Runs inside the container. Verification is metadata-level (config declarations), not numeric.

Runs inside the container. Loads `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, requantizes the `mtp.*` head to fp8, re-declares it by module-name (so vLLM does not apply mxfp4 to a bf16 head), and writes `Qwen3.8-27B-MXFP4-mtpfp8`. ~15 min. Body stays MXFP4; only the MTP head changes. Verification is metadata-level: config declares mxfp4 for the body and fp8 for the MTP head.

### Component 4: libr4d build (inside serve/setup)

**Interface** — Consumes: `AUTO_R4D=1`, `R4D_PIN=b9e42ab`, gfx1201 target. Produces: pinned kernel build cached at `~/.cache/radiance-libr4d`. Contract: the serve script verifies the cache before launch and fails loudly if absent — it never falls back to stock kernels.

`AUTO_R4D=1` clones and builds `libr4d` pinned at `R4D_PIN=b9e42ab` for gfx1201, caching at `~/.cache/radiance-libr4d`. The serve script verifies the cache before launching and fails loudly if the pinned build is missing — never silently falls back to the stock (NaN-producing) kernels.

### Component 5: serve-mxfp4.sh

**Interface** — Consumes: config knobs (table below), detected TP, verified libr4d cache. Produces: a running container serving the mtpfp8 checkpoint on `${PORT}`, or (with `DRY_RUN=1`) the resolved command and env printed with no execution. Fails: non-zero exit if the libr4d cache is missing.

The launcher. Resolves all config knobs as `${VAR:-default}`, sets `RADIANCE_MXFP4=1` and `RADIANCE_MXFP4_W4A8=1`, uses detected TP=2, and assembles the `vllm serve` command with the fixed flags and chat template. Supports `--help` (documents every knob) and `DRY_RUN=1` (prints resolved command and env, executes nothing).

Config knobs (`${VAR:-default}`):

| Knob | Default | Effect |
|------|---------|--------|
| `MODELS` | `~/models` | Model directory (mounted rw) |
| `SNAP` | `Qwen3.8-27B-MXFP4-mtpfp8` | Served checkpoint |
| `DRAFTER` | `Qwen3.8-27B-DFlash2-FP8` | Speculative drafter (dflash) |
| `PORT` | `8080` | HTTP port |
| `SPEC_METHOD` | `dflash` | `dflash` (drafter) or `mtp` (in-target head) |
| `SPEC` | `7` | Speculative tokens |
| `MAXSEQS` | `8` | Max concurrent sequences |
| `MAXLEN` | `262144` | Max context length |
| `CHUNK` | `8192` | Chunked prefill size |
| `GPU_UTIL` | `0.98` | GPU memory utilisation |
| `KV_MEM` | `auto` | KV cache size (looks up `kv-profiles.tsv`) |
| `TP` / `GPUS` | `auto` | Overrides detection (default 2 on target) |

Illustrative command assembly (native MXFP4 → no `--quantization`):

```bash
vllm serve "${MODELS}/${SNAP}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAXLEN}" \
  --max-num-seqs "${MAXSEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-chunked-prefill --max-num-batched-tokens "${CHUNK}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template qwen-fixed-v22.3.jinja \
  --port "${PORT}"
  # speculative flags added per SPEC_METHOD (dflash → drafter model; mtp → in-target head)
```

### Component 6: serving flags and template (fixed)

**Interface** — Consumes: nothing configurable. Produces: fixed flag set appended to every `vllm serve` invocation. Not knobs.

`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` plus `qwen-fixed-v22.3.jinja` are fixed, not knobs. They match upstream and are required for correct tool-call and reasoning parsing.

### Component 7: calibrate-kv.sh + kv-profiles.tsv

**Interface** — Consumes: hardware signature, batch shape (`MAXSEQS`/`MAXLEN`/`CHUNK`), a running server. Produces / updates: one row in `kv-profiles.tsv` keyed on (hardware signature, batch shape). Consumed by the launcher when `KV_MEM=auto`. Re-runnable in place; off the first-serve critical path.

Post-first-serve optimization. Determines a safe KV cache size for the hardware signature `2x7551-32624` and the batch shape (`MAXSEQS`/`MAXLEN`/`CHUNK`), writes a row to `kv-profiles.tsv` keyed on both. With `KV_MEM=auto`, the launcher looks up a matching profile and uses it; otherwise it falls back to a conservative auto value. Re-runnable without a fresh clone. Reference target: ~943,581 KV tokens at 262,144 context on 2x R9700.

### Component 8: docker-compose.yml

**Interface** — Consumes: same device nodes, mounts, and env as `serve-mxfp4.sh`. Produces: the same running container. Optional wrapper, not the primary entry point.

Convenience wrapper for the launcher path (FP8/MXFP4 serve), passing the same device nodes, mounts, and env as `serve-mxfp4.sh`. Not required — the shell launcher is the primary entry point.

### Component 9: Dockerfile (Option B, fallback)

**Interface** — Consumes: pinned build deps. Produces: a from-source gfx1201 image (`builder` → `rocmprune` → `assemble` → `final`). Not invoked on the first-serve path.

4 stages — `builder`, `rocmprune`, `assemble`, `final` — for build-from-source. Present for completeness and future work. Not exercised on the first-serve path. Documented in README as fallback only.

## Data Models

This stack has no application database. Its persistent "data models" are the on-disk artifacts and the calibration table that the scripts read and write. Three schemas are load-bearing.

### Hardware signature

A string key identifying the GPU configuration for KV calibration and detection. Format:

```
<count>x<deviceId>-<usableVramMiB>
```

| Field | Type | Meaning | Target value |
|-------|------|---------|--------------|
| `count` | int | Number of usable in-scope cards | `2` |
| `deviceId` | hex (no `0x`) | PCI device id of the card | `7551` (R9700) |
| `usableVramMiB` | int | Usable VRAM per card, MiB, from sysfs `mem_info_vram_total` | `32624` |

Target signature: `2x7551-32624`. Produced by `gpu-detect.sh`, consumed by the launcher and `calibrate-kv.sh`.

### kv-profiles.tsv row schema

Tab-separated. One row per (hardware signature, batch shape). The composite key is `hw_sig` + `max_seqs` + `max_len` + `chunk`; `KV_MEM=auto` looks up a row on this key.

| Column | Type | Key? | Meaning |
|--------|------|------|---------|
| `hw_sig` | string | key | Hardware signature, e.g. `2x7551-32624` |
| `max_seqs` | int | key | Batch shape: `MAXSEQS` at calibration |
| `max_len` | int | key | Batch shape: `MAXLEN` at calibration |
| `chunk` | int | key | Batch shape: `CHUNK` at calibration |
| `kv_mem` | string/number | value | Calibrated safe KV cache size for this key |
| `kv_tokens` | int | value | Achieved KV tokens (reference: ~943581 at 262144 ctx) |

Re-running calibration for an existing key updates that row in place. When no row matches, the launcher delegates KV sizing to vLLM's default auto behaviour.

### Checkpoint config.json quant declarations

Each on-disk checkpoint carries a `config.json` whose quantisation declarations determine how vLLM loads it. The rewrite exists precisely to fix these declarations.

| Checkpoint | Body quant | MTP head quant | Loadable as-is |
|-----------|-----------|----------------|----------------|
| `Qwen3.8-27B-Quark-AWQ-MXFP4` (source) | `mxfp4` | bf16, excluded by tensor-name | No — vLLM applies mxfp4 to bf16 head, load dies |
| `Qwen3.8-27B-MXFP4-mtpfp8` (served) | `mxfp4` | `fp8`, declared by module-name | Yes |
| `Qwen3.8-27B-DFlash2-FP8` (drafter) | `fp8` | n/a | Yes (drafter only) |

Native MXFP4 is driven by the `quant_method` field in `config.json` (read when `RADIANCE_MXFP4=1`); the launcher therefore passes no `--quantization` flag. Rewrite verification is a check of these `config.json` fields, not a numeric comparison of weights.

## Data / Model Layout

```
${MODELS:-~/models}/
├── Qwen3.8-27B-Quark-AWQ-MXFP4/     # source, ~19 GiB (Apache-2.0)
├── Qwen3.8-27B-MXFP4-mtpfp8/        # rewritten, ~19 GiB (served checkpoint)
└── Qwen3.8-27B-DFlash2-FP8/         # drafter, ~2 GiB (dflash only)

~/.cache/radiance-libr4d/            # pinned libr4d build (R4D_PIN=b9e42ab)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

This stack is mostly shell orchestration and I/O against real hardware, so most acceptance criteria are verified by the manual playbook (see Testing Strategy). The properties below isolate the pure, input-varying logic — TP arithmetic, device filtering, idempotent orchestration, command assembly, config/kernel invariants, the smoke-test validator, and KV profile lookup — that is worth checking across many generated inputs rather than one example.

### Property 1: TP divides the head counts

*For any* candidate tensor-parallel size TP, the detection logic SHALL accept TP as valid if and only if TP divides each of `num_attention_heads=24`, `linear_num_key_heads=16`, and `linear_num_value_heads=48` (i.e. TP ∈ {1, 2, 4, 8}); on the target 2x R9700 the derived TP SHALL be exactly 2, and any non-dividing TP SHALL be refused with a non-zero exit.

**Validates: Requirements 1.4, 1.6**

### Property 2: sub-threshold GPUs are always excluded

*For any* set of detected render nodes with arbitrary reported VRAM, the usable set produced by detection SHALL contain exactly those devices reporting ≥ 8192 MiB and no device below it, so an iGPU is never counted.

**Validates: Requirements 1.3**

### Property 3: setup is idempotent

*For any* initial on-disk state, applying a setup step (image pull, checkpoint download, MTP rewrite, drafter download, libr4d build) twice SHALL leave the same final state as applying it once, and the second application SHALL perform no work.

**Validates: Requirements 1.8, 3.4, 4.3, 5.2**

### Property 4: the served checkpoint is always the mtpfp8 rewrite

*For any* resolution of the launcher config knobs, the served checkpoint path SHALL resolve to `Qwen3.8-27B-MXFP4-mtpfp8` and SHALL never resolve to the raw source `Qwen3.8-27B-Quark-AWQ-MXFP4`; the served checkpoint's `config.json` SHALL declare `mxfp4` for the body and `fp8` for the MTP head.

**Validates: Requirements 3.2, 3.3, 3.6, 7.1**

### Property 5: the pinned libr4d is always used, never the stock kernels

*For any* state of the kernel cache, the serve script SHALL launch only when the pinned `R4D_PIN=b9e42ab` build is present at `~/.cache/radiance-libr4d`; when it is absent the serve script SHALL exit non-zero and SHALL NOT launch, and it SHALL never fall back to the image's stock libr4d.

**Validates: Requirements 5.3, 5.4, 5.5**

### Property 6: native MXFP4 means no `--quantization` flag

*For any* resolution of the launcher config knobs with `RADIANCE_MXFP4=1`, the assembled `vllm serve` command SHALL contain no `--quantization` argument, and SHALL always include the fixed serving flags (`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`) and the `qwen-fixed-v22.3.jinja` template.

**Validates: Requirements 6.1, 6.4**

### Property 7: DRY_RUN prints exactly what would run, and runs nothing

*For any* resolution of the launcher config knobs, invoking the launcher with `DRY_RUN=1` SHALL print the same `vllm serve` command that a real invocation would execute and SHALL start zero containers.

**Validates: Requirements 6.6**

### Property 8: smoke-test content is non-empty and NaN-free

*For any* completion response body, the smoke-test validator SHALL accept it if and only if `choices[0].message.content` is a non-empty string containing no NaN token; an empty or NaN-bearing body SHALL be rejected and attributed to the pinned libr4d build.

**Validates: Requirements 7.4, 7.6**

### Property 9: KV profile lookup round-trips and stays unique

*For any* hardware signature and batch shape (`MAXSEQS`, `MAXLEN`, `CHUNK`), after calibration upserts a profile for that key, a `KV_MEM=auto` lookup on the same key SHALL return the written KV size, exactly one row SHALL exist for that key regardless of how many times calibration re-runs, and a lookup on an absent key SHALL delegate to vLLM's default auto behaviour.

**Validates: Requirements 9.3, 9.4, 9.5**

## Error Handling

| Error / Symptom | Cause | Resolution |
|-----------------|-------|------------|
| `/dev/kfd: No such file` | amdgpu driver not loaded / nodes not exposed | Load amdgpu; verify `/dev/kfd` and `/dev/dri` (Req 1.1) |
| Only 1 usable GPU detected | iGPU excluded, or a R9700 not visible | Check `HIP_VISIBLE_DEVICES=0,3`; confirm 2x `0x7551` present (Req 1.4) |
| `TP does not divide 24/16/48` | TP other than 2 attempted | Only TP=2 is valid for 2x R9700; refuse (Req 1.6) |
| Load dies applying mxfp4 to bf16 head | Rewrite skipped; serving source checkpoint | Run `fp8_mtp.py`; serve `Qwen3.8-27B-MXFP4-mtpfp8` (Req 3) |
| Completions return NaN / empty | Stock libr4d NaNs gated-delta-net | Rebuild pinned libr4d (`AUTO_R4D=1`, `R4D_PIN=b9e42ab`); verify cache (Req 5, Req 7.6) |
| OOM at load / KV alloc | KV cache too large for this hardware | Run `calibrate-kv.sh`; lower `MAXLEN`/`MAXSEQS`/`GPU_UTIL` (Req 9) |
| Image pull fails / registry down | Upstream unavailable | Retry; as fallback consider Option B build-from-source (Req 8.3) |
| Wrong TP or device set at serve | Detection overridden by stale env | Unset `TP`/`GPUS`/`HIP_VISIBLE_DEVICES`; re-run detection (Req 1, Req 6.3) |

## Testing Strategy

The stack requires real R9700 hardware, so the end-to-end path is verified manually via TEST_PLAYBOOK.md. Alongside it, the pure input-varying logic — the same logic the Correctness Properties describe — is covered by a small property-based test suite that needs no GPU. The two layers are complementary: property tests catch logic bugs in detection, command assembly, idempotence, and profile lookup; the playbook confirms the assembled stack actually serves on hardware.

### Property-based tests (no hardware required)

Cover the nine correctness properties. These exercise the extractable pure logic: TP divisibility, VRAM filtering, idempotent step application, checkpoint-path resolution, libr4d cache gating, command assembly, DRY_RUN fidelity, the smoke-test content validator, and KV profile lookup/upsert.

- Pick a property-based testing library for the target language (the scripts' testable logic can be exercised via a Python harness with **Hypothesis**, or a Bats/shell harness driving the functions). Do not hand-roll a generator framework.
- Each property test SHALL run a minimum of 100 iterations.
- Each test SHALL be tagged referencing its design property, format: **Feature: vllm-mxfp4-r9700, Property {number}: {property_text}**.
- Each correctness property SHALL be implemented by a single property-based test.

### Manual verification (TEST_PLAYBOOK.md, on hardware)

The playbook gives exact commands with expected outputs for the parts that touch real GPUs, the registry, and the HTTP server — none of which vary meaningfully with input and so are integration/smoke checks, not property tests:

1. Preflight: `/dev/kfd` + `/dev/dri` present, runtime detected, disk ≥ ~60 GiB, GPU detection reports 2x R9700 → TP=2
2. Setup idempotency: re-run `setup-mxfp4.sh`, confirm completed steps skip
3. Rewrite: `Qwen3.8-27B-MXFP4-mtpfp8` exists, config declares mxfp4 body + fp8 MTP head
4. Kernels: `~/.cache/radiance-libr4d` populated with pinned build
5. Dry run: `DRY_RUN=1 ./serve-mxfp4.sh` prints TP=2, native MXFP4 (no `--quantization`), fixed flags, template
6. First serve: `GET /health` → 200; `GET /v1/models` → served model id
7. Smoke test: single `curl POST /v1/chat/completions` → well-formed, non-empty, non-NaN content
8. Post-serve: `calibrate-kv.sh` writes a profile; `KV_MEM=auto` picks it up

Smoke test (illustrative):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-MXFP4-mtpfp8",
       "messages":[{"role":"user","content":"Say hello in one word."}],
       "max_tokens":16}' | jq '.choices[0].message.content'
```

Expected: a short, non-empty, non-NaN string. NaN/empty → libr4d (see Error Handling).

## References

- [radiance-vllm-mxfp4 (upstream launcher)](https://codeberg.org/ggz14/radiance-vllm-mxfp4)
- [libr4d (gfx1201 HIP kernels)](https://codeberg.org/StillDeadcode/libr4d) — pinned `b9e42ab`
- Base image: `stilldeadcode/vllm-radiance:0.9.3`
- Checkpoint: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (Apache-2.0)
- Drafter: `tcclaviger/Qwen3.8-27B-DFlash2-FP8`
- Rewritten (produced locally): `Qwen3.8-27B-MXFP4-mtpfp8`
