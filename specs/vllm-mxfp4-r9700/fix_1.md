# Implementation and Testing Plan: R4D/libr4d Integration and TP=2 Memory Failure

## Scope

Correct the AMD R9700 R4D/libr4d integration in `vllm_docker`, prove that the intended patched `r4d.so` is the active Python module, and then recalibrate the TP=2 memory configuration if the existing startup failure persists.

The current evidence establishes two separate facts:

1. The local launcher builds raw `libr4d@b9e42ab` but does not apply the Radiance extras patch or replace the image's active `r4d.so`.
2. The latest container failure reaches R4D/GDN initialization and then fails with `Memory critical error ... Reason: Memory in use` during TP worker startup. The R4D replacement mismatch is verified; its causal relationship to the memory failure must be tested, not assumed.

The prior architecture-inspection traceback in `serve-live.log` is stale or from a different launch mode and is not the primary failure for the current dflash container run.

## Fixed component identity

Hold these values constant during implementation and validation:

```text
Image: stilldeadcode/vllm-radiance:0.9.3
Image digest: sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
Radiance source: 3368c488916644e5730c55a519260c51e16affb0
libr4d base: b9e42ab7202f53a3bc13d415f5d41481f9ca311b
libr4d identity: b9e42ab-rx6
GPU architecture: gfx1201
GPU selection: HIP_VISIBLE_DEVICES=0,3
Tensor parallelism: TP=2
Checkpoint: Qwen3.8-27B-MXFP4-mtpfp8
```

`rx6` is the current Radiance reference identity. It is a reproducibility identity, not a proven fix for the TP=2 memory failure. Do not mix it with the unpatched raw `b9e42ab` cache or another RX generation.

The `stew675/llama-cpp-rdna-boosts` and `sixvolts/llama-halo-hybrid` repositories are llama.cpp references only. Their source and patches must not be imported into this vLLM repository.

## Phase 1: Make the R4D build reproducible

### Files

- `vllm_docker/setup-mxfp4.sh`
- New vendored `vllm_docker/r4d_radiance_extras.patch`
- `vllm_docker/README.md`
- `vllm_docker/TEST_PLAYBOOK.md`

### Changes

1. Record the complete image, Radiance, libr4d, patch, and binary identity instead of recording only `.pin=b9e42ab`.
2. Apply the exact `r4d_radiance_extras.patch` from the pinned Radiance source. The patch adds the fused GDN update and additional R4D attention/all-reduce units.
3. Build libr4d in a temporary directory and publish only after a successful build:

```text
.r4d-cache/.build/
.r4d-cache/b9e42ab-rx6/r4d.so
.r4d-cache/b9e42ab-rx6/identity
```

4. The identity file must record:

```text
radiance_source=3368c488916644e5730c55a519260c51e16affb0
libr4d_base=b9e42ab7202f53a3bc13d415f5d41481f9ca311b
r4d_key=b9e42ab-rx6
patch_sha256=<exact patch hash>
r4d_so_sha256=<built library hash>
image_digest=sha256:456942...
```

5. Reject a cache when the raw unpatched `b9e42ab` identity is requested for production, the patch hash is missing or wrong, `r4d.so` is absent, or the cache identity does not match the image/source combination.
6. Keep the existing raw cache available for rollback, but do not accept it as the production cache.

### Gate

Setup must fail before serving if the requested R4D identity is incomplete or mismatched. Cache publication must be atomic so an interrupted build cannot appear valid.

## Phase 2: Replace the active Python R4D module

### Files

- New `vllm_docker/r4d-container-entrypoint.sh`
- `vllm_docker/serve-mxfp4.sh`
- `vllm_docker/docker-compose.yml`

### Changes

The current launcher mounts a cache directory but never replaces the image's embedded module:

```text
/opt/vllm/lib/python3.12/site-packages/r4d.so
```

Add a shared container wrapper used by both launcher paths. It must:

1. Verify `/r4d/r4d.so` exists.
2. Verify its SHA-256 against the expected identity.
3. Copy it to `/opt/vllm/lib/python3.12/site-packages/r4d.so`.
4. Verify the copied module's SHA-256.
5. Import `r4d` and verify the required patched symbols, including the fused GDN entry point.
6. Print the active R4D identity and kernel list.
7. Execute `/opt/radiance_entrypoint.sh` with the original vLLM arguments.

Do not set:

```text
LD_LIBRARY_PATH=/r4d-cache
```

That can hide the image's ROCm library path. Use `/r4d` for the Python extension replacement and preserve the image's ROCm loader configuration.

### Gate

The startup log must prove that the active site-packages module has the expected hash and patched kernel symbols. A missing or mismatched replacement must abort before vLLM starts.

## Phase 3: Unify launcher and Compose behavior

### Files

- `vllm_docker/serve-mxfp4.sh`
- `vllm_docker/docker-compose.yml`
- `vllm_docker/r4d-container-entrypoint.sh`

Both paths must pass the same values:

```text
RADIANCE_USE_R4D=1
RADIANCE_USE_R4D_AR=1
RADIANCE_USE_R4D_AR_QUANT=1
R4D_PIN=b9e42ab
R4D_KEY=b9e42ab-rx6
RADIANCE_MXFP4=1
RADIANCE_MXFP4_W4A8=1
```

Both paths must:

- mount the validated cache at `/r4d`;
- use the shared container wrapper;
- use the same checkpoint and chat template;
- use the same TP and memory arguments;
- provide sufficient shared memory;
- print the same identity information.

`serve-mxfp4.sh` remains the primary command builder. Compose must not silently implement a divergent R4D path.

### Gate

Rendered launcher and Compose commands must be equivalent for the R4D identity, environment, mounts, model, TP, and memory settings.

## Phase 4: Add TP=2 memory calibration

The current launch uses:

```text
MAXLEN=262144
MAXSEQS=8
CHUNK=8192
GPU_UTIL=0.98
```

`kv-profiles.tsv` contains only a header, so there is no tested TP=2 KV-memory profile.

Do not copy the single-R9700 reference values directly. Add a TP=2 calibration mode keyed by:

```text
hardware_signature
image_digest
r4d_key
tensor_parallel_size
max_seqs
max_model_len
chunk
spec_method
spec_tokens
gpu_memory_utilization
```

Use `KV_MEM=auto` or an explicit profile mode that genuinely permits vLLM profiling. Do not interpret `KV_MEM=0` as a byte value unless the launcher explicitly defines that behavior.

If the corrected R4D module is active and the original launch still fails, vary one memory dimension at a time:

1. `GPU_UTIL`: `0.90`, `0.95`, `0.98`.
2. `MAXLEN`: conservative calibrated value, then `262144`.
3. `MAXSEQS`: `2`, `4`, `8`.
4. `CHUNK`: `2048`, `4096`, `8192`.

The single-R9700 reference values of `MAXLEN=204800`, `MAXSEQS=2`, and `CHUNK=2048` are calibration guidance only, not TP=2 defaults.

### Gate

A memory profile is valid only when it is tied to the complete TP=2 hardware, R4D, image, model, and batch-shape identity. Intermittent configurations are not production-safe.

## Testing plan

### 1. Static and unit tests

Extend `vllm_docker/tests/test_properties.py` to cover:

- wrong cache identity rejection;
- rejection of raw `b9e42ab` when `b9e42ab-rx6` is required;
- patch and binary hash validation;
- atomic cache publication behavior;
- required R4D environment variables;
- `/r4d` mount presence;
- shared wrapper presence;
- absence of `LD_LIBRARY_PATH=/r4d-cache`;
- active module replacement command;
- launcher and Compose command parity;
- rewritten MTP checkpoint enforcement;
- absence of a `--quantization` flag.

Run:

```bash
./.venv/bin/pytest -q
bash -n setup-mxfp4.sh serve-mxfp4.sh r4d-container-entrypoint.sh
shellcheck setup-mxfp4.sh serve-mxfp4.sh r4d-container-entrypoint.sh
```

### 2. Container-level R4D provenance test

Without starting the model server:

1. Start the pinned image with the built `/r4d` directory.
2. Run the shared wrapper.
3. Verify the active site-packages `r4d.so` hash equals the cache hash.
4. Import `r4d`.
5. Assert the patched kernel symbols are present.
6. Verify that a wrong hash or cache identity aborts before vLLM starts.

This proves that the intended R4D binary is active rather than merely mounted.

### 3. Exact-launch rerun

After Phases 1–3, rerun the existing failure shape unchanged:

```text
TP=2
HIP_VISIBLE_DEVICES=0,3
MAXLEN=262144
MAXSEQS=8
CHUNK=8192
GPU_UTIL=0.98
SPEC_METHOD=dflash
SPEC=7
```

Required startup evidence:

```text
R4D identity: b9e42ab-rx6
using patched r4d.so
patched kernel list present
Resolved architecture: Qwen3_5ForConditionalGeneration
Resolved architecture: DFlash2DraftModel
```

Classify the result as successful startup, continued GPU memory failure, an invalid patched module, or a newly exposed error. Do not claim that `rx6` fixes the memory issue unless this controlled rerun demonstrates it.

### 4. TP=2 memory calibration

Only if the exact launch still fails:

- hold image, source, model, TP, GPU selection, and R4D identity fixed;
- vary only one memory parameter at a time;
- classify each cell as pass, intermittent, or fail;
- discard intermittent configurations.

A successful screening cell must:

- initialize all TP workers;
- avoid the KFD `Memory critical error`;
- return HTTP `200` from `/health`;
- list the expected model from `/v1/models`;
- produce non-empty, NaN-free output.

Use three screening starts per cell. Confirm any candidate with five cold starts before recommending it.

The prior communication diagnostics are not repeated in this first implementation plan. Communication-path experiments remain follow-up work if active-module correction and memory calibration do not explain the failure.

### 5. HTTP correctness checks

For every successful candidate:

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/v1/models
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-MXFP4-mtpfp8","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
  | python3 smoke_validate.py
```

The completion must be non-empty and NaN-free.

## Rollback and recovery

Keep the current image digest, old launcher state, and raw `b9e42ab` cache intact while introducing `b9e42ab-rx6`.

Rollback must restore the previous repository revision and cache selection. It must not silently fall back to the image's stock R4D library.

Stock R4D may be used for diagnostic comparison only. It is not an acceptable production fallback because the reference documents correctness and NaN failures with the old embedded kernel.

## Follow-up work

Keep these outside the first implementation:

- rebuilding the entire vLLM image from source;
- importing code from either llama.cpp repository;
- repeating communication/RCCL diagnostics already attempted;
- broad TP=1 versus TP=2 research;
- support for additional GPU counts;
- replacing the pinned image with a newer vLLM/ROCm stack;
- upstreaming Radiance patches;
- general cleanup of stale historical logs and launch artefacts.
