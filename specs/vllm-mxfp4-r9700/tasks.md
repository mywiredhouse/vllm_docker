# Implementation Plan: vLLM MXFP4 on 2x R9700 (gfx1201)

## Overview

Ordered to reach a first successful `vllm serve` answering `/v1/chat/completions` on two R9700 (TP=2) with the MXFP4 model, at minimum risk. Steps 1–7 are the critical path to the definition of done. Step 8 (KV calibration) is a post-first-serve optimization. Steps 9–10 are repo hygiene and the documented fallback.

Path: **Option A** (pull pinned upstream image, patch at start). Build-from-source (Option B) is fallback / future work.

Target only: **MXFP4 Qwen3.8-27B on 2x R9700 at TP=2.** Out of scope: TP≠2, ParoQuant, uncensored variant, build-from-source.

## Phase 1: Reach First Serve (critical path)

- [ ] 1. Host preflight + GPU/TP detection
  - **Files:** `setup-mxfp4.sh` (preflight section), `gpu-detect.sh`
  - **Do:**
    - Verify `/dev/kfd` and `/dev/dri` exist; fail with actionable message if missing
    - Detect runtime (podman preferred, docker fallback); report which
    - Read sysfs `mem_info_vram_total` per amdgpu render node; exclude < 8192 MiB (skip iGPU)
    - Identify 2x R9700 (`0x7551`, sig `2x7551-32624`); derive TP as largest of 8/4/2/1 the usable cards fill
    - Enforce TP divides `num_attention_heads=24` AND `linear_num_key_heads=16` AND `linear_num_value_heads=48` → TP=2; refuse otherwise
    - Restrict device selection to `HIP_VISIBLE_DEVICES=0,3` on the mixed box
    - Report free disk vs ~60 GiB budget; warn if short
  - **Verify:** `./gpu-detect.sh` prints `TP=2`, sig `2x7551-32624`, and exactly two usable R9700; preflight passes on the dev box and fails cleanly when `/dev/kfd` is hidden
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [ ] 2. Image acquisition (launcher path)
  - **File:** `setup-mxfp4.sh` (image section)
  - **Do:**
    - Pull pinned `stilldeadcode/vllm-radiance:0.9.3` with the selected runtime
    - Never pull `latest`/unpinned; skip pull if image already present (idempotent)
    - Record the image digest for reproducibility
  - **Verify:** `podman image exists stilldeadcode/vllm-radiance:0.9.3` (or `docker image inspect`) returns success; digest recorded; re-run skips the pull
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.2_

- [ ] 3. Checkpoint download + fp8 MTP rewrite
  - **Files:** `setup-mxfp4.sh` (checkpoint section), `fp8_mtp.py`
  - **Do:**
    - Download `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (~19 GiB) into `${MODELS:-~/models}` (skip if present)
    - Run `fp8_mtp.py` **inside the container** to requantize the `mtp.*` head to fp8 and re-declare it by module-name → `Qwen3.8-27B-MXFP4-mtpfp8` (~15 min; skip if output exists)
    - No host Python/ROCm/HF CLI
  - **Why:** source checkpoint's bf16 MTP head is excluded by tensor-name; vLLM applies mxfp4 to bf16 and load dies. Rewrite is mandatory before serve.
  - **Verify:** `${MODELS}/Qwen3.8-27B-MXFP4-mtpfp8/config.json` declares mxfp4 for the body and fp8 for the MTP head; re-run skips the rewrite
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 4. Drafter download (dflash)
  - **File:** `setup-mxfp4.sh` (drafter section)
  - **Do:**
    - When `SPEC_METHOD=dflash` (default): download `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (~2 GiB) into `${MODELS}` (skip if present)
    - When `SPEC_METHOD=mtp`: use in-target head, do not download the drafter
    - Ensure `DRAFTER` knob resolves to the downloaded path
  - **Verify:** with default `SPEC_METHOD`, `${MODELS}/Qwen3.8-27B-DFlash2-FP8` present; with `SPEC_METHOD=mtp`, no drafter downloaded; re-run skips download
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. libr4d kernel build (pinned)
  - **Files:** `setup-mxfp4.sh` and/or `serve-mxfp4.sh` (kernel section)
  - **Do:**
    - With `AUTO_R4D=1` (default), build `libr4d` pinned at `R4D_PIN=b9e42ab` for gfx1201 inside the container
    - Cache at `~/.cache/radiance-libr4d`; reuse cache on subsequent runs
    - Never fall back to stock libr4d silently (stock NaNs the gated-delta-net); fail loudly on build failure
  - **Why:** load-bearing — stock kernels produce NaNs on this model.
  - **Verify:** `~/.cache/radiance-libr4d` populated with the pinned build; re-run reuses cache; forced build failure exits non-zero with a clear message
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6. Launcher with detected TP=2 and fixed serving flags
  - **Files:** `serve-mxfp4.sh`, `qwen-fixed-v22.3.jinja`
  - **Do:**
    - Set `RADIANCE_MXFP4=1` (native MXFP4 from `config.json`); drop `--quantization`
    - Set `RADIANCE_MXFP4_W4A8=1` by default (W4A8 fp8-WMMA GEMM; more accurate than declared W4A4, 1.47–2.26x vs aiter); document the numerics change
    - Use detected TP=2 and `HIP_VISIBLE_DEVICES=0,3`
    - Pass fixed flags `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` and `--chat-template qwen-fixed-v22.3.jinja`
    - Expose `${VAR:-default}` knobs: `MODELS`, `SNAP`, `DRAFTER`, `PORT` (8080), `SPEC_METHOD` (dflash), `SPEC` (7), `MAXSEQS` (8), `MAXLEN` (262144), `CHUNK` (8192), `GPU_UTIL` (0.98), `KV_MEM` (auto), `TP`/`GPUS` (auto)
    - Implement `--help` (documents every knob) and `DRY_RUN=1` (print command + resolved env, execute nothing)
    - Run the container with podman `--replace`, keep-groups, non-root video/render, `/dev/kfd` + `/dev/dri`, mount `${MODELS}` and the libr4d cache
    - Verify pinned libr4d cache present before launch; fail if missing
  - **Verify:** `DRY_RUN=1 ./serve-mxfp4.sh` prints a command with `--tensor-parallel-size 2`, no `--quantization`, the three fixed parser flags, the chat template, and `RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1`; `./serve-mxfp4.sh --help` lists all knobs and defaults
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 8.2_

- [ ] 7. First serve + /health + /v1/chat/completions smoke test  ← DEFINITION OF DONE
  - **Files:** `serve-mxfp4.sh`, `TEST_PLAYBOOK.md`
  - **Do:**
    - Launch serving; load `Qwen3.8-27B-MXFP4-mtpfp8` across both R9700 at TP=2; listen on `${PORT:-8080}`
    - Confirm startup logs show TP=2, native MXFP4 (quant_method from config), pinned libr4d in use, ~9.4 GiB/GPU weights
    - `GET /health` → 200
    - `GET /v1/models` → served model id
    - `POST /v1/chat/completions` → well-formed OpenAI-compatible response, non-empty, non-NaN
    - If NaN/empty: treat as libr4d failure, re-verify task 5
    - Record the exact smoke-test `curl` in TEST_PLAYBOOK.md with expected response shape
  - **Verify:**
    ```bash
    curl -s http://localhost:8080/health -o /dev/null -w '%{http_code}\n'   # expect 200
    curl -s http://localhost:8080/v1/models | jq '.data[].id'                # expect served model id
    curl -s http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.8-27B-MXFP4-mtpfp8","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
      | jq '.choices[0].message.content'                                     # expect short non-empty non-NaN string
    ```
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

## Phase 2: Optimization (after first serve)

- [ ] 8. KV cache calibration
  - **Files:** `calibrate-kv.sh`, `kv-profiles.tsv`
  - **Do:**
    - Determine a safe KV cache size for hardware sig `2x7551-32624` and batch shape (`MAXSEQS`/`MAXLEN`/`CHUNK`)
    - Write the profile to `kv-profiles.tsv` keyed on sig + batch shape
    - Make `KV_MEM=auto` look up the profile; conservative fallback if no match
    - Re-runnable without a fresh clone
  - **Note:** NOT on the critical path to definition of done. Reference target: ~943,581 KV tokens at 262,144 context.
  - **Verify:** `calibrate-kv.sh` writes/updates a row; a subsequent `KV_MEM=auto` serve logs the profile value; re-run updates in place
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

## Phase 3: Repo Hygiene and Fallback

- [ ] 9. Standalone integrity + README (clean-clone build/run/test)
  - **Files:** `README.md`, all scripts
  - **Do:**
    - Confirm the repo contains its own copy of every script/config/template; no sourcing from `rocm_docker` or siblings
    - Document that only hardware/operational conventions are borrowed, not code
    - README: clean-clone quickstart (setup → serve → smoke test), config knobs, hardware context, and the image-strategy decision (A default, B fallback)
  - **Verify:** `grep -R "rocm_docker" .` returns no code dependency; a clean clone runs `setup-mxfp4.sh` then `serve-mxfp4.sh` with no sibling-repo assumption
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 10. Document build-from-source as fallback (Option B, future work)
  - **Files:** `Dockerfile`, `README.md`
  - **Do:**
    - Record the 4-stage Dockerfile (builder / rocmprune / assemble / final) as a documented fallback
    - README states A is default and why; B applies only when the upstream image is unavailable or a different pin is needed; B is off the first-serve critical path
    - Keep the tradeoff table (time, reproducibility, control, registry dependency)
  - **Note:** Implementation of B is out of scope for this spec. This task documents the decision only.
  - **Verify:** README image-strategy section names both options, recommends A, and marks B fallback/future work
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

## Notes

- Steps 1–7 are strictly ordered — each depends on the prior. Do not reorder.
- The two fragile facts gate first-serve: fp8 MTP rewrite (task 3) and pinned libr4d (task 5). A NaN/empty completion at task 7 points to task 5 first.
- No image build is required to reach first-serve. Task 10 documents build-from-source but does not implement it.
- All setup steps are idempotent — re-running after partial setup skips completed work.
- Out of scope (do not implement): TP=3 or any TP≠2, ParoQuant, the uncensored checkpoint variant, and build-from-source (Option B beyond documentation).
