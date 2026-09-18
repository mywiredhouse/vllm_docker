# MXFP4 verification playbook

This repository targets a Fedora host with `/dev/kfd`, `/dev/dri`, and exactly two
R9700 cards (gfx1201, device `0x7551`). Run commands from the repository root.
The upstream image and the model rewrite are large; do not run the setup while
the model directory has less than the documented ~60 GiB budget.

## 1. Preflight and setup

```bash
./setup-mxfp4.sh
./setup-mxfp4.sh                 # idempotency check; completed steps must skip
```

Expected detection output includes `TP=2`, `HIP_VISIBLE_DEVICES=0,3`,
`signature=2x7551-32624`, and exactly two usable R9700 GPUs. The setup records the
image ID in `${MODELS}/.vllm-radiance-image-digest` and writes the pinned kernel
marker `${HOME}/.cache/radiance-libr4d/.pin` containing `b9e42ab`.

Verify the rewritten metadata inside the pinned image:

```bash
podman run --rm -v "${MODELS:-$HOME/models}:/models:ro" \
  stilldeadcode/vllm-radiance:0.9.3 \
  python /workspace/fp8_mtp.py --verify /models/Qwen3.8-27B-MXFP4-mtpfp8
```

## 2. Inspect the exact launch

```bash
DRY_RUN=1 ./serve-mxfp4.sh
```

The output must contain `--tensor-parallel-size 2`, the three parser flags,
`qwen-fixed-v22.3.jinja`, `RADIANCE_MXFP4=1`, and
`RADIANCE_MXFP4_W4A8=1`. It must not contain `--quantization` and must state
that no container was started.

## 3. First serve

```bash
./serve-mxfp4.sh
```

Startup logs should show TP=2, native MXFP4 loaded from `config.json`, the
pinned libr4d cache, and approximately 9.4 GiB of weights per GPU. If loading
fails or output contains NaN, stop the server and verify the libr4d marker before
trying anything else.

## 4. HTTP checks

```bash
curl -s http://localhost:8080/health -o /dev/null -w '%{http_code}\n'
curl -s http://localhost:8080/v1/models | jq '.data[].id'
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-MXFP4-mtpfp8","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
  | jq '.choices[0].message.content'
```

Expected results are HTTP `200`, model id `Qwen3.8-27B-MXFP4-mtpfp8`, and a
short non-empty string without `NaN` in the completion content. The response
shape is an OpenAI-compatible object with `choices[0].message.content`. Empty
or NaN content is a failed smoke test and points first to a missing or incorrect
pinned libr4d build.

## 5. Optional KV calibration

Calibration is after first serve and is not required for first-serve:

```bash
KV_MEM_VALUE=8589934592 KV_TOKENS=943581 ./calibrate-kv.sh
KV_MEM=auto DRY_RUN=1 ./serve-mxfp4.sh
```

The calibration key is hardware signature plus `MAXSEQS`, `MAXLEN`, and `CHUNK`.
Repeated calibration updates the existing row instead of adding a duplicate.
