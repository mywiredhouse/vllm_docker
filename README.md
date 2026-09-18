# vLLM MXFP4 on 2× AMD R9700

Standalone container launcher for **Qwen3.8-27B native MXFP4** on exactly two
AMD Radeon AI PRO R9700 GPUs (gfx1201, TP=2). It borrows the hardware and
podman conventions from `rocm_docker`, but does not source code or require that
repository.

## Quickstart from a clean clone

The host needs Fedora/amdgpu with `/dev/kfd` and `/dev/dri`, podman (preferred)
or docker, and about 60 GiB free in the model filesystem. Host ROCm, Python,
and the Hugging Face CLI are not required; downloads, the MTP rewrite, and
libr4d build run inside the pinned image.

```bash
git clone <repository-url> vllm_docker
cd vllm_docker
./setup-mxfp4.sh
DRY_RUN=1 ./serve-mxfp4.sh
./serve-mxfp4.sh
```

`setup-mxfp4.sh` is idempotent. It detects the two `0x7551` cards, excludes
sub-8-GiB render nodes, pulls `stilldeadcode/vllm-radiance:0.9.3`, downloads
the source checkpoint, creates `Qwen3.8-27B-MXFP4-mtpfp8`, downloads the dflash
drafter by default, and builds libr4d at `b9e42ab`. It records the image ID at
`${MODELS}/.vllm-radiance-image-digest` and the kernel pin at
`~/.cache/radiance-libr4d/.pin`.

The rewrite is mandatory. The source checkpoint's bf16 MTP head is excluded by
tensor name and causes vLLM to apply MXFP4 to it. The served checkpoint declares
an FP8 `mtp` module while retaining MXFP4 for the body.

## Hugging Face cache handling

Setup checks the host Hugging Face Hub cache (`HF_HOME`, default
`~/.cache/huggingface`) before downloading either exact required repository. A
matching cached snapshot is staged into `MODELS` with symlinks dereferenced, so
subsequent setup and serving do not depend on the cache remaining unchanged.
Override the cache location with `HF_HOME` or `HF_HUB_CACHE`.

The supplied cache entry can be reported explicitly:

```bash
HF_CACHE_REPO=tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8 ./setup-mxfp4.sh
```

That repository is distinct from the fixed target
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and the fixed dflash drafter
`tcclaviger/Qwen3.8-27B-DFlash2-FP8`. Setup reports its presence but does not
substitute it. The mandatory MTP rewrite and fixed served checkpoint remain
unchanged.

## Smoke test

See [TEST_PLAYBOOK.md](TEST_PLAYBOOK.md) for setup verification and failure
handling. After startup:

```bash
curl -s http://localhost:8080/health -o /dev/null -w '%{http_code}\n'
curl -s http://localhost:8080/v1/models | jq '.data[].id'
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-MXFP4-mtpfp8","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
  | python3 smoke_validate.py
```

The expected health status is `200`, the model id is
`Qwen3.8-27B-MXFP4-mtpfp8`, and the validator prints a non-empty completion
without `NaN`. Empty or NaN output means the pinned libr4d build must be
checked first.

## Configuration knobs

All knobs use shell `${VAR:-default}` resolution. `MODELS=~/models`,
`SNAP=Qwen3.8-27B-MXFP4-mtpfp8`, `DRAFTER=Qwen3.8-27B-DFlash2-FP8`,
`PORT=8080`, `SPEC_METHOD=dflash`, `SPEC=7`, `MAXSEQS=8`, `MAXLEN=262144`,
`CHUNK=8192`, `GPU_UTIL=0.98`, `KV_MEM=auto`, and `TP`/`GPUS=auto` are the
launcher defaults. `SPEC_METHOD=mtp` uses the rewritten in-target MTP head and
skips the separate drafter download. `KV_MEM=auto` uses a matching row in
`kv-profiles.tsv`, or leaves sizing to vLLM when no row matches.

`RADIANCE_MXFP4=1` is fixed and means the quantization method comes from
`config.json`; the launcher intentionally has no `--quantization` flag.
`RADIANCE_MXFP4_W4A8=1` is enabled by default. This selects fp8-WMMA W4A8
numerics rather than the declared W4A4 path; it is more accurate and is
reported as 1.47–2.26× faster than aiter in the reference stack.

## Image strategy

Option A is the default and the only first-serve critical path: pull the pinned
`stilldeadcode/vllm-radiance:0.9.3` image and apply the MTP rewrite and pinned
libr4d cache at setup time. It reaches first serve in minutes, matches the
upstream Radiance stack, and is reproducible through the tag plus recorded image
ID. It depends on the upstream registry and gives less control over ROCm/vLLM
versions.

Option B is a documented fallback/future-work path in [Dockerfile](Dockerfile).
It describes four stages: `builder`, `rocmprune`, `assemble`, and `final`. It
applies only when the upstream image is unavailable or a different pin is
needed. It is deliberately off the first-serve path and is not implemented as
a replacement for Option A.

| Dimension | A: pull + patch | B: build from source |
|---|---|---|
| Time | minutes | hours |
| Reproducibility | pinned tag + recorded ID | pinned Dockerfile/dependencies |
| Version control | lower | higher |
| Registry dependency | yes | no after build |
| Critical path | yes | no, fallback only |

`docker-compose.yml` is a convenience wrapper. `serve-mxfp4.sh` is the primary
launcher because it performs detection, pinned-cache validation, fixed flag
assembly, and `DRY_RUN=1` inspection.

## Tests

The property suite uses Hypothesis and does not require a GPU:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest -q
```

The tests cover TP divisibility, VRAM filtering, idempotent setup behavior,
checkpoint and libr4d invariants, command flags, dry-run non-execution, smoke
content validation, and KV profile upserts.
