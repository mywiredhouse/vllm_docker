# vllm_docker

A containerised vLLM inference server for **2× AMD Radeon AI PRO R9700** (gfx1201 / RDNA4).

This repo is the vLLM counterpart to `rocm_docker` (which serves llama.cpp). It is
standalone: it borrows the hardware profile and operational conventions from
`rocm_docker` but shares no code or build dependency with it.

## Goal

Run **Qwen3.8-27B in native 4-bit MXFP4** on two R9700 cards (tensor-parallel = 2),
using the vLLM patch/kernel stack and model checkpoints identified in
[radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4).

## Status

Bootstrapping. The build-out is specified under `specs/` (requirements, design, tasks).
Nothing here serves yet.

## Target hardware

| Item | Value |
|------|-------|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB each, wave32 |
| Host OS | Fedora (amdgpu kernel driver; `/dev/kfd` + `/dev/dri`) |
| Runtime | podman (preferred) or docker |
| Device selection | `HIP_VISIBLE_DEVICES` (dual-R9700 validated as indices `0,3` on the dev box) |

## Reference stack (upstream)

The MXFP4 path is defined by radiance-vllm-mxfp4:

- Base image: `stilldeadcode/vllm-radiance` (ROCm + PyTorch + Triton + AITER + vLLM, compiled for gfx1201)
- Target checkpoint: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (needs an fp8 MTP-head rewrite)
- Drafter: `tcclaviger/Qwen3.8-27B-DFlash2-FP8` (speculative decoding)
- Kernels: `libr4d` (gfx1201 HIP attention / GEMM / all-reduce), built at container start

## License

MIT. See [LICENSE](LICENSE).
