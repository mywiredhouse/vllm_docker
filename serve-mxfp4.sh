#!/usr/bin/env bash
# Launch the pinned native-MXFP4 vLLM server without stock-kernel fallback.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=stilldeadcode/vllm-radiance:0.9.3
R4D_PIN=${R4D_PIN:-b9e42ab}
MODELS=${MODELS:-${HOME}/models}
SNAP=${SNAP:-Qwen3.8-27B-MXFP4-mtpfp8}
DRAFTER=${DRAFTER:-Qwen3.8-27B-DFlash2-FP8}
PORT=${PORT:-8080}
SPEC_METHOD=${SPEC_METHOD:-dflash}
SPEC=${SPEC:-7}
MAXSEQS=${MAXSEQS:-8}
MAXLEN=${MAXLEN:-262144}
CHUNK=${CHUNK:-8192}
GPU_UTIL=${GPU_UTIL:-0.98}
KV_MEM=${KV_MEM:-auto}
R4D_CACHE=${R4D_CACHE:-${HOME}/.cache/radiance-libr4d}
SHM_SIZE=${SHM_SIZE:-1g}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-mxfp4}
RADIANCE_MXFP4=${RADIANCE_MXFP4:-1}
RADIANCE_MXFP4_W4A8=${RADIANCE_MXFP4_W4A8:-1}
RADIANCE_RUN_BWTEST=${RADIANCE_RUN_BWTEST:-0}
RADIANCE_USE_R4D=${RADIANCE_USE_R4D:-1}
RADIANCE_USE_R4D_AR=${RADIANCE_USE_R4D_AR:-1}
RADIANCE_USE_R4D_AR_QUANT=${RADIANCE_USE_R4D_AR_QUANT:-1}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'HELP'
Usage: ./serve-mxfp4.sh [--help]

Launch Qwen3.8-27B-MXFP4-mtpfp8 on exactly two R9700 GPUs at TP=2.
Set DRY_RUN=1 to print the resolved environment and vllm serve command without
starting a container.

Configuration knobs (VAR=value ./serve-mxfp4.sh):
  MODELS       Model directory, default ~/models, mounted read/write.
  SNAP         Served checkpoint directory, default Qwen3.8-27B-MXFP4-mtpfp8.
               The raw Quark checkpoint is rejected; the fp8 MTP rewrite is required.
  DRAFTER      DFlash model directory, default Qwen3.8-27B-DFlash2-FP8.
  PORT         Host/container HTTP port, default 8080.
  SPEC_METHOD  dflash (default, uses DRAFTER) or mtp (uses the in-target MTP head).
  SPEC         Speculative token count, default 7.
  MAXSEQS      Maximum concurrent sequences, default 8.
  MAXLEN       Maximum context length, default 262144.
  CHUNK        Chunked prefill token size, default 8192.
  GPU_UTIL     vLLM GPU memory utilization, default 0.98.
  KV_MEM       auto (default, use kv-profiles.tsv when present) or a byte count.
  TP / GPUS    auto (default, detected as TP=2 and HIP_VISIBLE_DEVICES=0,3).
  R4D_CACHE    Host pinned libr4d cache, default ~/.cache/radiance-libr4d.
  SHM_SIZE     Container /dev/shm size, default 1g (required by TP=2 engine IPC).
  R4D_PIN      Required libr4d commit, default b9e42ab.
  CONTAINER_RUNTIME  podman (preferred) or docker; auto-detected by default.
  DRY_RUN      1 prints the exact resolved launch and executes nothing.
  RADIANCE_RUN_BWTEST  Startup GPU bandwidth probe, default 0 because the probe
               is nonessential and can report false memory-in-use failures.
  RADIANCE_USE_R4D  Required pinned R4D kernels, default 1; do not disable for production.
  RADIANCE_USE_R4D_AR  Fast R4D all-reduce, default 1; set 0 to use RCCL.
  RADIANCE_USE_R4D_AR_QUANT  Compressed R4D all-reduce payload, default 1.

Fixed serving behavior:
  RADIANCE_MXFP4=1 enables native MXFP4 from config.json; no --quantization flag.
  RADIANCE_MXFP4_W4A8=1 selects fp8-WMMA W4A8 numerics instead of declared W4A4.
  W4A8 is more accurate and is documented as 1.47–2.26x faster than aiter.
  Tool/reasoning parsers and qwen-fixed-v22.3.jinja are always enabled.
HELP
}

select_runtime() {
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 || fail "CONTAINER_RUNTIME=$CONTAINER_RUNTIME is not installed"
        RUNTIME=$CONTAINER_RUNTIME
    elif command -v podman >/dev/null 2>&1; then
        RUNTIME=podman
    elif command -v docker >/dev/null 2>&1; then
        RUNTIME=docker
    else
        fail 'no supported container runtime found; install podman or docker'
    fi
    RUNTIME_KIND=$(basename "$RUNTIME")
}

profile_kv_mem() {
    local profile=${KV_PROFILE_FILE:-${SCRIPT_DIR}/kv-profiles.tsv}
    [[ -f "$profile" ]] || return 0
    awk -F '\t' -v sig="$HARDWARE_SIG" -v seqs="$MAXSEQS" -v len="$MAXLEN" -v chunk="$CHUNK" \
        'NR > 1 && $1 == sig && $2 == seqs && $3 == len && $4 == chunk {print $5; exit}' "$profile"
}

resolve_gpu_config() {
    GPU_DETECT_LIB_ONLY=1 source "$SCRIPT_DIR/gpu-detect.sh"
    detect_gpus
    local requested_tp=${TP:-${GPUS:-auto}}
    [[ "$requested_tp" == auto ]] && requested_tp=$TP
    [[ "$requested_tp" == "$TP" ]] || fail "only detected TP=$TP is supported; requested TP=$requested_tp"
    TP=$requested_tp
    GPUS=${GPUS:-$GPU_IDS}
    [[ "$GPUS" == auto ]] && GPUS=$GPU_IDS
    [[ "$GPUS" == "$GPU_IDS" ]] || fail "only HIP_VISIBLE_DEVICES=$GPU_IDS is supported; requested GPUS=$GPUS"
    HIP_VISIBLE_DEVICES=$GPU_IDS
}

verify_r4d_cache() {
    [[ -d "$R4D_CACHE" ]] || fail "pinned libr4d cache is missing: $R4D_CACHE; run setup-mxfp4.sh"
    [[ -f "$R4D_CACHE/.pin" ]] || fail "libr4d cache has no pinned marker: $R4D_CACHE; refusing stock fallback"
    [[ "$(<"$R4D_CACHE/.pin")" == "$R4D_PIN" ]] || fail "libr4d cache pin is not $R4D_PIN; refusing stock fallback"
}

append_kv_flag() {
    local resolved
    if [[ "$KV_MEM" == auto ]]; then
        resolved=$(profile_kv_mem || true)
        if [[ -n "$resolved" ]]; then
            [[ "$resolved" =~ ^[0-9]+$ ]] || fail "invalid kv_mem in profile: $resolved"
            VLLM_COMMAND+=(--kv-cache-memory-bytes "$resolved")
            KV_RESOLVED="$resolved (profile)"
        else
            KV_RESOLVED=auto
        fi
    else
        [[ "$KV_MEM" =~ ^[0-9]+$ ]] || fail 'KV_MEM must be auto or a byte count'
        VLLM_COMMAND+=(--kv-cache-memory-bytes "$KV_MEM")
        KV_RESOLVED="$KV_MEM (explicit)"
    fi
}

assemble_command() {
    local model_path="/models/${SNAP}"
    [[ "$SNAP" == Qwen3.8-27B-MXFP4-mtpfp8 ]] || fail 'SNAP must be Qwen3.8-27B-MXFP4-mtpfp8; serving the raw checkpoint is forbidden'
    [[ "$SPEC_METHOD" == dflash || "$SPEC_METHOD" == mtp ]] || fail 'SPEC_METHOD must be dflash or mtp'
    [[ "$RADIANCE_MXFP4" == 1 ]] || fail 'RADIANCE_MXFP4 must remain 1 for native MXFP4'

    VLLM_COMMAND=(
        vllm serve "$model_path"
        --tensor-parallel-size "$TP"
        --max-model-len "$MAXLEN"
        --max-num-seqs "$MAXSEQS"
        --gpu-memory-utilization "$GPU_UTIL"
        --enable-chunked-prefill
        --max-num-batched-tokens "$CHUNK"
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --reasoning-parser qwen3
        --chat-template /workspace/qwen-fixed-v22.3.jinja
        --port "$PORT"
    )
    if [[ "$SPEC_METHOD" == dflash ]]; then
        VLLM_COMMAND+=(
            --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/${DRAFTER}\"}"
            --spec-tokens "$SPEC"
        )
    else
        VLLM_COMMAND+=(
            --speculative-config "{\"method\":\"mtp\"}"
            --spec-tokens "$SPEC"
        )
    fi
    append_kv_flag
}

container_command() {
    local -n result=$1
    result=(run --name "$CONTAINER_NAME")
    if [[ "$RUNTIME_KIND" == podman ]]; then
        result+=(--replace --userns=keep-id --group-add=keep-groups)
    else
        result+=(--rm --user "$(id -u):$(id -g)")
    fi
    result+=(
        --device /dev/kfd
        --device /dev/dri
        --shm-size "$SHM_SIZE"
        -p "${PORT}:${PORT}"
        -v "${MODELS}:/models:rw,Z"
        -v "${R4D_CACHE}:/r4d-cache:rw,Z"
        -v "${R4D_CACHE}:/tmp/.cache/radiance-libr4d:rw,Z"
        -v "${SCRIPT_DIR}/qwen-fixed-v22.3.jinja:/workspace/qwen-fixed-v22.3.jinja:ro,Z"
        -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
        -e "RADIANCE_MXFP4=${RADIANCE_MXFP4}"
        -e "RADIANCE_MXFP4_W4A8=${RADIANCE_MXFP4_W4A8}"
        -e "RADIANCE_RUN_BWTEST=${RADIANCE_RUN_BWTEST}"
        -e "RADIANCE_USE_R4D=${RADIANCE_USE_R4D}"
        -e "RADIANCE_USE_R4D_AR=${RADIANCE_USE_R4D_AR}"
        -e "RADIANCE_USE_R4D_AR_QUANT=${RADIANCE_USE_R4D_AR_QUANT}"
        -e "R4D_PIN=${R4D_PIN}"
        -e HOME=/tmp
        -e XDG_CACHE_HOME=/tmp/vllm-cache
        -e "RADIANCE_LIBR4D_CACHE=/r4d-cache"
        -e LD_LIBRARY_PATH=/r4d-cache
        --entrypoint vllm
        "$IMAGE"
    )
    # The pinned image already uses vllm as ENTRYPOINT; do not pass the
    # executable a second time or it parses `vllm` as an argument.
    result+=("${VLLM_COMMAND[@]:1}")
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

main() {
    if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
        usage
        return 0
    fi
    [[ -z "${1:-}" ]] || fail "unknown argument: $1 (use --help)"
    select_runtime
    MODELS=$(cd -- "$MODELS" 2>/dev/null && pwd) || fail "model directory does not exist: $MODELS"
    R4D_CACHE=$(cd -- "$R4D_CACHE" 2>/dev/null && pwd) || fail "libr4d cache directory does not exist: $R4D_CACHE"
    verify_r4d_cache
    resolve_gpu_config
    assemble_command

    printf 'Resolved environment: HIP_VISIBLE_DEVICES=%s RADIANCE_MXFP4=%s RADIANCE_MXFP4_W4A8=%s TP=%s KV_MEM=%s\n' \
        "$HIP_VISIBLE_DEVICES" "$RADIANCE_MXFP4" "$RADIANCE_MXFP4_W4A8" "$TP" "$KV_RESOLVED"
    printf 'Serving checkpoint: /models/%s\n' "$SNAP"
    printf 'vllm command: '
    print_command "${VLLM_COMMAND[@]}"

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        local -a dry_container
        container_command dry_container
        printf 'container command: '
        print_command "$RUNTIME" "${dry_container[@]}"
        printf 'DRY_RUN=1: no container started\n'
        return 0
    fi

    local -a args
    container_command args
    "$RUNTIME" "${args[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${SERVE_LIB_ONLY:-0}" != 1 ]]; then
    main "$@"
fi
