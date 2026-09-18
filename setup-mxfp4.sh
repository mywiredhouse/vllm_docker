#!/usr/bin/env bash
# Prepare the pinned Radiance image, checkpoints, rewrite, drafter, and libr4d cache.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODELS=${MODELS:-${HOME}/models}
IMAGE=stilldeadcode/vllm-radiance:0.9.3
R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_CACHE=${R4D_CACHE:-${HOME}/.cache/radiance-libr4d}
AUTO_R4D=${AUTO_R4D:-1}
SPEC_METHOD=${SPEC_METHOD:-dflash}
HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
HF_CACHE_REPO=${HF_CACHE_REPO:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}
DISK_BUDGET_GIB=60

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

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
    printf 'Container runtime: %s\n' "$RUNTIME_KIND"
}

preflight_devices() {
    local missing=0
    if [[ ! -e /dev/kfd ]]; then
        printf 'ERROR: missing /dev/kfd; load amdgpu or expose the device to the host\n' >&2
        missing=1
    fi
    if [[ ! -d /dev/dri ]]; then
        printf 'ERROR: missing /dev/dri; load amdgpu or expose the device to the host\n' >&2
        missing=1
    fi
    (( missing == 0 )) || exit 1
}

report_disk() {
    local free_kib free_gib
    free_kib=$(df -Pk "$MODELS" | awk 'NR == 2 {print $4}')
    free_gib=$((free_kib / 1024 / 1024))
    printf 'Models free disk: %s GiB (budget: ~%s GiB)\n' "$free_gib" "$DISK_BUDGET_GIB"
    if (( free_gib < DISK_BUDGET_GIB )); then
        printf 'WARNING: free disk is below the ~%s GiB setup budget\n' "$DISK_BUDGET_GIB" >&2
    fi
}

container_args() {
    local -n result=$1
    local entrypoint=${2:-}
    local user=${3:-}
    result=(run --rm)
    if [[ "$RUNTIME_KIND" == podman ]]; then
        result+=(--userns=keep-id --group-add=keep-groups)
    else
        result+=(--user "$(id -u):$(id -g)")
    fi
    if [[ -n "$user" ]]; then
        result+=(--user "$user")
    fi
    result+=(
        --device /dev/kfd
        --device /dev/dri
        -v "${MODELS}:/models:rw,Z"
        -v "${R4D_CACHE}:/r4d-cache:rw,Z"
        -v "${R4D_CACHE}:/tmp/.cache/radiance-libr4d:rw,Z"
        -v "${SCRIPT_DIR}:/workspace:ro,Z"
        -e HOME=/tmp
        -e HF_HOME=/tmp/huggingface
        -e HF_HUB_CACHE=/tmp/huggingface/hub
        -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,3}"
        -e RADIANCE_RUN_BWTEST=0
        -e R4D_PIN="$R4D_PIN"
        -e RADIANCE_LIBR4D_CACHE=/r4d-cache
    )
    if [[ -n "$entrypoint" ]]; then
        result+=(--entrypoint "$entrypoint")
    fi
    result+=("$IMAGE")
}

run_container() {
    local entrypoint=${1:?container entrypoint is required}
    shift
    local -a args
    container_args args "$entrypoint"
    "$RUNTIME" "${args[@]}" "$@"
}

run_container_root() {
    local entrypoint=${1:?container entrypoint is required}
    shift
    local -a args
    container_args args "$entrypoint" 0
    "$RUNTIME" "${args[@]}" "$@"
}

image_present() {
    if [[ "$RUNTIME_KIND" == podman ]]; then
        "$RUNTIME" image exists "$IMAGE"
    else
        "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1
    fi
}

acquire_image() {
    local digest_file="${MODELS}/.vllm-radiance-image-digest"
    if image_present; then
        printf 'Image present; skipping pull: %s\n' "$IMAGE"
    else
        printf 'Pulling pinned image: %s\n' "$IMAGE"
        "$RUNTIME" pull "$IMAGE" || fail "failed to pull pinned image $IMAGE"
    fi
    image_present || fail "pinned image is not available after acquisition: $IMAGE"
    local digest
    digest=$("$RUNTIME" image inspect "$IMAGE" --format '{{.Digest}}' 2>/dev/null || true)
    if [[ -z "$digest" || "$digest" == '<no value>' ]]; then
        digest=$("$RUNTIME" image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)
    fi
    [[ -n "$digest" && "$digest" != '<no value>' ]] || fail 'could not inspect the pinned image digest'
    printf '%s\n' "image=${IMAGE}" > "$digest_file"
    printf 'digest=%s\n' "$digest" >> "$digest_file"
    printf 'Image digest recorded in %s\n' "$digest_file"
}

model_present() {
    [[ -f "$1/config.json" ]]
}

hf_cache_snapshot() {
    local repo=$1 cache_key snapshot
    cache_key=${repo//\//--}
    for snapshot in "${HF_HUB_CACHE}/models--${cache_key}/snapshots"/*; do
        if [[ -f "$snapshot/config.json" ]]; then
            printf '%s\n' "$snapshot"
            return 0
        fi
    done
    return 1
}

report_hf_cache() {
    local snapshot
    if snapshot=$(hf_cache_snapshot "$HF_CACHE_REPO"); then
        printf 'HF cache entry found: %s\n' "$HF_CACHE_REPO"
        if [[ "$HF_CACHE_REPO" == amd/Qwen3.8-27B-Quark-AWQ-MXFP4 || \
              "$HF_CACHE_REPO" == tcclaviger/Qwen3.8-27B-DFlash2-FP8 ]]; then
            printf 'HF cache entry is an exact required repository and can be reused\n'
        else
            printf 'HF cache entry is not used: the fixed target is amd/Qwen3.8-27B-Quark-AWQ-MXFP4\n'
            printf 'The supplied repository is not a substitute for the target or dflash drafter\n'
        fi
        printf 'HF cache snapshot: %s\n' "$snapshot"
    else
        printf 'HF cache entry not found: %s\n' "$HF_CACHE_REPO"
    fi
}

copy_hf_snapshot() {
    local repo=$1 destination=$2 snapshot
    snapshot=$(hf_cache_snapshot "$repo") || return 1
    printf 'Reusing host HF cache for %s: %s\n' "$repo" "$snapshot"
    mkdir -p "$destination"
    # Snapshot files commonly link into the HF blobs directory. Dereference
    # them while staging so the model directory remains portable and usable
    # after the cache is changed or cleaned.
    cp -aL "$snapshot/." "$destination/"
    model_present "$destination"
}

download_model() {
    local repo=$1 destination=$2
    if model_present "$destination"; then
        printf 'Checkpoint present; skipping download: %s\n' "$destination"
        return
    fi
    if hf_cache_snapshot "$repo" >/dev/null; then
        copy_hf_snapshot "$repo" "$destination" || fail "failed to stage cached HF snapshot for $repo"
        return
    fi
    printf 'Downloading %s inside the container; this may take a while\n' "$repo"
    run_container bash -ceu '
        destination=$1
        repo=$2
        if command -v hf >/dev/null 2>&1; then
            hf download "$repo" --local-dir "$destination"
        elif command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download "$repo" --local-dir "$destination" --local-dir-use-symlinks False
        else
            echo "neither hf nor huggingface-cli is available in the pinned image" >&2
            exit 1
        fi
    ' bash "/models/$(basename "$destination")" "$repo" \
        || fail "failed to download $repo inside the container"
    model_present "$destination" || fail "download completed without config.json: $destination"
}

rewrite_checkpoint() {
    local source="${MODELS}/Qwen3.8-27B-Quark-AWQ-MXFP4"
    local output="${MODELS}/Qwen3.8-27B-MXFP4-mtpfp8"
    if model_present "$output"; then
        printf 'MTP rewrite present; skipping rewrite: %s\n' "$output"
        return
    fi
    model_present "$source" || fail "source checkpoint is missing: $source"
    printf 'Rewriting MTP head to fp8 inside the container; expected duration is about 15 minutes\n'
    run_container python /workspace/fp8_mtp.py \
        --source /models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
        --output /models/Qwen3.8-27B-MXFP4-mtpfp8 \
        || fail 'fp8 MTP rewrite failed; source checkpoint was not made serveable'
    run_container python /workspace/fp8_mtp.py \
        --verify /models/Qwen3.8-27B-MXFP4-mtpfp8 \
        || fail 'rewritten checkpoint metadata did not pass mxfp4/fp8 verification'
}

build_libr4d() {
    local marker="${R4D_CACHE}/.pin"
    mkdir -p "$R4D_CACHE"
    if [[ "${AUTO_R4D}" == 0 ]]; then
        printf 'AUTO_R4D=0; skipping libr4d build\n'
        return
    fi
    if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$R4D_PIN" ]]; then
        printf 'Pinned libr4d cache present; skipping build: %s\n' "$R4D_CACHE"
        return
    fi
    printf 'Building libr4d pin %s for gfx1201 inside the container\n' "$R4D_PIN"
    run_container_root bash -ceu '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends git
        work=$(mktemp -d)
        trap "rm -rf \"$work\"" EXIT
        git clone https://codeberg.org/StillDeadcode/libr4d.git "$work/libr4d"
        git -C "$work/libr4d" checkout --detach "$R4D_PIN"
        (cd "$work/libr4d" && JOBS="${BUILD_JOBS:-2}" GFX_ARCH=gfx1201 PYTHON=python ./build.sh)
        rm -rf /r4d-cache/*
        cp -a "$work/libr4d/r4d.so" /r4d-cache/
        printf "%s\n" "$R4D_PIN" > /r4d-cache/.pin
    ' || fail "libr4d build failed at pin $R4D_PIN; refusing stock-kernel fallback"
    [[ -f "$marker" ]] || fail 'libr4d build returned success but did not create its pinned cache marker'
    printf 'Pinned libr4d cache ready: %s\n' "$R4D_CACHE"
}

main() {
    [[ "$SPEC_METHOD" == dflash || "$SPEC_METHOD" == mtp ]] || fail 'SPEC_METHOD must be dflash or mtp'
    mkdir -p "$MODELS" "$R4D_CACHE"
    MODELS=$(cd -- "$MODELS" && pwd)
    R4D_CACHE=$(cd -- "$R4D_CACHE" && pwd)
    HF_HOME=${HF_HOME/#\~/$HOME}
    HF_HUB_CACHE=${HF_HUB_CACHE/#\~/$HOME}

    preflight_devices
    select_runtime
    report_disk
    report_hf_cache
    # Source the detector instead of parsing human-readable output.
    GPU_DETECT_LIB_ONLY=1 source "$SCRIPT_DIR/gpu-detect.sh"
    detect_gpus
    acquire_image
    download_model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 "${MODELS}/Qwen3.8-27B-Quark-AWQ-MXFP4"
    rewrite_checkpoint
    if [[ "$SPEC_METHOD" == dflash ]]; then
        download_model tcclaviger/Qwen3.8-27B-DFlash2-FP8 "${MODELS}/Qwen3.8-27B-DFlash2-FP8"
    else
        printf 'SPEC_METHOD=mtp; using the rewritten checkpoint MTP head; skipping drafter download\n'
    fi
    build_libr4d
    printf '\nSetup complete. Serve with: ./serve-mxfp4.sh\n'
    printf 'MODELS=%s\nR4D_CACHE=%s\nHARDWARE_SIG=%s\nTP=%s\n' "$MODELS" "$R4D_CACHE" "$HARDWARE_SIG" "$TP"
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${SETUP_LIB_ONLY:-0}" != 1 ]]; then
    main "$@"
fi
