#!/usr/bin/env bash
# Detect the two in-scope R9700 GPUs and derive the only supported TP size.
set -euo pipefail

SYSFS_ROOT=${SYSFS_ROOT:-/sys}
DEV_ROOT=${DEV_ROOT:-/dev}
VRAM_MIN_MIB=${VRAM_MIN_MIB:-8192}
TARGET_DEVICE_ID=7551
TARGET_GPU_INDICES=0,3

# These functions are intentionally shell-testable. Set GPU_DETECT_LIB_ONLY=1
# before sourcing this file to avoid running the CLI entry point.
tp_is_valid() {
    local tp=${1:?tensor parallel size is required}
    case "$tp" in
        1|2|4|8) return 0 ;;
        *) return 1 ;;
    esac
}

derive_tp() {
    local count=${1:?usable GPU count is required}
    local candidate
    for candidate in 8 4 2 1; do
        if (( count >= candidate && count % candidate == 0 )) && tp_is_valid "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '1\n'
}

require_device_nodes() {
    local missing=0
    if [[ ! -e "$DEV_ROOT/kfd" ]]; then
        printf 'ERROR: missing %s/kfd; load the amdgpu driver and expose /dev/kfd\n' "$DEV_ROOT" >&2
        missing=1
    fi
    if [[ ! -d "$DEV_ROOT/dri" ]]; then
        printf 'ERROR: missing %s/dri; load the amdgpu driver and expose /dev/dri\n' "$DEV_ROOT" >&2
        missing=1
    fi
    (( missing == 0 ))
}

is_vram_usable() {
    local vram_mib=${1:?VRAM in MiB is required}
    [[ "$vram_mib" =~ ^[0-9]+$ ]] && (( vram_mib >= VRAM_MIN_MIB ))
}

read_vram_mib() {
    local file=$1 raw bytes
    raw=$(tr -d '[:space:]' < "$file")
    [[ "$raw" =~ ^[0-9]+$ ]] || {
        printf 'ERROR: invalid VRAM value in %s: %s\n' "$file" "$raw" >&2
        return 1
    }
    bytes=$raw
    printf '%s\n' "$((bytes / 1024 / 1024))"
}

read_device_id() {
    local device_dir=$1 value=''
    for file in "$device_dir/device" "$device_dir/uevent"; do
        [[ -r "$file" ]] || continue
        if [[ "$file" == */uevent ]]; then
            value=$(awk -F= '$1 == "PCI_ID" {print $2; exit}' "$file")
            value=${value#*:}
        else
            value=$(tr -d '[:space:]' < "$file")
        fi
        value=${value#0x}
        [[ -n "$value" ]] && {
            printf '%s\n' "${value,,}"
            return 0
        }
    done
    return 1
}

detect_gpus() {
    local vram_file device_dir render_node vram_mib device_id
    local usable_count=0 target_count=0 target_vram=''
    local -a vram_files=()

    if [[ "${GPU_DETECT_SKIP_DEVICE_CHECK:-0}" != 1 ]]; then
        require_device_nodes
    fi

    while IFS= read -r -d '' vram_file; do
        vram_files+=("$vram_file")
    done < <(compgen -G "$SYSFS_ROOT/class/drm/renderD*/device/mem_info_vram_total" | while read -r file; do printf '%s\0' "$file"; done)

    if ((${#vram_files[@]} == 0)); then
        printf 'ERROR: no amdgpu render nodes with mem_info_vram_total found below %s/class/drm\n' "$SYSFS_ROOT" >&2
        return 1
    fi

    for vram_file in "${vram_files[@]}"; do
        device_dir=${vram_file%/mem_info_vram_total}
        render_node=${vram_file#"$SYSFS_ROOT/class/drm/"}
        render_node=${render_node%%/*}
        vram_mib=$(read_vram_mib "$vram_file")
        device_id=$(read_device_id "$device_dir") || device_id=unknown
        printf 'GPU: %s device=0x%s vram=%s MiB' "$render_node" "$device_id" "$vram_mib"
        if ! is_vram_usable "$vram_mib"; then
            printf ' excluded (below %s MiB)\n' "$VRAM_MIN_MIB"
            continue
        fi
        printf ' usable\n'
        ((usable_count += 1))
        if [[ "$device_id" == "$TARGET_DEVICE_ID" ]]; then
            ((target_count += 1))
            if [[ -z "$target_vram" ]]; then
                target_vram=$vram_mib
            elif [[ "$target_vram" != "$vram_mib" ]]; then
                printf 'ERROR: R9700 VRAM is inconsistent (%s vs %s MiB); refusing mixed cards\n' "$target_vram" "$vram_mib" >&2
                return 1
            fi
        fi
    done

    if (( target_count != 2 )); then
        printf 'ERROR: expected exactly two usable R9700 GPUs (device 0x%s), found %s\n' "$TARGET_DEVICE_ID" "$target_count" >&2
        return 1
    fi

    TP=$(derive_tp "$target_count")
    if ! tp_is_valid "$TP" || ((24 % TP != 0)) || ((16 % TP != 0)) || ((48 % TP != 0)); then
        printf 'ERROR: TP=%s does not divide num_attention_heads=24, linear_num_key_heads=16, and linear_num_value_heads=48\n' "$TP" >&2
        return 1
    fi

    GPU_COUNT=$target_count
    USABLE_GPU_COUNT=$usable_count
    GPU_IDS=$TARGET_GPU_INDICES
    HIP_VISIBLE_DEVICES=$TARGET_GPU_INDICES
    HARDWARE_SIG="${target_count}x${TARGET_DEVICE_ID}-${target_vram}"
    export TP GPU_COUNT USABLE_GPU_COUNT GPU_IDS HIP_VISIBLE_DEVICES HARDWARE_SIG

    printf 'Detected %s usable GPU(s); %s usable R9700; signature=%s; TP=%s; HIP_VISIBLE_DEVICES=%s\n' \
        "$USABLE_GPU_COUNT" "$GPU_COUNT" "$HARDWARE_SIG" "$TP" "$HIP_VISIBLE_DEVICES"
}

emit_env() {
    printf 'TP=%q\nGPU_COUNT=%q\nUSABLE_GPU_COUNT=%q\nGPU_IDS=%q\nHIP_VISIBLE_DEVICES=%q\nHARDWARE_SIG=%q\n' \
        "$TP" "$GPU_COUNT" "$USABLE_GPU_COUNT" "$GPU_IDS" "$HIP_VISIBLE_DEVICES" "$HARDWARE_SIG"
}

main() {
    local mode=${1:-report}
    case "$mode" in
        --env)
            detect_gpus >/dev/stderr
            emit_env
            ;;
        --help|-h)
            printf 'Usage: %s [--env]\n' "$0"
            printf 'Detect usable amdgpu render nodes and the target 2x R9700 TP=2 configuration.\n'
            ;;
        report)
            detect_gpus
            ;;
        *)
            printf 'ERROR: unknown option: %s\n' "$mode" >&2
            return 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${GPU_DETECT_LIB_ONLY:-0}" != 1 ]]; then
    main "$@"
fi
