#!/usr/bin/env bash
# Calibrate and upsert a safe KV cache profile for one hardware/batch shape.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE_FILE=${KV_PROFILE_FILE:-${SCRIPT_DIR}/kv-profiles.tsv}
MAXSEQS=${MAXSEQS:-8}
MAXLEN=${MAXLEN:-262144}
CHUNK=${CHUNK:-8192}
KV_MEM_VALUE=${KV_MEM_VALUE:-8589934592}
KV_TOKENS=${KV_TOKENS:-943581}

usage() {
    cat <<'HELP'
Usage: ./calibrate-kv.sh [--kv-mem BYTES] [--kv-tokens COUNT]

The default is a conservative profile for 2x7551-32624. For a measured run,
set KV_MEM_VALUE and KV_TOKENS (or pass the flags) after exercising the live
server. The row is keyed by hardware signature and MAXSEQS/MAXLEN/CHUNK.

Environment: MODELS, MAXSEQS (8), MAXLEN (262144), CHUNK (8192),
HW_SIG (otherwise detected), KV_PROFILE_FILE, KV_MEM_VALUE, KV_TOKENS.
HELP
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --kv-mem) (($# >= 2)) || fail '--kv-mem requires a byte count'; KV_MEM_VALUE=$2; shift 2 ;;
        --kv-tokens) (($# >= 2)) || fail '--kv-tokens requires a token count'; KV_TOKENS=$2; shift 2 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ "$MAXSEQS" =~ ^[0-9]+$ && "$MAXLEN" =~ ^[0-9]+$ && "$CHUNK" =~ ^[0-9]+$ ]] || fail 'MAXSEQS, MAXLEN, and CHUNK must be integers'
[[ "$KV_MEM_VALUE" =~ ^[0-9]+$ && "$KV_TOKENS" =~ ^[0-9]+$ ]] || fail 'KV memory and token count must be integers'

if [[ -n "${HW_SIG:-}" ]]; then
    HARDWARE_SIG=$HW_SIG
else
    GPU_DETECT_LIB_ONLY=1 source "$SCRIPT_DIR/gpu-detect.sh"
    detect_gpus
fi
[[ "$HARDWARE_SIG" =~ ^[0-9]+x[0-9a-fA-F]+-[0-9]+$ ]] || fail "invalid hardware signature: $HARDWARE_SIG"

mkdir -p "$(dirname -- "$PROFILE_FILE")"
tmp=$(mktemp "${PROFILE_FILE}.tmp.XXXXXX")
trap 'rm -f "$tmp"' EXIT
if [[ -f "$PROFILE_FILE" ]]; then
    awk -F '\t' -v OFS='\t' \
        -v sig="$HARDWARE_SIG" -v seqs="$MAXSEQS" -v len="$MAXLEN" -v chunk="$CHUNK" \
        'NR == 1 {print; next} $1 == sig && $2 == seqs && $3 == len && $4 == chunk {next} {print}' \
        "$PROFILE_FILE" > "$tmp"
else
    printf 'hw_sig\tmax_seqs\tmax_len\tchunk\tkv_mem\tkv_tokens\n' > "$tmp"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$HARDWARE_SIG" "$MAXSEQS" "$MAXLEN" "$CHUNK" "$KV_MEM_VALUE" "$KV_TOKENS" >> "$tmp"
mv -- "$tmp" "$PROFILE_FILE"
trap - EXIT
printf 'KV profile updated: sig=%s max_seqs=%s max_len=%s chunk=%s kv_mem=%s kv_tokens=%s\n' \
    "$HARDWARE_SIG" "$MAXSEQS" "$MAXLEN" "$CHUNK" "$KV_MEM_VALUE" "$KV_TOKENS"
