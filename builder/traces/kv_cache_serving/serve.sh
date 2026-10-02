#!/bin/sh
# vLLM with LMCache's local-disk backend on the directory lmcache.yaml names.
#   sh serve.sh [MODEL] [PORT]        (prefix with strace to trace; see README.md)
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${2:-8000}
export VLLM_USE_FLASHINFER_SAMPLER=0      # its kernels are compiled at first use and need nvcc
export LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE:-$(dirname "$0")/lmcache.yaml}
exec vllm serve "$MODEL" --port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.7 ${KV_BYTES:+--kv-cache-memory-bytes $KV_BYTES} \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
