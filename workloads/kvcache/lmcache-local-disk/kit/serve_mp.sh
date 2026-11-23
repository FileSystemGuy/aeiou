#!/bin/sh
# LMCache's multi-process server with two fs_native L2 adapters, and vLLM on its connector.
#   sh serve_mp.sh server DIR3 DIR4   LMCache's server: G2 its L1, G3 and G4 the two directories, each evicting on its own
#   sh serve_mp.sh engine [MODEL]     vLLM as serve.sh starts it, its connector LMCache's multi-process one
#   L1_GB, G3_GB, G4_GB (defaults 4, 2, 6) size the tiers; MAX_MODEL_LEN, KV_BYTES, VLLM_ARGS as in serve.sh
#   SERVER_ARGS: more flags of the server. Where CUDA's interprocess sharing fails (WSL2), the KV
#   is passed through shared memory: SERVER_ARGS="--supported-transfer-mode engine_driven --shm-name NAME"
#   for the server and LMCACHE_MP_TRANSFER_MODE=engine_driven for the engine (README.md).
# Both report usage to their vendors unless told not to; the captures ran with these set.
export LMCACHE_TRACK_USAGE=false DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
case "$1" in
server)
    adapter() { printf '{"type":"fs_native","base_path":"%s","max_capacity_gb":%s,"eviction":{"eviction_policy":"LRU"}}' "$1" "$2"; }
    exec lmcache server --host localhost --port 5555 --chunk-size 256 $SERVER_ARGS --l1-size-gb "${L1_GB:-4}" --eviction-policy LRU \
        --l2-adapter "$(adapter "$2" "${G3_GB:-2}")" --l2-adapter "$(adapter "$3" "${G4_GB:-6}")" ;;
engine)
    export VLLM_USE_FLASHINFER_SAMPLER=0      # its kernels are compiled at first use and need nvcc
    exec vllm serve "${2:-Qwen/Qwen2.5-0.5B-Instruct}" --port 8000 --max-model-len "${MAX_MODEL_LEN:-4096}" --gpu-memory-utilization 0.7 ${KV_BYTES:+--kv-cache-memory-bytes $KV_BYTES} \
        --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}}' $VLLM_ARGS ;;
*)  echo "usage: sh serve_mp.sh server DIR3 DIR4 | engine [MODEL]" >&2; exit 2 ;;
esac
