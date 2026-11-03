#!/bin/bash
# Builds the spike's programs and starts its two MinIO servers (HTTP 19000, HTTPS 19443, data in
# /dev/shm). Needs gcc, openssl, uv, and a minio binary: MINIO=path (default dev.sh's pinned one).
set -eu
cd "$(dirname "$0")"
MINIO=${MINIO:-$HOME/.cache/aeiou-dev/$(uname -m)/minio}
gcc -O2 -Wall -o bench bench.c
gcc -O2 -Wall -o mtrace mtrace.c
gcc -O2 -Wall -shared -fPIC -o shim.so shim.c -ldl -lpthread
gcc -O2 -Wall -shared -fPIC -o pshim.so pshim.c -ldl
[ -f data.bin ] || head -c 64M /dev/urandom > data.bin
# uv's standalone Python (static OpenSSL) with boto3 and the CRT client; the system's (shared libssl) with boto3
[ -d venv ] || { uv venv -q venv && VIRTUAL_ENV=$PWD/venv uv pip install -q boto3 s3torchconnectorclient; }
[ -d sysvenv ] || { uv venv -q --python /usr/bin/python3 sysvenv && VIRTUAL_ENV=$PWD/sysvenv uv pip install -q boto3; }
[ -f certs/public.crt ] || { mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/private.key \
    -out certs/public.crt -days 30 -subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" 2>/dev/null; }
export MINIO_ROOT_USER=spike MINIO_ROOT_PASSWORD=spikespike123
mkdir -p /dev/shm/spike-minio /dev/shm/spike-minio-tls
nohup "$MINIO" server /dev/shm/spike-minio --address 127.0.0.1:19000 --console-address 127.0.0.1:19001 > minio.log 2>&1 &
echo $! > minio.pid
nohup "$MINIO" server /dev/shm/spike-minio-tls --certs-dir certs --address 127.0.0.1:19443 --console-address 127.0.0.1:19444 > minio-tls.log 2>&1 &
echo $! > minio-tls.pid
sleep 3
venv/bin/python s3get.py setup 200 8 1
echo "ready; stop the servers with: kill \$(cat minio.pid minio-tls.pid)"
