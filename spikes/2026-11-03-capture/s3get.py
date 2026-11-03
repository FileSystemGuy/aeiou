# GET-only load over preloaded objects: n small (4 KiB) and m large (8 MiB), each read `rep` times; `setup` writes them
import sys, time, os
which, n, m, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ep = "http://127.0.0.1:19000"
os.environ.setdefault("AWS_ACCESS_KEY_ID", "spike"); os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "spikespike123")
if which in ("boto3", "setup"):
    import boto3
    s3 = boto3.client("s3", endpoint_url=ep, region_name="us-east-1")
    if which == "setup":
        try: s3.create_bucket(Bucket="ggg")
        except Exception: pass
        for i in range(n): s3.put_object(Bucket="ggg", Key=f"s/{i}", Body=os.urandom(4096))
        for i in range(m): s3.put_object(Bucket="ggg", Key=f"l/{i}", Body=os.urandom(8 << 20))
        sys.exit(0)
    get = lambda k: s3.get_object(Bucket="ggg", Key=k)["Body"].read()
else:
    from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client
    c = MountpointS3Client("us-east-1", endpoint=ep, force_path_style=True)
    get = lambda k: b"".join(c.get_object("ggg", k))
get("s/0")  # connection and imports outside the timing
t = time.time()
for r in range(rep):
    for i in range(n): get(f"s/{i}")
    for i in range(m): get(f"l/{i}")
print(f"get={time.time()-t:.4f}", file=sys.stderr)
