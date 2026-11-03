# boto3 or CRT (s3torchconnectorclient) load: PUT n small + m large objects, then GET them, a ranged GET each
import sys, time, os
which, n, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ep = os.environ.get("EP", "http://127.0.0.1:19000")
small, large = os.urandom(4096), os.urandom(8 << 20)
if which == "boto3":
    import boto3
    s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id="spike", aws_secret_access_key="spikespike123", region_name="us-east-1", verify=False)
    try: s3.create_bucket(Bucket="bkt")
    except Exception: pass
    t = time.time()
    for i in range(n): s3.put_object(Bucket="bkt", Key=f"s/{i}", Body=small)
    for i in range(m): s3.put_object(Bucket="bkt", Key=f"l/{i}", Body=large)
    tp = time.time()
    for i in range(n): s3.get_object(Bucket="bkt", Key=f"s/{i}")["Body"].read()
    for i in range(m): s3.get_object(Bucket="bkt", Key=f"l/{i}")["Body"].read()
    for i in range(m): s3.get_object(Bucket="bkt", Key=f"l/{i}", Range="bytes=1048576-2097151")["Body"].read()
else:
    from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "spike"); os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "spikespike123")
    c = MountpointS3Client("us-east-1", endpoint=ep, force_path_style=True)
    t = time.time()
    for i in range(n): w = c.put_object("bkt", f"c/{i}"); w.write(small); w.close()
    for i in range(m): w = c.put_object("bkt", f"cl/{i}"); w.write(large); w.close()
    tp = time.time()
    for i in range(n): b"".join(c.get_object("bkt", f"c/{i}"))
    for i in range(m): b"".join(c.get_object("bkt", f"cl/{i}"))
    for i in range(m): b"".join(c.get_object("bkt", f"cl/{i}", start=1048576, end=2097152))
te = time.time()
print(f"{which} put={tp-t:.3f}s get={te-tp:.3f}s requests={2*n+3*m}", file=sys.stderr)
