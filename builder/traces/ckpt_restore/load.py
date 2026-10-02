"""ABSTRACTS.md §11 row 4a: torch.distributed.checkpoint.load on R ranks (gloo, CPU) of a checkpoint of
DTensors sharded over the ranks (the FSDP2 shape, as ../ckpt_write_dcp/save.py), each rank loading its
own shard. `--save` writes the checkpoint first (untraced); ITEMS is a comma list of full-tensor MiB."""
import os, subprocess, sys

def rank_main(mode, root, rank, ranks, step, items):
    import torch, torch.distributed as dist, torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    dist.init_process_group("gloo", rank=rank, world_size=ranks)
    mesh = init_device_mesh("cpu", (ranks,))
    g = torch.Generator().manual_seed(1)
    make = (lambda n: torch.rand(n, generator=g)) if mode == "--save" else torch.zeros
    state = {f"layer{i}.weight": distribute_tensor(make(int(m * (1 << 20)) // 4 // 2 * 2), mesh, [Shard(0)])
             for i, m in enumerate(items)}
    path = os.path.join(root, "ckpt", f"step_{step:06}")
    if mode == "--save":
        dcp.save(state, checkpoint_id=path)
    else:
        dcp.load(state, checkpoint_id=path)
        assert all(float(v.to_local().sum()) > 0 for v in state.values())
    dist.destroy_process_group()

if __name__ == "__main__":
    if sys.argv[1] == "--rank":
        rank_main(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), [float(x) for x in sys.argv[7].split(",")])
    else:
        mode = "--save" if sys.argv[1] == "--save" else "--load"
        root, ranks, step, items = sys.argv[-4], int(sys.argv[-3]), int(sys.argv[-2]), sys.argv[-1]
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="29512")
        ps = [subprocess.Popen([sys.executable, __file__, "--rank", mode, root, str(r), str(ranks), str(step), items], env=env) for r in range(ranks)]
        sys.exit(max(p.wait() for p in ps))
