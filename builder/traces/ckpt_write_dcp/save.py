"""ABSTRACTS.md §11 row 3: torch.distributed.checkpoint.save on R ranks (gloo, CPU), a state dict of
DTensors sharded over the ranks (the FSDP2 shape: one write item per parameter, 1/R of it per rank);
`--torch-save` instead writes one torch.save archive of the same items on one rank."""
import os, subprocess, sys
ITEMS_MIB = [32, 64, 32, 16, 1, 0.004]          # full-tensor sizes, fp32; each rank holds 1/R

def tensors(torch):
    g = torch.Generator().manual_seed(1)
    return {f"layer{i}.weight": torch.rand(int(m * (1 << 20)) // 4 // 2 * 2, generator=g) for i, m in enumerate(ITEMS_MIB)}

def rank_main(root, rank, ranks, steps):
    import torch, torch.distributed as dist, torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    dist.init_process_group("gloo", rank=rank, world_size=ranks)
    mesh = init_device_mesh("cpu", (ranks,))
    state = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in tensors(torch).items()}
    for s in range(steps):
        dcp.save(state, checkpoint_id=os.path.join(root, "ckpt", f"step_{100 * (s + 1):06}"))
    dist.destroy_process_group()

if __name__ == "__main__":
    if sys.argv[1] == "--rank":
        rank_main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == "--torch-save":
        import torch
        os.makedirs(os.path.join(sys.argv[2], "ckpt"), exist_ok=True)
        torch.save(tensors(torch), os.path.join(sys.argv[2], "ckpt", "rank0.pt"))
    else:
        root, ranks, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
        ps = [subprocess.Popen([sys.executable, __file__, "--rank", root, str(r), str(ranks), str(steps)], env=env) for r in range(ranks)]
        sys.exit(max(p.wait() for p in ps))
