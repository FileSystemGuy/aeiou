"""TEST FAKE of `torch.distributed` (see `standins/standin.py`): the process group's calls, with the
rendezvous the launcher gave the rank."""
import os

from standin import record


def init_process_group(backend, rank, world_size):
    record("init_process_group", backend=backend, rank=rank, world_size=world_size,
           master=[os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")])


def destroy_process_group():
    record("destroy_process_group")
