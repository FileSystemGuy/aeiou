"""TEST FAKE of `torch.distributed.device_mesh` (see `standins/standin.py`)."""
from standin import record


def init_device_mesh(device_type, mesh_shape):
    record("init_device_mesh", device_type=device_type, mesh_shape=list(mesh_shape))
    return mesh_shape
