"""TEST FAKE of `torch.distributed.checkpoint` (see `standins/standin.py`): `save` and `load` record
the checkpoint and each item's local element count; `load` fills the tensors in place, as the real
one does, with ones."""
from standin import record


def save(state_dict, checkpoint_id):
    record("dcp.save", checkpoint_id=checkpoint_id, items={k: v.to_local().n for k, v in state_dict.items()})


def load(state_dict, checkpoint_id):
    record("dcp.load", checkpoint_id=checkpoint_id, items={k: v.to_local().n for k, v in state_dict.items()})
    for v in state_dict.values():
        v.to_local().value = 1.0
