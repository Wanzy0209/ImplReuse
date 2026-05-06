import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

class AppState(Stateful):
    def __init__(self, value: int = 0):
        self.value = value

    def state_dict(self):
        return {
            "value": self.value,
        }

    def load_state_dict(self, state):
        self.value = state['value']

if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            init_method="file:///tmp/tmpfile"
        )

    app_state = AppState(value=7)
    state = {"app": app_state}
    future = dcp.async_save(state, checkpoint_id="checkpoints/ckpt")
    future.result() # AttributeError: 'FileSystemReader' object has no attribute 'storage_meta'

    restored = AppState()
    state = {"app": restored}
    dcp.load(state, checkpoint_id="checkpoints/ckpt")

    assert restored.value == app_state.value