import pathlib
import shutil

import torch
import torch.optim as optim
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint import DefaultLoadPlanner

class AppState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False)
        )

tmp_dir = pathlib.Path("test/tmp/test_dataset_datamodule_resume")
shutil.rmtree(tmp_dir, ignore_errors=True)

# Model + optimizer
model = torch.nn.Linear(4, 2, bias=True)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# One training step
loss = torch.mean(model.weight)
loss.backward()
optimizer.step()

# Save checkpoint
dcp.save({'app': AppState(model, optimizer)}, checkpoint_id=tmp_dir)

# Recreate model + optimizer and load
model2 = torch.nn.Linear(4, 2, bias=True)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.01)

# RuntimeError: Missing key in checkpoint state_dict: optimizer.state.bias.step.
dcp.load({'app': AppState(model, optimizer)}, checkpoint_id=tmp_dir,
         planner=DefaultLoadPlanner(allow_partial_load=True))