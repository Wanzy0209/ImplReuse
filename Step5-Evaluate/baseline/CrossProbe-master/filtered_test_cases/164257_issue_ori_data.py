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
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False)
        )

model = torch.nn.Linear(4, 2, bias=True)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

loss = torch.mean(model.weight)
loss.backward()
optimizer.step()

dcp.save({'app': AppState(model, optimizer)}, checkpoint_id='test_checkpoint')

model2 = torch.nn.Linear(4, 2, bias=True)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.01)

dcp.load({'app': AppState(model2, optimizer2)}, checkpoint_id='test_checkpoint',
         planner=DefaultLoadPlanner(allow_partial_load=True))