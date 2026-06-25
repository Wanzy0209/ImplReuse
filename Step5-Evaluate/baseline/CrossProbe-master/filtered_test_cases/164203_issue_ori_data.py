import torch
from torch.distributed.pipelining import pipeline, SplitPoint

class X(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states.to(torch.float32)

class Y(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 16, -1)
        self.layers = torch.nn.ModuleList([X() for _ in range(2)])

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

model = Y()
split_point = "layers.1"
example_input_ids = torch.randint(0, 32, (2, 32), dtype=torch.long)
pipe = pipeline(
    model,
    mb_args=(example_input_ids,),
    split_spec={split_point: SplitPoint.BEGINNING},
)
output = pipe(example_input_ids)