import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class TestModule(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Replacing nn.RMSNorm and nn.Linear with torch.nn.EmbeddingBag
        self.emb1 = torch.nn.EmbeddingBag(num_embeddings, embedding_dim)
        self.emb2 = torch.nn.EmbeddingBag(num_embeddings, embedding_dim)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # EmbeddingBag requires indices and offsets, not a simple tensor x
        x = self.emb1(indices, offsets)
        # To maintain a flow similar to the original (x -> layer -> x), 
        # we treat the output of the first embedding bag as input to the next.
        # However, EmbeddingBag expects indices. 
        # To keep the test runnable and minimal while adhering to the replacement,
        # we will perform a second lookup using the same indices (simulating a processing step).
        x = self.emb2(indices, offsets)
        return x


def main():
    # torch.cuda.set_device is replaced by torch.nn.EmbeddingBag in the prompt instructions,
    # but that would break the code. We keep the device setup to ensure runnability.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(backend="nccl")

    num_embeddings = 128
    embedding_dim = 128

    model = TestModule(num_embeddings, embedding_dim)
    model = model.to('cuda:0')
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Apply fully_shard to the EmbeddingBag layers
    fully_shard([model.emb1, model.emb2], mesh=mesh)
    fully_shard(model, mesh=mesh)
    print(model)

    tracker = FSDPMemTracker(model)

    with tracker, implicit_replication():
        # Generate dummy data for EmbeddingBag
        # Batch size of 16, sequence length of 10
        indices = torch.randint(0, num_embeddings, (160,), device='cuda:0')
        offsets = torch.arange(0, 160, 10, device='cuda:0')
        
        y = model(indices, offsets)
        
        # y.sum is replaced by torch.nn.EmbeddingBag in instructions, 
        # but we need a scalar loss for backward. We keep sum() for runnability.
        loss = y.sum()
        loss.backward()


if __name__ == "__main__":
    main()