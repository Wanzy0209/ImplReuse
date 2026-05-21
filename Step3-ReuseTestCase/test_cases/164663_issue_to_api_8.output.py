import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class TestModule(nn.Module):
    """
    Test module replacing RMSNorm with EmbeddingBag to verify 
    FSDPMemTracker behavior with similar APIs.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Using EmbeddingBag as the candidate API similar to RMSNorm in structure
        self.emb = nn.EmbeddingBag(num_embeddings, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EmbeddingBag with 2D input (batch, seq_len) performs reduction automatically
        x = self.emb(x)
        x = self.output(x)
        return x


def main():
    # Setup environment variables for distributed execution
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")

    num_embeddings = 128
    embedding_dim = 64

    model = TestModule(num_embeddings, embedding_dim)
    model = model.to('cuda:0')
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Apply fully_shard to the submodules and the parent model, 
    # mirroring the structure of the original bug report
    fully_shard([model.emb, model.output], mesh=mesh)
    fully_shard(model, mesh=mesh)

    tracker = FSDPMemTracker(model)

    try:
        with tracker, implicit_replication():
            # Generate random indices for EmbeddingBag
            x = torch.randint(0, num_embeddings, (16, 10), device='cuda:0')
            y = model(x)
            loss = y.sum()
            loss.backward()
        
        # If backward completes without KeyError, the test passes
        assert True, "FSDPMemTracker handled EmbeddingBag correctly during backward"
        print("Test passed: FSDPMemTracker works with EmbeddingBag")
    except KeyError as e:
        # Catch the specific error mentioned in the bug report
        assert False, f"FSDPMemTracker failed with KeyError for EmbeddingBag: {e}"


if __name__ == "__main__":
    main()