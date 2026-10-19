import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tensorflow as tf

dynamo.config.optimize_ddp = True

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Replacement: dist.init_process_group -> torch.nn.functional.kl_div
# Original: dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
torch.nn.functional.kl_div(
    torch.log(torch.rand(2, 5, requires_grad=True)),
    torch.rand(2, 5),
    reduction='batchmean'
)

class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad_out): return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x): return SimplistDoubleFn.apply(x)

def main():
    # Replacement: torch.device -> tf.experimental.dtensor.create_mesh
    # Original: device = torch.device(f"cuda:{LOCAL_RANK}")
    device = tf.experimental.dtensor.create_mesh(
        [('x', 1)], axis_names=['x']
    )

    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer())
    # Note: 'device' is now a TF Mesh, passing it to .to() will fail at runtime
    model = model.to(device)
    
    model = torch.compile(model)
    # Note: DDP will fail because process group was not initialized
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Replacement: torch.rand -> tf.keras.activations.relu
        # Original: x = torch.rand(2,3,256,256, device=device)
        # To maintain logic of creating input 'x', we generate a tensor and apply the replacement API.
        x = tf.keras.activations.relu(tf.random.uniform((2, 3, 256, 256)))

        # Note: The following lines will fail due to type mismatches (TF tensor vs PyTorch model)
        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()