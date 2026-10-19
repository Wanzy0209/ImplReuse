import os
import torch
import tensorflow as tf  # Imported for the replacement APIs
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dynamo.config.optimize_ddp = True  # <== necessary for the bug

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Replacement: dist.init_process_group -> tf.experimental.dtensor.create_distributed_mesh
# Original: dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
tf.experimental.dtensor.create_distributed_mesh(backend="nccl" if dist.is_nccl_available() else "gloo")

class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad_out): return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x): return SimplistDoubleFn.apply(x)

def main():
    # Replacement: torch.device -> tf.compat.v1.image.resize_bilinear
    # Original: device = torch.device(f"cuda:{LOCAL_RANK}")
    device = tf.compat.v1.image.resize_bilinear(f"cuda:{LOCAL_RANK}")

    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer()).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Replacement: torch.rand -> tf.keras.backend.relu
        # Original: x = torch.rand(2,3,256,256, device=device)
        x = tf.keras.backend.relu(2,3,256,256, device=device)
        
        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    # dist.destroy_process_group() # Removed as init_process_group was replaced

if __name__ == "__main__":
    main()