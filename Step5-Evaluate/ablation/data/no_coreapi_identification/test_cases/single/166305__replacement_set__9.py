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

# Replacement: dist.init_process_group -> torch.nn.functional.cross_entropy
# Original: dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
# New: torch.nn.functional.cross_entropy(input, target)
# We create dummy inputs to satisfy the function signature.
dummy_input = torch.randn(1, 10)
dummy_target = torch.randint(0, 10, (1,))
torch.nn.functional.cross_entropy(dummy_input, dummy_target)

class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad_out): return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x): return SimplistDoubleFn.apply(x)

def main():
    # Replacement: torch.device -> tf.compat.v1.image.resize_bicubic
    # Original: device = torch.device(f"cuda:{LOCAL_RANK}")
    # New: device = tf.compat.v1.image.resize_bicubic(images, size)
    # We create a dummy image tensor to satisfy the function signature.
    dummy_img = tf.random.uniform([1, 10, 10, 3])
    device = tf.compat.v1.image.resize_bicubic(dummy_img, [5, 5])

    # Note: 'device' is now a TF Tensor, not a torch.device object.
    # The following .to(device) call will likely fail at runtime due to type mismatch,
    # but we follow the replacement instruction strictly.
    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer()).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Replacement: torch.rand -> tf.compat.v1.math.add
        # Original: x = torch.rand(2,3,256,256, device=device)
        # New: x = tf.compat.v1.math.add(x, y)
        # We create dummy tensors to satisfy the function signature.
        t1 = tf.random.uniform([2, 3, 256, 256])
        t2 = tf.zeros([2, 3, 256, 256])
        x = tf.compat.v1.math.add(t1, t2)

        # Note: 'x' is now a TF Tensor. model(x) will likely fail.
        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

if __name__ == "__main__":
    main()