import os, torch
import torch.nn as nn, torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tensorflow as tf

dynamo.config.optimize_ddp = True

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Replacing dist.init_process_group with torch.distributed.scatter_object_list
# Note: init_process_group is required for DDP and scatter_object_list to function.
# We keep it to ensure the test case is runnable, but use scatter_object_list for data distribution.
dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad_out): return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x): return SimplistDoubleFn.apply(x)

def main():
    # Replacing torch.device with tf.compat.v1.colocate_with
    # We still need the torch device for the model, but we use the TF API for data context.
    device = torch.device(f"cuda:{LOCAL_RANK}")
    
    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer()).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Replacing torch.rand with tf.nn.relu
        # Rank 0 generates data using TensorFlow
        if LOCAL_RANK == 0:
            # Create a TF tensor
            tf_x = tf.random.uniform((2,3,256,256))
            # Use tf.compat.v1.colocate_with to manage device placement for the TF operation
            with tf.compat.v1.colocate_with(tf_x):
                # Use tf.nn.relu instead of torch.rand
                tf_x = tf.nn.relu(tf_x)
            
            # Convert to PyTorch tensor
            x = torch.from_numpy(tf_x.numpy()).to(device)
        else:
            x = None

        # Replacing local data generation logic with torch.distributed.scatter_object_list
        # Scatter the input data from rank 0 to all other ranks
        scatter_list = [x] if LOCAL_RANK == 0 else [None]
        dist.scatter_object_list(scatter_list, scatter_list, src=0)
        x = scatter_list[0]

        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()