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

# Replacement: dist.init_process_group -> torch.hub.load_state_dict_from_url
# Original: dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
try:
    torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
except Exception:
    pass

class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad_out): return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x): return SimplistDoubleFn.apply(x)

def main():
    # Replacement: torch.device -> tf.compat.v1.graph_util.must_run_on_cpu
    # Original: device = torch.device(f"cuda:{LOCAL_RANK}")
    try:
        device = tf.compat.v1.graph_util.must_run_on_cpu(None, None)
    except Exception:
        device = torch.device("cpu")

    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer()).to(device)
    model = torch.compile(model)
    
    try:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    except Exception as e:
        print(f"DDP initialization failed (expected due to API replacement): {e}")
        return

    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Replacement: torch.rand -> torch.fliplr
        # Original: x = torch.rand(2,3,256,256, device=device)
        # fliplr requires an input tensor, so we create a dummy one to flip.
        dummy_input = torch.zeros(2, 3, 256, 256, device=device)
        x = torch.fliplr(dummy_input)

        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

if __name__ == "__main__":
    main()