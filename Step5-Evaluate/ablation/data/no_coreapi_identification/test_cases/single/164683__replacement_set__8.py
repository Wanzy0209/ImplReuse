import torch
import torch.distributed as dist
import tensorflow as tf
import sys
import os

# Setup for torch.distributed
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
# Initialize process group for single-process testing
dist.init_process_group(backend='gloo', rank=0, world_size=1)

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, sentinel):
    t0 = arg0
    t1 = torch.tanh(t0)
    t2 = arg1
    t3 = t2.min()
    t4 = t1.clone(); t4.fill_(t3.item())
    t5 = arg2
    
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.softsign
    t6 = torch.nn.functional.softsign(t5)
    
    # Replacement: torch.nn.functional.silu -> torch.nn.functional.celu
    t7 = torch.nn.functional.celu(t6)

    # Replacement: torch.clamp -> torch.tril
    # Original: torch.clamp(t4, 0, t7.size(0) - 1)
    # We perform the replacement on the argument logic
    indices = torch.tril(t4).to(torch.long)

    # Replacement: torch.nn.functional.embedding -> torch.nn.LazyConvTranspose3d
    # Original: torch.nn.functional.embedding(indices, t7)
    # We instantiate the layer. Since it's Lazy, we do it on first call.
    if not hasattr(foo, "conv_layer"):
        # t7 shape is (5000, 4). ConvTranspose3d expects 5D input (N, C, D, H, W).
        # We assume in_channels=4 based on t7.
        foo.conv_layer = torch.nn.LazyConvTranspose3d(out_channels=4, kernel_size=3).to('cuda')
    
    # Reshape t7 to 5D: (N, C, D, H, W) -> (5000, 4, 1, 1, 1)
    t7_5d = t7.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    t8 = foo.conv_layer(t7_5d)

    t9 = t8.min()
    output = t9 + sentinel
    return output

# Replacement: torch.randint -> torch.argmax
# arg0: size=(4, 4), dtype=int64, device=cuda
# We use argmax on a larger tensor to generate the (4, 4) shape
arg0 = torch.argmax(torch.randn(4, 4, 1000, device='cuda'), dim=2).to(torch.int64)

# arg1: size=(5,), dtype=int64, device=cuda
arg1 = torch.argmax(torch.randn(5, 1000, device='cuda'), dim=1).to(torch.int64)

# Replacement: torch.rand -> tf.compat.v1.math.add
# arg2: size=(5000, 4), dtype=bfloat16, device=cuda, requires_grad=True
# We generate two random tensors in TF and add them, then convert to Torch
t1_tf = tf.random.uniform((5000, 4))
t2_tf = tf.random.uniform((5000, 4))
arg2_tf = tf.compat.v1.math.add(t1_tf, t2_tf)
arg2 = torch.from_numpy(arg2_tf.numpy()).to(torch.bfloat16).cuda().requires_grad_(True)

# Replacement: torch.tensor -> torch.cummin
# sentinel: scalar, dtype=bfloat16, device=cuda, requires_grad=True
# We create a tensor, perform cummin, and take the first element
tmp_sentinel = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device='cuda', requires_grad=True)
sentinel = torch.cummin(tmp_sentinel, dim=0)[0][0]

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Replacement: torch.compile -> torch.distributed.send_object_list
    # Original: compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    # We replace the compile call with the distributed send call.
    # Since send_object_list does not return a callable, we cannot call it like a function.
    # To maintain the test structure (running the function again), we simply call foo again.
    torch.distributed.send_object_list([foo], dst=0)
    
    out_compiled = foo(arg0, arg1, arg2, sentinel)
    out_compiled.sum().backward()
    print('Compile Success! ')