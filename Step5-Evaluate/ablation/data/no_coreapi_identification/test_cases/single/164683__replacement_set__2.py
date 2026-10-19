import torch
import torch.distributed as dist
import tempfile
import os
import sys

# Setup for distributed (replacement for torch.compile context)
# We use a simple setup to allow the script to run if distributed is available.
# Note: torch.distributed.reduce requires a process group.
try:
    if not dist.is_initialized():
        # Using tcp for local single process if supported.
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
except Exception as e:
    print(f"Warning: Distributed init failed (expected in some envs): {e}")
    # We continue, but the reduce step might fail.

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

# Create a temp file for torch.load replacement
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
temp_file_name = temp_file.name
temp_file.close()

def foo(arg0, arg1, arg2, sentinel, temp_path):
    t0 = arg0
    t1 = torch.tanh(t0)
    t2 = arg1
    t3 = t2.min()
    t4 = t1.clone(); t4.fill_(t3.item())

    # Replacement: torch.clamp -> torch.load
    # We save the clamped result to the temp file to load it back
    clamped_val = torch.clamp(t4, 0, 4999) # t7.size(0) is 5000
    torch.save(clamped_val, temp_path)
    loaded_indices = torch.load(temp_path)

    t5 = arg2
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.relu6
    t6 = torch.nn.functional.relu6(t5)
    # Replacement: torch.nn.functional.silu -> torch.nn.functional.selu
    t7 = torch.nn.functional.selu(t6)

    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.cross_entropy
    # embedding(indices, weights) -> cross_entropy(input, target)
    # t7 is (5000, 4), loaded_indices is (4, 4)
    # We slice t7 to (4, 4) to match batch size of indices
    # We use loaded_indices as target (must be class indices 0..C-1)
    # We use the first column of indices as target for shape (4).
    t8 = torch.nn.functional.cross_entropy(t7[:4], loaded_indices[:, 0].long())

    t9 = t8.min()
    output = t9 + sentinel
    return output

# Replacement: torch.randint -> torch.amin
# arg0 needs to be (4, 4) int64
# We create a larger tensor, take amin, and expand to shape
dummy0 = torch.randn(10, 10, dtype=torch.float32, device='cuda')
arg0 = torch.amin(dummy0, dim=0).expand(4, 4).to(torch.int64)

# arg1 needs to be (5,) int64
dummy1 = torch.randn(10, dtype=torch.float32, device='cuda')
arg1 = torch.amin(dummy1, dim=0).expand(5).to(torch.int64)

# Replacement: torch.rand -> torch.fliplr
# arg2 needs to be (5000, 4) bf16
dummy2 = torch.randn([5000, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.fliplr(dummy2)

# Replacement: torch.tensor -> torch.isinf
# sentinel needs to be a scalar
dummy_sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda')
sentinel = torch.isinf(dummy_sentinel)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, sentinel, temp_file_name)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Replacement: torch.compile -> torch.distributed.reduce
    # Instead of compiling, we run the function (or reuse output) and reduce it.
    # To mimic the "compile" step, we run the function again (or clone output) and apply reduce.
    # Note: distributed reduce modifies tensor in place.
    out_compiled = out_eager.clone().detach() # Clone to simulate a separate run's output
    out_compiled.requires_grad = True 

    try:
        # dst=0, op=Sum (default)
        dist.reduce(out_compiled, dst=0)
        print('Distributed Reduce Success! ')
        # Check divergence (original intent)
        if torch.allclose(out_eager, out_compiled):
            print("Divergence Check: Match ")
        else:
            print("Divergence Check: Mismatch ")
    except Exception as e:
        print(f'Distributed Reduce Failed: {e}')

    # Cleanup
    os.remove(temp_file_name)
    if dist.is_initialized():
        dist.destroy_process_group()