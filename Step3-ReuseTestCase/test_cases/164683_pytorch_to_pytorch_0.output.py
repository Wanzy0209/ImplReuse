import torch
import sys

# Configuration based on the bug report to trigger the specific path
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, sentinel):
    t0 = arg0 # size=(4, 4), stride=(4, 1), dtype=int64, device=cuda
    # The bug involves the type handling of torch.tanh in the compiler
    t1 = torch.tanh(t0) 
    t2 = arg1 # size=(5,), stride=(1,), dtype=int64, device=cuda
    t3 = t2.min() # size=(), stride=(), dtype=int64, device=cuda
    t4 = t1.clone(); t4.fill_(t3.item()) # size=(4, 4), stride=(4, 1), dtype=int64, device=cuda
    t5 = arg2 # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t6 = torch.nn.functional.relu(t5) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t7 = torch.nn.functional.silu(t6) # size=(5000, 4), stride=(5000, 1), dtype=bfloat16, device=cuda
    t8 = torch.nn.functional.embedding(torch.clamp(t4, 0, t7.size(0) - 1).to(torch.long), t7) # size=(4, 4, 4), stride=(16, 4, 1), dtype=bfloat16, device=cuda
    t9 = t8.min() # size=(), stride=(), dtype=bfloat16, device=cuda
    output = t9 + sentinel  # output tensor with sentinel for gradient flow
    return output

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        sys.exit(0)

    arg0 = torch.randint(0, 1000, [4, 4], dtype=torch.int64, device='cuda')
    arg1 = torch.randint(0, 1000, [5], dtype=torch.int64, device='cuda')
    arg2 = torch.rand([5000, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Test Eager Mode
    try:
        out_eager = foo(arg0, arg1, arg2, sentinel)
        out_eager.sum().backward()
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')
        sys.exit(1)

    # Test Compiled Mode
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, sentinel)
        out_compiled.sum().backward()
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed! : {e}')
        sys.exit(1)