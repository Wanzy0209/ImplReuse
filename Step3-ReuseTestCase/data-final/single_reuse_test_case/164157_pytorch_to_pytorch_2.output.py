import torch
import sys

# Reproduce the configuration settings from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    t0 = arg0 # size=(47,), stride=(1,), dtype=int64, device=cuda
    t1 = torch.tanh(t0) # size=(47,), stride=(1,), dtype=int64, device=cuda
    t2 = arg1 # size=(), stride=(), dtype=int64, device=cuda
    t3 = arg2 # size=(), stride=(), dtype=int64, device=cuda
    t4 = t2 * t3 # size=(), stride=(), dtype=int64, device=cuda
    t5 = t1.clone(); t5.fill_(t4.item()) # size=(47,), stride=(1,), dtype=int64, device=cuda
    t6 = arg3 # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
    t7 = arg4 # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
    t8 = arg5 # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
    t9 = torch.cat([t6, t6, t7, t8], dim=2) # size=(256, 88, 4), stride=(352, 4, 1), dtype=float16, device=cuda
    
    # --- Adapted Call Site ---
    # Original: t10 = t9.std(dim=2) # size=(256, 88), dtype=float16
    # Adapted to torch.arange: We generate a float16 range and reshape it to match the expected dimensions.
    # This tests if torch.arange with float16 dtype triggers the same IncompatibleTypeError during compilation.
    t10 = torch.arange(0, 256 * 88, dtype=torch.float16, device='cuda').view(256, 88)
    
    t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10) # size=(47, 88), stride=(88, 1), dtype=float16, device=cuda
    output = t11 + sentinel  # output tensor with sentinel for gradient flow
    return output

arg0 = torch.randint(0, 1000, [47], dtype=torch.int64, device='cuda')
arg1 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
arg2 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
arg3 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
arg4 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
arg5 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
sentinel = torch.tensor(0.0, dtype=torch.float16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager Mode
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled Mode
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    out_compiled.sum().backward()
    print('Compile Success! ')