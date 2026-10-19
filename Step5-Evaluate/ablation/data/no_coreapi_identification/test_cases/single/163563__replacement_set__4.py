import torch
import sys

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # t0 = arg0
    # Replacement: torch.sigmoid -> torch.square
    t1 = torch.square(arg0)

    # t2 = arg1
    # Replacement: torch.sigmoid -> torch.square
    t3 = torch.square(arg1)

    # t4 = arg2
    # Replacement: torch.exp -> torch.sigmoid
    t5 = torch.sigmoid(arg2)

    # Replacement: torch.baddbmm -> torch.as_tensor
    # torch.as_tensor takes data, dtype, device. We pass t1 to maintain tensor flow.
    t6 = torch.as_tensor(t1)

    # Original reshape was (193, 386, 459).
    # Since torch.unique returns 1D, t6 is 1D.
    # We reshape to -1 to keep it runnable.
    t7 = t6.reshape(-1)
    output = t7
    return output

# Replacement: torch.rand -> torch.unique
# torch.unique requires an input tensor. We use torch.randn to generate it.
# Note: torch.unique returns a 1D tensor.
arg0 = torch.unique(torch.randn([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg1 = torch.unique(torch.randn([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True))
arg2 = torch.unique(torch.randn([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True))

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')