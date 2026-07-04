import torch
import sys

# Reproduce the configuration from the bug report
# Added checks to handle environments where these internal modules are not available
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    t0 = arg0 # size=(36, 7112, 1, 1), stride=(7112, 1, 1, 1), dtype=bfloat16, device=cuda
    t1 = t0.reshape((28, 24, 3, 127)) # size=(28, 24, 3, 127), stride=(9144, 381, 127, 1), dtype=bfloat16, device=cuda
    
    # Original: t2 = t1.var(dim=2)
    # Adaptation: Replace torch.var with torch.promote_types to test type promotion logic.
    # We simulate the type promotion that might occur during variance calculation (often to float32)
    # and verify if torch.promote_types handles the bfloat16 context correctly under emulation.
    promoted_dtype = torch.promote_types(t1.dtype, torch.float32)
    
    # To maintain the tensor shape and flow for the rest of the graph, 
    # we perform a reduction (sum) and cast to the promoted dtype.
    t2 = t1.sum(dim=2).to(promoted_dtype)
    
    t3 = arg1 # size=(30, 24), stride=(30, 1), dtype=int64, device=cuda
    t4 = arg2 # size=(512, 127), stride=(512, 1), dtype=bfloat16, device=cuda
    t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4) # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    t6 = arg3 # size=(30, 24, 15), stride=(720, 24, 1), dtype=bfloat16, device=cuda
    t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0) # size=(30, 24, 16), stride=(384, 16, 1), dtype=bfloat16, device=cuda
    t8 = arg4 # size=(30, 4, 16, 127), stride=(8128, 2032, 127, 1), dtype=bfloat16, device=cuda
    t9 = t8.sum(dim=1) # size=(30, 16, 127), stride=(2032, 127, 1), dtype=bfloat16, device=cuda
    t10 = torch.baddbmm(t5, t7, t9) # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
    
    # t2 is now float32 (due to promotion), t10 is bfloat16. 
    # torch.cat will promote t10 to float32.
    t11 = torch.cat([t2, t10], dim=0) # size=(58, 24, 127), stride=(3048, 127, 1)
    
    output = t11 + sentinel  # output tensor with sentinel for gradient flow
    return output

# Check for CUDA availability to avoid runtime errors if CUDA is not present
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("Warning: CUDA not available. Falling back to CPU. This might affect bfloat16 behavior.")

arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device=device, requires_grad=True)
arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device=device)
arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device=device, requires_grad=True)
arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device=device, requires_grad=True)
arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device=device, requires_grad=True)
sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device=device, requires_grad=True)

if __name__ == '__main__':
    print("Testing Eager Execution...")
    try:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_eager.sum().backward()
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')
        sys.exit(1)

    print("\nTesting Compiled Execution...")
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
            out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
            out_compiled.sum().backward()
            print('Compile Success! ')
        except Exception as e:
            print(f'Compile Failed! : {e}')
            sys.exit(1)
    else:
        print("Skipping Compiled Execution: torch.compile is not available in this environment.")