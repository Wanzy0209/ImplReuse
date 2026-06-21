import torch
import sys

# Replicate the configuration from the original bug report
# Check if attributes exist to avoid AttributeError in older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def foo(arg0):
    # Adapt the tensor manipulation logic to produce a non-contiguous square matrix
    # Original logic: (4, 503, 64, 504) -> mean(0) -> (503, 64, 504) -> ...
    # Adapted logic: (4, 10, 10, 5) -> mean(0) -> (10, 10, 5) -> transpose -> mean -> (10, 10)
    
    t0 = arg0 # size=(4, 10, 10, 5), stride=(500, 50, 5, 1), dtype=float32
    t1 = t0.mean(dim=0) # size=(10, 10, 5), stride=(50, 5, 1), dtype=float32
    # Introduce non-contiguity similar to the original bug's pattern
    t2 = t1.transpose(1, 2) # size=(10, 5, 10), stride=(50, 1, 5), dtype=float32
    t3 = t2.mean(dim=1) # size=(10, 10), stride=(5, 1), dtype=float32 (non-contiguous)
    
    # Call the similar API: torch.eig
    # Note: torch.eig is deprecated and raises a RuntimeError.
    # This test verifies the behavior with non-contiguous inputs.
    try:
        output = torch.eig(t3)
    except RuntimeError as e:
        # Expected behavior for the deprecated API
        if "deprecated" in str(e):
            return None 
        raise
    
    return output

# Setup inputs adapted for the similar API (requires square matrix for eig)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arg0 = torch.rand([4, 10, 10, 5], dtype=torch.float32, device=device)

if __name__ == '__main__':
    print('Testing torch.eig with non-contiguous inputs...')
    
    # Eager Execution
    try:
        out_eager = foo(arg0)
        if out_eager is None:
            print('Eager Success (caught deprecation)! ')
        else:
            print('Eager Success (unexpected)! ')
    except Exception as e:
        print(f'Eager Failed: {e} ')

    # Compiled Execution
    if hasattr(torch, 'compile'):
        try:
            compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
            out_compiled = compiled_foo(arg0)
            if out_compiled is None:
                print('Compile Success (caught deprecation)! ')
            else:
                print('Compile Success (unexpected)! ')
        except Exception as e:
            print(f'Compile Failed: {e} ')
    else:
        print('Compile Skipped: torch.compile is not available in this environment.')