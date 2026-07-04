import torch
import sys

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available (requires PyTorch 2.0+).")
    sys.exit(0)

# Check if torch._inductor is available
if not hasattr(torch, '_inductor'):
    print("Skipping test: torch._inductor is not available.")
    sys.exit(0)

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def test_convolution_backward_stride_divergence():
    """
    Test case for Issue 163604: Eager/Compile Divergence in convolution_backward.
    
    This test leverages the semantic pattern of tf.compat.v1.train.SessionRunArgs
    to structure the execution, separating the 'fetches' (the model logic),
    'feed_dict' (inputs), and 'options' (compile settings).
    """

    # --- 1. Define the Model (Fetches) ---
    # This corresponds to the 'fetches' argument in SessionRunArgs.
    # It contains the logic that triggers the bug in torch.ops.aten.convolution_backward.default
    def model_fn(arg0, arg1):
        t0 = arg0
        # t1 = t0.mean(dim=0) often results in specific strides that might cause 
        # divergence in the compiler's stride assumptions.
        t1 = t0.mean(dim=0) 
        t2 = torch.nn.functional.relu(t1)
        t3 = arg1
        t4 = t3.sum(dim=0)
        t5 = t4.transpose(2, 1)
        # The backward pass of this conv1d triggers the assertion failure.
        t6 = torch.nn.functional.conv1d(t2, t5, stride=1, padding=0)
        return t6

    # --- 2. Define Inputs (Feed Dict) ---
    # This corresponds to the 'feed_dict' argument in SessionRunArgs.
    # We use the specific shapes from the bug report.
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    arg0 = torch.rand([4, 503, 64, 504], dtype=torch.float32, device='cuda', requires_grad=True)
    arg1 = torch.rand([5, 16, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True)
    
    feed_dict = {'arg0': arg0, 'arg1': arg1}

    # --- 3. Define Execution Options ---
    # This corresponds to the 'options' argument in SessionRunArgs.
    # Here we define the compilation settings that trigger the bug.
    options = {
        'fullgraph': True,
        'dynamic': True
    }

    # --- 4. Execution Logic (Session Run) ---
    # Mimicking the session execution flow: Run Eager, then Run Compiled.
    
    print("Running Eager Execution...")
    try:
        out_eager = model_fn(**feed_dict)
        out_eager.sum().backward()
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        raise

    # Reset gradients for the compiled run
    for t in feed_dict.values():
        t.grad = None

    print("Running Compiled Execution (torch.compile)...")
    try:
        # Apply the 'options' to the compilation
        compiled_model = torch.compile(model_fn, **options)
        
        out_compiled = compiled_model(**feed_dict)
        out_compiled.sum().backward()
        print('Compile Success! ')
    except AssertionError as e:
        # Catching the specific assert_size_stride error mentioned in the issue
        if "assert_size_stride" in str(e):
            print(f"Bug Reproduced! \nError: {e}")
            raise
        else:
            raise
    except Exception as e:
        print(f"Compiled run failed with unexpected error: {e}")
        raise

if __name__ == '__main__':
    test_convolution_backward_stride_divergence()