import torch
import sys

# Reproduce the configuration settings from the original bug report
# as these are often crucial for triggering the specific compilation path.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    # Reproduce the tensor transformations leading to the non-contiguous tensor
    t0 = arg0 # size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1)
    t1 = t0.mean(dim=0) # size=(503, 64, 504), stride=(32192, 64, 1) - Non-contiguous
    t2 = torch.nn.functional.relu(t1) # size=(503, 64, 504), stride=(32192, 64, 1)
    
    # The original bug used arg1 to create a convolution filter.
    # We will use arg1 to determine the number of classes for one_hot to maintain
    # a relationship with the second input, though one_hot primarily operates on indices.
    num_classes = int(arg1.shape[1]) 
    
    # Convert the float tensor to indices for one_hot.
    # We use abs() and modulo to ensure valid indices.
    indices = t2.abs().long() % num_classes
    
    # Replace torch.nn.functional.conv1d with torch.nn.functional.one_hot
    # This tests if the similar API handles the non-contiguous input strides correctly.
    output = torch.nn.functional.one_hot(indices, num_classes=num_classes)
    
    return output

# Setup inputs matching the original bug report's dimensions and properties
# Note: Using 'cuda' if available to match the original environment, otherwise 'cpu'.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

arg0 = torch.rand([4, 503, 64, 504], dtype=torch.float32, device=device, requires_grad=True)
arg1 = torch.rand([5, 16, 1, 64], dtype=torch.float32, device=device, requires_grad=True)

if __name__ == '__main__':
    print("Running Eager Mode...")
    try:
        out_eager = foo(arg0, arg1)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed! : {e}')
        sys.exit(1)

    print("\nRunning Compiled Mode...")
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1)
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed! : {e}')
        sys.exit(1)

    print("\nVerifying Results...")
    # Check for divergence between eager and compiled outputs
    if torch.equal(out_eager, out_compiled):
        print("Verification Passed: Eager and Compiled outputs are identical. ")
    else:
        print("Verification Failed: Eager and Compiled outputs diverge. ")
        print(f"Max difference: {(out_eager - out_compiled).abs().max()}")