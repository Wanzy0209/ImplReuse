import torch
import torch.nn.functional as F


def f(a, weight):
    # Adapted print to check properties relevant to prelu (shape/ndim)
    # similar to how the original bug checked layout.
    print(f"a.shape: {a.shape}, weight.shape: {weight.shape}")

    return F.prelu(a, weight)


# Create inputs suitable for torch.nn.functional.prelu
# a: input tensor (e.g., batch_size=2, channels=3)
# weight: 1D tensor (size=3 for channels)
a = torch.randn(2, 3)
weight = torch.randn(3)

print("Direct call:")
result_direct = f(a, weight)
print(result_direct)

print("\nVJP call:")
# The original bug occurred during the construction of the vjp (forward pass)
# because the layout was lost. We test if vjp works correctly here.
try:
    vjp_fn = torch.func.vjp(f, a, weight)[1]
    print("VJP construction successful.")
    
    # Optional: Verify the vjp function works by calling it with cotangents
    # This ensures the backward pass also handles the tensor attributes correctly.
    cotangents = torch.ones_like(result_direct)
    grads = vjp_fn(cotangents)
    print("VJP execution successful.")
    print(f"Gradients computed: {grads[0].shape}, {grads[1].shape}")

except Exception as e:
    print(f"VJP failed with error: {e}")