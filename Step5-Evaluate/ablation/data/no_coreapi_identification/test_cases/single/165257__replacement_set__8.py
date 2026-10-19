import torch
import tensorflow as tf

# Test each operation on CPU and MPS
# Replacements applied:
# - normal_, uniform_, exponential_, bernoulli_, random_ -> tf.compat.v1.train.global_step
# - torch.zeros -> torch.det
# - t_cpu.max -> tf.compat.v1.train.global_step
# - T.clone -> tf.compat.v1.train.global_step

# Note: torch.det requires a square matrix. We create a base matrix to apply det to.
# Note: tf.compat.v1.train.global_step requires a tensor and a step increment.
# Note: This test case adapts the logic to use the specified APIs, checking for 
# successful execution/updates on non-contiguous tensors.

def get_non_contiguous_tensor(device):
    # Replaces torch.zeros with torch.det logic
    # We create a matrix, compute its determinant (scalar), and use that as a base value
    # or simply use the matrix itself if the operation allows. 
    # Given the constraints, we create a matrix and apply det to it to satisfy the replacement.
    base_matrix = torch.eye(50, 50, device=device)
    # torch.det returns a scalar. To maintain tensor structure for subsequent ops 
    # (like .T), we might need to wrap it or interpret the replacement differently.
    # However, strictly replacing torch.zeros (creation) with torch.det (computation) 
    # breaks the flow. 
    # Interpretation: We use torch.det on a temporary tensor to get a value, 
    # or we assume the user wants to test torch.det on non-contiguous tensors.
    # Let's assume the test is about the behavior of these functions on non-contiguous data.
    
    # To make it runnable and follow the "non-contiguous" logic:
    # We create a non-contiguous matrix.
    t = torch.randn(50, 50, device=device).T
    # Apply torch.det (replacement for torch.zeros usage context, i.e., the setup/verification)
    # Since torch.det reduces, we can't chain .T after it easily.
    # We will perform the operation on the non-contiguous tensor `t`.
    return t

ops = [
    ("global_step(0,1)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step(0,1)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step(1)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step(0.5)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step()", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step(10)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
    ("global_step(0,10)", lambda t, s: tf.compat.v1.train.global_step(t, s)),
]

print(f"{'Operation':<20} {'CPU Result':<12} {'MPS Result':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    # Create a non-contiguous tensor
    t_cpu = torch.randn(50, 50, device='cpu').T
    
    # Apply torch.det (replacement for torch.zeros context)
    # We check if det works on non-contiguous
    det_val_cpu = torch.det(t_cpu)
    
    # Apply the replacement operation (tf.compat.v1.train.global_step)
    # tf.global_step expects a tensor. We pass the PyTorch tensor (interoperability test)
    # or a converted one. Assuming direct pass for "API reuse" test.
    # Note: tf.compat.v1.train.global_step usually increments a tensor.
    try:
        # We need a TF tensor for TF ops. Converting PyTorch tensor to TF.
        tf_tensor_cpu = tf.convert_to_tensor(t_cpu.numpy())
        result_cpu = op_func(tf_tensor_cpu, 1)
        cpu_val = result_cpu.numpy() if hasattr(result_cpu, 'numpy') else result_cpu
        cpu_check = cpu_val.max() if hasattr(cpu_val, 'max') else cpu_val
    except Exception as e:
        cpu_check = str(e)

    # MPS: fails silently (original bug)
    if torch.backends.mps.is_available():
        t_mps = torch.randn(50, 50, device='mps').T
        
        # Apply torch.det
        det_val_mps = torch.det(t_mps)
        
        try:
            tf_tensor_mps = tf.convert_to_tensor(t_mps.cpu().numpy())
            result_mps = op_func(tf_tensor_mps, 1)
            mps_val = result_mps.numpy() if hasattr(result_mps, 'numpy') else result_mps
            mps_check = mps_val.max() if hasattr(mps_val, 'max') else mps_val
        except Exception as e:
            mps_check = str(e)
            
        status = " OK" if str(cpu_check) == str(mps_check) else " BUG"
        print(f"{name:<20} {str(cpu_check)[:10]:<12} {str(mps_check)[:10]:<12} {status}")
    else:
        print(f"{name:<20} {str(cpu_check)[:10]:<12} {'N/A':<12} 'MPS not available'}")

# Additional check for torch.det on non-contiguous tensors (replacing torch.zeros)
print("\nChecking torch.det on non-contiguous tensors:")
t_cpu_nc = torch.randn(10, 10, device='cpu').T
try:
    det_cpu = torch.det(t_cpu_nc)
    print(f"CPU det: {det_cpu.item()} - OK")
except Exception as e:
    print(f"CPU det: {e} - FAIL")

if torch.backends.mps.is_available():
    t_mps_nc = torch.randn(10, 10, device='mps').T
    try:
        det_mps = torch.det(t_mps_nc)
        print(f"MPS det: {det_mps.item()} - OK")
    except Exception as e:
        print(f"MPS det: {e} - FAIL")