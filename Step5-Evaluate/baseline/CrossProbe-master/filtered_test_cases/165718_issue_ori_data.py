import torch
torch._dynamo.config.capture_scalar_outputs = True

def reproduce_bug():
    # Create float16 tensor
    a = torch.full((13, 14), 0.5, dtype=torch.float16, device='cuda')
    b = torch.full((13, 14), 0.3, dtype=torch.float16, device='cuda')
    result_float16 = torch.matmul(a, b)  # Result is float16
    
    # Create float32 tensor
    c = torch.full((13, 14), 0.2, dtype=torch.float32, device='cuda')
    d = torch.full((13, 14), 0.1, dtype=torch.float32, device='cuda')
    result_float32 = torch.matmul(c, d)  # Result is float32
    
    # This will cause the dtype mismatch error
    try:
        final_result = torch.add(result_float16, result_float32)
    except RuntimeError as e:
        print(f"Error: {e}")
        return False
    return True

if torch.cuda.is_available():
    reproduce_bug()