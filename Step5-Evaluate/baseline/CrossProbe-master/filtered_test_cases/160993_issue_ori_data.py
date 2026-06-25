import torch

def test_index_copy_mps():
    if not torch.backends.mps.is_available():
        print('MPS not available')
        return
    
    # Create test tensors similar to the failing test
    dest = torch.zeros(3, 3, 3, dtype=torch.float32, device='mps')
    index = torch.tensor([0, 2, 1], device='mps')
    source = torch.randn(3, 3, 3, dtype=torch.float32, device='mps')
    
    # Perform index_copy
    dest.index_copy_(0, index, source)
    
    # Compare with CPU implementation
    dest_cpu = torch.zeros(3, 3, 3, dtype=torch.float32)
    index_cpu = torch.tensor([0, 2, 1])
    source_cpu = source.cpu()
    dest_cpu.index_copy_(0, index_cpu, source_cpu)
    
    # Check if results match
    if not torch.allclose(dest.cpu(), dest_cpu, atol=1e-5, rtol=1.3e-6):
        print('MPS index_copy produces incorrect results')
        print(f'Max diff: {(dest.cpu() - dest_cpu).abs().max()}')
    else:
        print('Results match')

test_index_copy_mps()