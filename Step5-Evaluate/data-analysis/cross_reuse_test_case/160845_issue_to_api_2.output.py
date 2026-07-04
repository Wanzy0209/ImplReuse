import torch

def test_index_add_complex_block_lower_triangular():
    """
    Test case for index_add_ inconsistency on complex tensors (MPS).
    
    This test leverages the logic pattern of tf.linalg.LinearOperatorBlockLowerTriangular
    by constructing a block lower triangular matrix using index_add_. It verifies
    that complex values (both real and imaginary parts) are correctly added on the
    MPS backend compared to the CPU backend.
    """
    torch.manual_seed(42)
    
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        print("MPS device not available. Skipping test.")
        return

    device = "mps"
    cpu_device = "cpu"

    # Define dimensions for a 2x2 block matrix with block size 2 (Total 4x4)
    # Structure:
    # [[A, 0],
    #  [B, C]]
    block_size = 2
    total_size = 4

    # Initialize result tensors with zeros
    res_mps = torch.zeros((total_size, total_size), dtype=torch.complex64, device=device)
    res_cpu = torch.zeros((total_size, total_size), dtype=torch.complex64, device=cpu_device)

    # Helper to create a complex tensor with random real and imaginary parts
    def make_complex_block(rows, cols, dev):
        real = torch.randn(rows, cols, device=dev)
        imag = torch.randn(rows, cols, device=dev)
        return torch.complex(real, imag)

    # --- Block A (0,0) ---
    # Target indices: 0, 1
    # Source shape: (2, 4) -> [Block A, Zeros]
    A_mps = make_complex_block(block_size, block_size, device)
    A_cpu = A_mps.cpu()
    
    zero_block_mps = torch.zeros((block_size, block_size), dtype=torch.complex64, device=device)
    zero_block_cpu = torch.zeros((block_size, block_size), dtype=torch.complex64, device=cpu_device)
    
    A_src_mps = torch.cat([A_mps, zero_block_mps], dim=1)
    A_src_cpu = torch.cat([A_cpu, zero_block_cpu], dim=1)

    idx = torch.tensor([0, 1], dtype=torch.long)
    res_mps.index_add_(0, idx.to(device), A_src_mps)
    res_cpu.index_add_(0, idx.to(cpu_device), A_src_cpu)

    # --- Block B (1,0) ---
    # Target indices: 2, 3
    # Source shape: (2, 4) -> [Block B, Zeros]
    B_mps = make_complex_block(block_size, block_size, device)
    B_cpu = B_mps.cpu()
    
    B_src_mps = torch.cat([B_mps, zero_block_mps], dim=1)
    B_src_cpu = torch.cat([B_cpu, zero_block_cpu], dim=1)

    idx = torch.tensor([2, 3], dtype=torch.long)
    res_mps.index_add_(0, idx.to(device), B_src_mps)
    res_cpu.index_add_(0, idx.to(cpu_device), B_src_cpu)

    # --- Block C (1,1) ---
    # Target indices: 2, 3
    # Source shape: (2, 4) -> [Zeros, Block C]
    C_mps = make_complex_block(block_size, block_size, device)
    C_cpu = C_mps.cpu()
    
    C_src_mps = torch.cat([zero_block_mps, C_mps], dim=1)
    C_src_cpu = torch.cat([zero_block_cpu, C_cpu], dim=1)

    idx = torch.tensor([2, 3], dtype=torch.long)
    res_mps.index_add_(0, idx.to(device), C_src_mps)
    res_cpu.index_add_(0, idx.to(cpu_device), C_src_cpu)

    # Calculate the absolute difference between MPS and CPU results
    diff = (res_mps.cpu() - res_cpu).abs()
    max_diff = diff.max().item()

    print(f"Max absolute difference between MPS and CPU: {max_diff}")
    
    # Assert that the difference is within a reasonable floating point tolerance.
    # The original bug showed differences > 2.0, so 1e-5 is a safe threshold for correctness.
    assert max_diff < 1e-5, f"Bug detected: MPS and CPU results differ significantly ({max_diff})"

if __name__ == "__main__":
    test_index_add_complex_block_lower_triangular()