import torch

def test_torch_mm_with_sparse_bug_data():
    """
    Test case for torch.mm (dense matrix multiplication) using data 
    derived from the torch.sparse.mm bug report (Issue 167716).
    
    The bug report demonstrates a crash when using torch.sparse.mm with two sparse inputs.
    This test adapts the scenario to torch.mm by converting inputs to dense, 
    verifying that the dense multiplication logic handles the data correctly.
    """
    torch.manual_seed(42)

    # Recreate the sparse tensors from the bug report
    indices_A = torch.tensor([[0, 1, 2], [0, 2, 3]])
    values_A = torch.tensor([1.0, 2.0, 3.0])
    A_sparse = torch.sparse_coo_tensor(indices_A, values_A, size=(3, 4))

    indices_B = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]])
    values_B = torch.tensor([4.0, 5.0, 6.0, 7.0])
    B_sparse = torch.sparse_coo_tensor(indices_B, values_B, size=(4, 2))

    # Adaptation: torch.mm expects dense tensors.
    # We convert the sparse inputs to dense to verify the multiplication logic.
    A_dense = A_sparse.to_dense()
    B_dense = B_sparse.to_dense()

    # Original call site in bug report: C = torch.sparse.mm(A, B)
    # Adapted call site for similar API:
    C = torch.mm(A_dense, B_dense)

    # Verify the result
    # A_dense:
    # [[1., 0., 0., 0.],
    #  [0., 0., 2., 0.],
    #  [0., 0., 0., 3.]]
    # B_dense:
    # [[4., 0.],
    #  [0., 5.],
    #  [0., 6.],
    #  [0., 7.]]
    # Expected Result:
    # [[ 4.,  0.],
    #  [ 0., 12.],
    #  [ 0., 21.]]
    expected = torch.tensor([[4., 0.], [0., 12.], [0., 21.]])

    assert torch.equal(C, expected), f"Expected {expected}, but got {C}"
    print("Test passed: torch.mm works correctly with the dense version of the bug report data.")

if __name__ == "__main__":
    test_torch_mm_with_sparse_bug_data()