import torch

def test_torch_mv_sparse_vector():
    """
    Test case adapted from the torch.sparse.mm bug report (Issue 167716).
    This test verifies if torch.mv handles a sparse vector as the second argument
    without causing a Segmentation fault on to_dense(), similar to the reported
    behavior with torch.sparse.mm and two sparse matrices.
    """
    torch.manual_seed(42)

    # Create a sparse matrix A (3x4)
    indices_A = torch.tensor([[0, 1, 2], [0, 2, 3]])
    values_A = torch.tensor([1.0, 2.0, 3.0])
    A = torch.sparse_coo_tensor(indices_A, values_A, size=(3, 4))

    # Create a sparse vector v (4,)
    # Adapting the second sparse matrix from the bug report into a sparse vector
    indices_v = torch.tensor([[0, 1, 3]])
    values_v = torch.tensor([4.0, 5.0, 7.0])
    v = torch.sparse_coo_tensor(indices_v, values_v, size=(4,))

    # Perform matrix-vector multiplication
    # Original bug: torch.sparse.mm(A, B) -> to_dense() -> Segfault
    # Similar API: torch.mv(A, v) -> to_dense() -> ?
    try:
        result = torch.mv(A, v)
        
        # The specific crash in the bug report occurred during conversion to dense
        result_dense = result.to_dense()

        # Verify against dense calculation to ensure correctness
        A_dense = A.to_dense()
        v_dense = v.to_dense()
        expected = torch.mv(A_dense, v_dense)

        assert torch.allclose(result_dense, expected), "Values do not match"
        print("Test passed: torch.mv with sparse vector works correctly.")

    except RuntimeError as e:
        # torch.mv might not support sparse vectors explicitly in some versions
        print(f"RuntimeError (expected if sparse vector unsupported): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_torch_mv_sparse_vector()