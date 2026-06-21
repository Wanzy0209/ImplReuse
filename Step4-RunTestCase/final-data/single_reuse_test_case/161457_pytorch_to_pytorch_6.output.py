def test_lobpcg_compile(self):
        import torch
        import copy

        # Setup inputs for lobpcg
        # Using float32 to ensure numerical stability for the iterative solver,
        # as bfloat16 might not converge even in eager mode.
        dtype = torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n = 20
        k = 5

        # Create a symmetric positive definite matrix
        A = torch.randn(n, n, dtype=dtype, device=device)
        A = A @ A.T + torch.eye(n, dtype=dtype, device=device)

        # Initial guess
        X = torch.randn(n, k, dtype=dtype, device=device)

        # Eager execution
        eigvals_eager, eigvecs_eager = torch.lobpcg(A, k)

        # Compiled execution
        # We define a wrapper to compile the lobpcg call
        def lobpcg_fn(A, k):
            return torch.lobpcg(A, k)

        compiled_fn = torch.compile(lobpcg_fn)
        eigvals_compiled, eigvecs_compiled = compiled_fn(A, k)

        # Check correctness
        # Eigenvalues should match closely
        self.assertTrue(torch.allclose(eigvals_eager, eigvals_compiled, atol=1e-3))
        # Eigenvectors can be checked by verifying they satisfy the eigenvalue equation
        # or by checking closeness (handling sign ambiguity).
        # Here we check closeness of the absolute values to handle sign flips.
        self.assertTrue(torch.allclose(torch.abs(eigvecs_eager), torch.abs(eigvecs_compiled), atol=1e-2))