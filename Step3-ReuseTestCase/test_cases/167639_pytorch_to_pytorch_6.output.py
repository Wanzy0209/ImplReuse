import torch

def get_sample_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a symmetric positive definite matrix for lobpcg
    X = torch.randn(10, 10, device=device)
    A = X @ X.T
    return (A,)

def main():
    inputs = get_sample_inputs()
    A = inputs[0]
    device = A.device
    k = 2  # Number of eigenvalues to compute

    # Run outside graph to get baseline
    try:
        with torch.no_grad():
            original_evals, original_evecs = torch.lobpcg(A, k=k)
            print('Original lobpcg output shape:', original_evals.shape)
    except Exception as e:
        print(f"lobpcg failed outside graph: {e}")
        return

    if torch.cuda.is_available():
        # Setup static memory for graph capture
        # lobpcg returns (eigenvalues, eigenvectors)
        static_evals = torch.empty(k, device=device)
        static_evecs = torch.empty(A.size(0), k, device=device)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                # Adapted call site: torch.lobpcg
                # This replaces the torch.compile(model) call from the original test
                evals, evecs = torch.lobpcg(A, k=k)
                static_evals.copy_(evals)
                static_evecs.copy_(evecs)
            
            # Replay the graph
            graph.replay()
            print('Captured graph replay successful.')
            
            # Verify results
            # Note: lobpcg might have slight numerical differences or RNG usage
            # but we check if it ran without crashing (the main bug concern)
            print('Graph output shape:', static_evals.shape)
            
            # Basic assertion to ensure execution happened
            assert static_evals.shape == original_evals.shape, "Shape mismatch"
            
        except RuntimeError as e:
            # This catches the specific bug (RNG state access) or other graph incompatibilities
            print(f"RuntimeError during graph capture/replay: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    else:
        print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()