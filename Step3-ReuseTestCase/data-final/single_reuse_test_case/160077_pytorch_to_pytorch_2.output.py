import torch

def f(tensors):
    # Adapted from xs.split(1, dim=0) to torch.stack(tensors, dim=0)
    return torch.stack(tensors, dim=0)

def backend(gm, inps):
    gm.print_readable()
    return gm

if torch.cuda.is_available():
    with torch.device("cuda"):
        # Adapted input: torch.stack requires a list/tuple of tensors
        xs = [torch.randn(2, 2, device="cuda"), torch.randn(2, 2, device="cuda")]

        # Eager works
        print("Eager execution:")
        f(xs)

        # Check if torch.compile fails similarly under device context
        print("Compiled execution:")
        try:
            torch.compile(f, backend=backend)(xs)
            print("Success")
        except AttributeError as e:
            print(f"AttributeError: {e}")
        except Exception as e:
            print(f"Exception: {e}")
else:
    print("CUDA not available, skipping test.")