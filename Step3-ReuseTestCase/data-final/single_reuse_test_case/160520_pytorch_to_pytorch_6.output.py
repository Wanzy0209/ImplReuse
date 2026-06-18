import torch
import torch._inductor.config as config
import torch.profiler

# Configure Inductor with cpp_wrapper to trigger the specific code path
config.cpp_wrapper = True
# torch._logging.set_logs(output_code=True)

# Define a function that utilizes torch.lobpcg with non-tensor arguments
# This function will be compiled to test the behavior under Inductor
def lobpcg_func(A, k, niter, tol):
    # torch.lobpcg takes non-tensor arguments (k, niter, tol) which are
# susceptible to the redundant H2D-D2H memcpy issue described in the bug report.
    return torch.lobpcg(A, k=k, niter=niter, tol=tol)

# Compile the function using torch.compile
compiled_lobpcg = torch.compile(lobpcg_func)

# Setup input data: A symmetric positive definite matrix on CUDA
# LOBPCG requires the input matrix A to be symmetric positive definite
A = torch.randn(20, 20, device='cuda')
A = A @ A.T + 1e-3 * torch.eye(20, device='cuda')

# Run the test under a profiler and inside a DeviceContext
# This mimics the exact conditions under which the bug was reported
with torch.profiler.profile(
    with_stack=True,
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_lobpcg"),
) as prof:
    with torch.device("cuda"):
        for i in range(10):
            # Call the compiled function with tensor and non-tensor arguments
            # The bug manifests here if redundant copies occur for k, niter, or tol
            eigenvalues, eigenvectors = compiled_lobpcg(A, k=5, niter=20, tol=1e-5)
            
            # Perform a reduction to ensure the computation is not optimized away
            loss = eigenvalues.sum()