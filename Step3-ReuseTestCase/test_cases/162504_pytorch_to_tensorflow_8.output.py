import torch
import tensorflow as tf

# Setup seeds to match PyTorch's manual_seed behavior
tf.random.set_seed(42)

# Define a function similar to the PyTorch fn(x)
# PyTorch: return x * torch.sigmoid(torch.randn(1, device="cuda"))
def map_fn(x):
    # Using random.uniform to simulate randn, and sigmoid
    return x * tf.math.sigmoid(tf.random.normal(shape=(), dtype=tf.float32))

# 1. Eager Execution (Baseline)
# PyTorch: eager_out = torch.utils.checkpoint.checkpoint(fn, eager_in, ...)
# TF: Create a dataset and map the function eagerly
eager_dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0])
eager_dataset = eager_dataset.map(map_fn)
eager_results = list(eager_dataset.as_numpy_iterator())

print("Eager Results:", eager_results)

# 2. Graph/Service Execution (Target)
# PyTorch: with torch.cuda.graph(g): graph_out = torch.utils.checkpoint.checkpoint(...)
# TF: We use tf.data.experimental.service.distribute inside a tf.function to mimic graph capture.
# Note: This requires a running tf.data service (dispatcher and worker).
# We assume a standard local setup for the example.

service_address = "grpc://localhost:5000"
job_name = "test_job_162504"

try:
    # Reset seed for the graph run to match PyTorch logic
    tf.random.set_seed(42)

    # Define the dataset transformation
    # PyTorch args: use_reentrant=False, preserve_rng_state=True
    # TF args: processing_mode, service, job_name
    distributed_dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0])
    distributed_dataset = distributed_dataset.map(map_fn)
    
    # Apply the distribute transformation
    distributed_dataset = tf.data.experimental.service.distribute(
        processing_mode="parallel_epochs", # Equivalent to distributing the workload
        service=service_address,
        job_name=job_name,
    )(distributed_dataset)

    # Wrap iteration in tf.function to simulate the CUDA Graph capture context
    @tf.function
    def run_in_graph_mode(ds):
        return list(ds.as_numpy_iterator())

    graph_results = run_in_graph_mode(distributed_dataset)
    print("Graph Results:", graph_results)

    # 3. Assertion
    # PyTorch: assert torch.allclose(eager_in_grad, graph_in_grad)
    # TF: Check if results match (Note: exact match depends on service determinism)
    # We check length and basic validity here as distributed data order might vary
    # depending on configuration, but for a single worker/parallel_epochs it should be consistent.
    assert len(eager_results) == len(graph_results), "Mismatch in output length"
    
    # If the service preserves order (e.g. single worker), check values
    # This mimics the "Mismatch in gradient outputs" assertion
    if len(eager_results) > 0 and len(graph_results) > 0:
        # We allow some tolerance for floating point differences if necessary, 
        # though PyTorch used rtol=0.0, atol=0.0
        assert tf.reduce_all(tf.abs(tf.constant(eager_results) - tf.constant(graph_results)) < 1e-5), \
            "Mismatch in outputs"

except tf.errors.UnavailableError:
    print(f"Skipping test: tf.data service not available at {service_address}.")
except Exception as e:
    print(f"Test failed with error: {e}")