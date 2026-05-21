import torch
import tensorflow as tf
import functools

# Mimic the CustomPolicy class from the PyTorch bug report
class ServiceConfig:
    def __init__(self, name):
        self.name = name

# Mimic the create_selective_checkpoint_contexts function
def get_job_name(config):
    return config.name

# Setup the configuration
config = ServiceConfig("test_job")

# Reproduce the core logic: using functools.partial to create a context/argument function
# Original: context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
job_name_fn = functools.partial(get_job_name, config)

# Define a simple dataset creation function (mimicking f(x, y))
def create_dataset():
    return tf.data.Dataset.range(10)

# Mimic the @torch.compile decorator with @tf.function
@tf.function
def run_distributed_pipeline():
    # Resolve the argument using the partial function
    job_name = job_name_fn()

    # Call the target API: tf.compat.v1.distribute
    # Note: A valid service address is required for the API call, 
    # but we use a placeholder here to test the argument handling and graph construction.
    distribute_fn = tf.compat.v1.distribute(
        processing_mode="parallel_epochs",
        service="grpc://localhost:5000", 
        job_name=job_name
    )

    dataset = create_dataset()
    return distribute_fn(dataset)

# Execute the test case
# We verify that the API accepts the configuration derived from functools.partial
# and that the graph construction (compilation) succeeds.
try:
    distributed_ds = run_distributed_pipeline()
    assert isinstance(distributed_ds, tf.data.Dataset), "Output should be a Dataset"
    print("Test passed: tf.compat.v1.distribute handled functools.partial context correctly.")
except Exception as e:
    print(f"Test failed with error: {e}")