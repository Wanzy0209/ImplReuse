import sys

# Attempt to import required libraries. 
# If the environment is missing required system libraries (like GLIBCXX_3.4.29),
# catch the ImportError and exit gracefully instead of crashing.
try:
    import torch
    import tensorflow as tf
    import tensorflow.experimental.dtensor as dtensor
except ImportError as e:
    print(f"Skipping test: Failed to import dependencies due to environment issues.")
    print(f"Details: {e}")
    sys.exit(0)

def get_sample_tensors():
    """
    Simulates the output of a VAE model which returns a tuple of tensors:
    (reconstruction, mu, log_var).
    """
    batch_size = 32
    reconstruction = tf.random.normal((batch_size, 784))
    mu = tf.random.normal((batch_size, 20))
    log_var = tf.random.normal((batch_size, 20))
    return (reconstruction, mu, log_var)

def main():
    # Setup for DTensor: Create a mesh and a layout
    # Using CPU devices to ensure the test runs in most environments
    devices = ["CPU:0", "CPU:1"]
    try:
        mesh = dtensor.create_mesh([("batch", len(devices))], devices=devices)
    except Exception as e:
        print(f"Skipping test: Mesh creation failed. {e}")
        return

    # Define a replicated layout
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

    # Get the tuple of tensors (simulating the eager mode model output)
    eager_output = get_sample_tensors()
    print(f"Eager output type: {type(eager_output)}")
    
    # Verify eager mode behavior (accessing shape on individual elements works)
    print(f"Eager element shapes: {[t.shape for t in eager_output]}")

    # The Bug Reproduction Logic:
    # In the PyTorch bug, torch.compile(model)(*inputs) returned a tuple, 
    # but the user code attempted to access .shape on the tuple itself.
    # Here, we test if copy_to_mesh preserves the tuple structure when passed a tuple,
    # and if accessing .shape on the result triggers an AttributeError.
    
    print("\nTesting similar API: tf.experimental.dtensor.copy_to_mesh")
    try:
        # Pass the tuple of tensors to copy_to_mesh
        # We check if the API maps over the tuple or treats it as a single object
        compiled_output = dtensor.copy_to_mesh(eager_output, layout)
        
        print(f"API Output type: {type(compiled_output)}")

        # Attempt to access .shape on the returned object
        # If the API returns a tuple (like torch.compile did in the bug), this will fail
        print(f"Attempting to access .shape on API output...")
        shape = compiled_output.shape
        print(f"Output shape: {shape}")

    except AttributeError as e:
        print(f"Bug Reproduced! AttributeError: {e}")
        print("The API returned a tuple, but .shape was accessed on the tuple object.")
    except TypeError as e:
        print(f"API TypeError: {e}")
        print("The API might not support tuple inputs directly.")
    except Exception as e:
        print(f"Unexpected Exception: {e}")

if __name__ == "__main__":
    main()