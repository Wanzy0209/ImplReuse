import torch
import torch.distributed as dist
import torch.nn.functional as F
import tensorflow as tf
import os

def main():
    # Setup for distributed environment to allow torch.distributed.send_object_list
    # Note: In a real scenario, this requires multiple processes. 
    # For a single runnable script, we initialize with world_size=1 if possible, 
    # or handle the lack of initialization gracefully.
    if not dist.is_initialized():
        try:
            # Using a simple TCP init for demonstration purposes
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            # This init might fail if not run with torchrun, but we include it for API correctness
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
        except Exception as e:
            print(f"Distributed init skipped (expected if not running with torchrun): {e}")

    # --- Data Generation ---
    # Replacement: torch.rand -> tf.nn.collapse_repeated
    # Original: x = torch.rand(2, device="cuda")
    # We create a constant tensor and collapse it to simulate the data generation step
    # using the requested API.
    
    # Generate x
    raw_x = tf.constant([[1.0, 1.0], [2.0, 2.0]], dtype=tf.float32)
    x_tf = tf.nn.collapse_repeated(raw_x, [2, 2]) # Shape (2, 2)
    x = torch.from_numpy(x_tf.numpy()).cuda()

    # Generate a
    raw_a = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=tf.float32)
    a_tf = tf.nn.collapse_repeated(raw_a, [3, 3]) # Shape (2, 3)
    a = torch.from_numpy(a_tf.numpy()).cuda()

    # Generate b
    raw_b = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=tf.float32)
    b_tf = tf.nn.collapse_repeated(raw_b, [2, 2, 2]) # Shape (3, 2)
    b = torch.from_numpy(b_tf.numpy()).cuda()

    # --- Function Definition ---
    # Original: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.softsign
    f = lambda x, a, b: F.softsign(torch.addmm(x, a, b, alpha=0.5, beta=0.5))

    # --- "Compilation" Step ---
    # Original: fc = torch.compile(f)
    # Replacement: torch.compile -> torch.distributed.send_object_list
    # We send the function object to simulate the preparation/compilation step.
    if dist.is_initialized():
        try:
            # Sending to rank 0 (self) to avoid hanging in a single-process context
            dist.send_object_list([f], dst=0)
        except Exception as e:
            print(f"Send failed (might require peer): {e}")
    
    # We assign fc to f to maintain the logic flow of the original test case
    fc = f

    # --- Execution ---
    # Run eager version
    res_eager = f(x, a, b)
    
    # Run "compiled" version
    res_compiled = fc(x, a, b)

    # --- Verification using Similar API ---
    # Similar API: tf.keras.metrics.sparse_categorical_accuracy
    # We use this to compare the results of the eager and compiled runs.
    # sparse_categorical_accuracy expects y_true (int labels) and y_pred (logits).
    
    # Convert results to numpy for TensorFlow metric
    res_eager_np = res_eager.cpu().numpy()
    res_compiled_np = res_compiled.cpu().numpy()

    # Use argmax of eager result as "ground truth" labels
    y_true = res_eager_np.argmax(axis=1)
    y_pred = res_compiled_np

    # Calculate accuracy
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    
    print(f"Eager Result:\n{res_eager}")
    print(f"Compiled Result:\n{res_compiled}")
    print(f"Sparse Categorical Accuracy (Eager vs Compiled): {accuracy.numpy()}")

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()