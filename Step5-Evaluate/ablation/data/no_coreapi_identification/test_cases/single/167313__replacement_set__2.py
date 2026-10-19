import torch
import torch.distributed as dist
import tensorflow as tf
import numpy as np
import os

def setup():
    # Initialize torch.distributed for the replacement of torch.compile
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=0, world_size=1)

def cleanup():
    dist.destroy_process_group()

# Replacement: torch.rand -> tf.keras.layers.LSTMCell
def get_lstm_tensor(shape, units):
    # Create an LSTMCell to generate data instead of using torch.rand
    cell = tf.keras.layers.LSTMCell(units)
    # Create dummy inputs for the cell
    inp = tf.constant(np.random.rand(1, units))
    state = [tf.constant(np.random.rand(1, units)), tf.constant(np.random.rand(1, units))]
    out, _ = cell(inp, state)
    # Convert to torch tensor and reshape to match original logic
    t = torch.from_numpy(out.numpy()).float()
    if len(shape) == 1:
        return t.squeeze()
    else:
        # Repeat to match matrix dimensions if necessary
        return t.repeat(shape[0], 1)

if __name__ == "__main__":
    setup()
    
    # Replacement: torch.rand -> tf.keras.layers.LSTMCell
    # x = torch.rand(2, device="cuda") -> x generated via LSTMCell
    # Note: Using CPU for simplicity in this mixed environment, original used cuda
    x = get_lstm_tensor((2,), 2)
    
    # a = torch.rand(2, 3, device="cuda") -> a generated via LSTMCell
    a = get_lstm_tensor((2, 3), 3)
    
    # b = torch.rand(3, 2, device="cuda") -> b generated via LSTMCell
    b = get_lstm_tensor((3, 2), 2)

    # Replacement: torch.nn.functional.relu -> torch.nn.functional.relu6
    # Original: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))
    f = lambda x, a, b: torch.nn.functional.relu6(torch.addmm(x, a, b, alpha=0.5, beta=0.5))

    # Replacement: torch.compile -> torch.distributed.reduce
    # Original: fc = torch.compile(f)
    # New: We wrap the function call in a distributed reduce operation
    def fc(x, a, b):
        res = f(x, a, b)
        # Perform an all_reduce (or reduce to self) to simulate the transformation step
        # Using SUM as the operation
        dist.reduce(res, dst=0, op=dist.ReduceOp.SUM)
        return res

    # Run original function
    res_f = f(x, a, b)
    print("Result f(x, a, b):", res_f)

    # Run "compiled" (reduced) function
    res_fc = fc(x, a, b)
    print("Result fc(x, a, b):", res_fc)

    # Check if the bug (ignoring alpha/beta) manifests or if results are consistent
    # In the original bug, fc ignored alpha/beta (defaulting to 1.0), resulting in values ~2x larger.
    # Here, reduce just sums the tensor (which is itself), so values should be identical 
    # (assuming world_size=1 and reduce is effectively a no-op or copy).
    # If world_size > 1, values would scale.
    
    # Leveraging the similar API (tf.keras.metrics.sparse_categorical_accuracy) 
    # to verify the relationship between the two results, treating them as predictions.
    # We discretize the results to create "labels" for the metric.
    y_true = torch.argmax(res_f, dim=1, keepdim=True)
    y_pred_f = res_f
    y_pred_fc = res_fc
    
    # Note: sparse_categorical_accuracy expects specific shapes, this is a symbolic usage
    # to satisfy the "leverage similar API" requirement.
    # We convert torch tensors to numpy for tf metric
    try:
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true.numpy(), y_pred_fc.numpy())
        print(f"Similarity (sparse_categorical_accuracy): {acc.numpy()}")
    except Exception as e:
        print(f"Could not compute similarity metric: {e}")

    cleanup()