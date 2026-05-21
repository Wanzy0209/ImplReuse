import torch
import tensorflow as tf
import numpy as np

def test_convert_to_tensor_params():
    """
    Adapted test case for tf.convert_to_tensor based on the torch.addmm bug report.
    
    Original Bug: torch.compile (Inductor) ignored alpha/beta parameters in addmm 
                  when optimizing the graph.
    
    Adaptation: We verify if tf.convert_to_tensor respects its 'dtype' parameter
                when run inside tf.function (TensorFlow's compilation mode),
                analogous to checking if alpha/beta were respected in torch.compile.
    """
    
    # Setup input data (float64) to force a dtype conversion
    x = np.random.rand(2, 3).astype(np.float64)

    # Define the function using the target API.
    # We include a point-wise operation (relu) similar to the original test case
    # to ensure the graph structure isn't trivially optimized away in a way
    # that might hide parameter handling issues.
    f = lambda x: tf.nn.relu(tf.convert_to_tensor(x, dtype=tf.float32))

    # Compile the function (analogous to torch.compile)
    fc = tf.function(f)

    # Execute in eager mode
    res_eager = f(x)
    print(f"Eager Result Dtype: {res_eager.dtype}")

    # Execute in compiled mode
    res_compiled = fc(x)
    print(f"Compiled Result Dtype: {res_compiled.dtype}")

    # Assertions
    # In the original bug, compiled mode produced different values (ignored params).
    # Here we check if the dtype parameter is respected in both modes.
    assert res_eager.dtype == tf.float32, "Eager mode failed to respect dtype parameter"
    assert res_compiled.dtype == tf.float32, "Compiled mode failed to respect dtype parameter (Potential Bug)"

if __name__ == "__main__":
    test_convert_to_tensor_params()