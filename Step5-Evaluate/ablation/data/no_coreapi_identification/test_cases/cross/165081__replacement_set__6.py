import torch
import tensorflow as tf
import torch.nn.functional as F

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> tf.keras.activations.serialize
tf.keras.activations.serialize(tf.keras.activations.linear)

# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.keras.activations.serialize
tf.keras.activations.serialize(tf.keras.activations.linear)

# torch.manual_seed -> torch.isnan
torch.isnan(torch.tensor(52676))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    var_node_6 = arg_0
    var_node_7 = arg_1
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    # torch.float64 -> tf.keras.activations.serialize
    var_node_5 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_6), tf.keras.activations.serialize(var_node_7)])
    
    # torch.full -> torch.argmin
    # torch.float64 -> tf.keras.activations.serialize
    var_node_9 = torch.argmin(arg_0, dim=0)
    
    var_node_10 = arg_2
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_8 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_9), tf.keras.activations.serialize(var_node_10)])
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_4 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_5), tf.keras.activations.serialize(var_node_8)])
    
    var_node_13 = arg_3
    var_node_14 = arg_4
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_12 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_13), tf.keras.activations.serialize(var_node_14)])
    
    var_node_15 = arg_5
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_11 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_12), tf.keras.activations.serialize(var_node_15)])
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_3 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_4), tf.keras.activations.serialize(var_node_11)])
    
    var_node_17 = arg_6
    var_node_18 = arg_7
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_16 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_17), tf.keras.activations.serialize(var_node_18)])
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_2 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_3), tf.keras.activations.serialize(var_node_16)])
    
    # torch.full -> torch.argmin
    var_node_23 = torch.argmin(arg_0, dim=0)
    
    # torch.full -> torch.argmin
    var_node_24 = torch.argmin(arg_0, dim=0)
    
    # torch.matmul -> tf.keras.layers.LSTMCell
    # .to(torch.float64) -> tf.keras.activations.serialize
    var_node_22 = tf.keras.layers.LSTMCell(units=10)([tf.keras.activations.serialize(var_node_23), tf.keras.activations.serialize(var_node_24)])
    
    # torch.full -> torch.argmin
    var_node_26 = torch.argmin(arg_0, dim=0)
    
    return var_node_2

if __name__ == "__main__":
    # Setup dummy arguments to make the test case runnable
    # Note: Using CPU to ensure it runs in all environments, though original used CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy tensors with shapes matching the original comments
    # arg_0: size=(9, 9, 9), dtype=float64
    args = [torch.randn(9, 9, 9, dtype=torch.float64, device=device) for _ in range(19)]
    sentinel = None

    try:
        # Execute the transformed program
        result = fuzzed_program(*args, sentinel)
        print("Test case executed.")
    except Exception as e:
        # Expected to fail due to semantic incompatibility of replacements (e.g., serialize on tensor)
        # but the code structure follows the replacement rules.
        print(f"Test case executed with expected exception: {type(e).__name__}: {e}")