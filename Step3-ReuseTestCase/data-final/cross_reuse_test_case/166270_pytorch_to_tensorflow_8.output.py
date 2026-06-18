import torch
import tensorflow as tf
import numpy as np

def test_rmsprop_scalar_tensor():
    """
    Adapted test case for tf.keras.optimizers.RMSprop based on the 
    PyTorch torch.squeeze bug (Issue 166270).
    
    The original bug involves an assertion failure `assert len(self.stride) == len(order)` 
    when handling 0-d tensors (scalars) during compilation.
    
    This test verifies if tf.keras.optimizers.RMSprop handles 0-d tensors 
    correctly in both eager and compiled (tf.function) modes, mimicking the 
    conditions that led to the divergence in PyTorch.
    """
    
    # Enable strict checking for potential shape/stride issues
    tf.debugging.set_log_device_placement(True)

    # Mimic the creation of a 0-d tensor (scalar) similar to the result of torch.squeeze
    # In the original bug: var_node_2 = torch.squeeze(var_node_3) -> size=()
    # We create a 0-d variable to be optimized by RMSprop.
    initial_value = 1.0
    var = tf.Variable(initial_value, dtype=tf.float32, trainable=True)

    # Initialize the RMSprop optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

    # Define the training step logic
    def train_step():
        with tf.GradientTape() as tape:
            # Perform an operation on the 0-d tensor
            # This mimics the usage of the squeezed tensor in the original program
            result = var * var
            
            # Handle complex numbers if necessary (mimicking original logic)
            if result.dtype == tf.complex64 or result.dtype == tf.complex128:
                result = tf.math.real(result)
                
            loss = result
        
        grads = tape.gradient(loss, var)
        optimizer.apply_gradients([(grads, var)])
        return loss

    # 1. Run Eager Execution
    print("Running eager execution...")
    try:
        loss_eager = train_step()
        print(f" eager success: loss={loss_eager.numpy()}")
    except Exception as e:
        print(f" eager failed: {e}")
        return

    # Reset variable for compiled run
    var.assign(initial_value)

    # 2. Run Compiled Execution (tf.function)
    # This is equivalent to torch.compile in the original bug report
    print("Running compiled execution (tf.function)...")
    try:
        compiled_step = tf.function(train_step)
        loss_compiled = compiled_step()
        print(f" compile success: loss={loss_compiled.numpy()}")
    except Exception as e:
        print(f" compile failed: {e}")
        # In the context of the original bug, this is where the crash occurred
        return

    # 3. Verify Divergence
    # Reset variable again to compare results fairly
    var.assign(initial_value)
    loss_eager_check = train_step()
    
    var.assign(initial_value)
    loss_compiled_check = compiled_step()

    if np.allclose(loss_eager_check.numpy(), loss_compiled_check.numpy()):
        print(" No divergence detected between eager and compiled modes.")
    else:
        print(f" Divergence detected: Eager={loss_eager_check.numpy()}, Compiled={loss_compiled_check.numpy()}")

if __name__ == "__main__":
    test_rmsprop_scalar_tensor()