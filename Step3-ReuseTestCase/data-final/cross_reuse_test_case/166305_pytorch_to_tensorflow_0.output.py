import torch
import tensorflow as tf
import numpy as np

# Use TF 1.x compatibility mode as required by the target API
tf.compat.v1.disable_eager_execution()

# --- Custom Autograd Function ---
# Equivalent to PyTorch's SimplistDoubleFn(torch.autograd.Function)
@tf.custom_gradient
def SimplistDoubleFn(x):
    def grad(dy):
        return dy * 2
    return x * 2, grad

# --- Model Definition ---
# Equivalent to PyTorch's DoubleLayer(nn.Module)
def double_layer(x):
    return SimplistDoubleFn(x)

# --- Computation Function ---
# This function represents the logic that will be compiled for TPU.
# Equivalent to the forward pass and loss calculation in the PyTorch loop.
def computation_fn(inputs):
    x = inputs[0]
    # Forward pass
    out = double_layer(x)
    # Loss calculation (MSE)
    loss = tf.reduce_mean(tf.square(out - x))
    return loss

def main():
    # --- Distributed/TPU Initialization ---
    # Equivalent to dist.init_process_group and DDP setup.
    # tf.compat.v1.tpu.rewrite is designed for TPU execution.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.tpu.experimental.initialize_tpu_system(resolver)
    except (ValueError, tf.errors.NotFoundError) as e:
        print(f"TPU initialization failed (expected if not on TPU): {e}")
        print("This test case requires a TPU environment to execute tf.compat.v1.tpu.rewrite.")
        return

    # --- Input Definition ---
    # Equivalent to torch.rand(...)
    x = tf.compat.v1.placeholder(tf.float32, shape=[2, 3, 256, 256])

    # --- Compilation ---
    # Equivalent to torch.compile(model) + DDP wrapping.
    # tf.compat.v1.tpu.rewrite compiles the 'computation_fn' for execution on the TPU.
    # This is the semantic equivalent of the PyTorch compile + DDP optimization path.
    compiled_op = tf.compat.v1.tpu.rewrite(computation_fn, [x])

    # --- Execution Loop ---
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        for it in range(3):
            # Generate random input
            input_data = np.random.rand(2, 3, 256, 256).astype(np.float32)
            
            # Run the compiled operation
            # In PyTorch: out = model(x); loss = ...; loss.backward()
            # In TF TPU: The computation_fn includes the loss, and rewrite handles the graph.
            # We fetch the result of the computation (the loss).
            loss_val = sess.run(compiled_op, feed_dict={x: input_data})
            
            print(f"iter={it+1} loss={loss_val:.6f}")

if __name__ == "__main__":
    main()