import torch
import tensorflow as tf

# Check for GPU availability to ensure memory tracking is relevant
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to see allocation changes clearly
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Define Custom Gradient Function (Analogous to MyOp in PyTorch)
@tf.custom_gradient
def my_op(inp):
    # Create large tensors similar to the PyTorch implementation
    out_0 = tf.zeros(2**20, dtype=tf.float32)
    out_1 = tf.zeros(2**20, dtype=tf.float32)

    # Define the gradient function
    def grad(dA, dB):
        # Access saved tensors (captured in closure)
        # This mimics _ = ctx.saved_tensors in PyTorch
        _ = inp, out_0, out_1
        return None # Gradient w.r.t inp is None

    return (out_0, out_1), grad

# 2. Setup Variables
# dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
dummy_input = tf.Variable(tf.random.normal([2**20], dtype=tf.float32))

# 3. Define Loop Condition and Body for tf.keras.ops.while_loop
# We structure the loop to run the operation once per step, 
# analogous to the single execution of the checkpointed function in PyTorch.
def cond(i, res):
    return i < 1

def body(i, res):
    # Call the custom op
    # PyTorch: return MyOp.apply(inp)[0]
    o0, o1 = my_op(dummy_input)
    return i + 1, o0

# 4. Run the test loop
print("Starting test loop...")
for i in range(1000):
    # PyTorch: full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
    # TF: Run the while loop
    with tf.GradientTape() as tape:
        # We execute the while_loop inside the tape to record operations for autograd
        _, full_out = tf.keras.ops.while_loop(
            cond,
            body,
            loop_vars=[tf.constant(0), tf.zeros(2**20, dtype=tf.float32)]
        )
        
        # PyTorch: full_out.sum().backward()
        loss = tf.reduce_sum(full_out)
    
    # Compute gradients
    grads = tape.gradient(loss, dummy_input)
    
    # PyTorch: dummy_input.grad = None
    # In TensorFlow, we don't explicitly nullify gradients on the variable, 
    # but we ensure we don't hold references to 'grads' to allow GC.
    
    # Print memory usage
    if i % 10 == 0 or i == 0:
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Iteration {i}: Allocated {mem_info['current'] / 1024**2:.2f} MiB")
        except Exception as e:
            # Fallback if GPU memory info isn't available (e.g. running on CPU)
            print(f"Iteration {i}: (Memory info unavailable: {e})")