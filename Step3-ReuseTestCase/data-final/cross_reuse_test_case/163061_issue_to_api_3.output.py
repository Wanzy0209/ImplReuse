import torch
import tensorflow as tf
import threading
import time

# Ensure we are using TensorFlow 2.x behavior (Eager Execution)
# The issue relates to GIL release during execution, which is relevant in Eager mode.
if tf.__version__.startswith('1.'):
    tf.compat.v1.enable_eager_execution()

def manual_update(var: tf.Variable, grad: tf.Tensor, learning_rate: float = 0.01):
    """
    Mimics torch_add: A standard, high-level operation.
    """
    return var.assign_sub(learning_rate * grad)

def proximal_optimizer_update(var: tf.Variable, grad: tf.Tensor, learning_rate: float = 0.01):
    """
    Mimics torch_compile_add: Uses the specific API similar to the issue.
    API: tf.compat.v1.train.ProximalGradientDescentOptimizer
    """
    opt = tf.compat.v1.train.ProximalGradientDescentOptimizer(
        learning_rate=learning_rate,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0
    )
    # apply_gradients returns an Operation in graph mode, or executes in-place in eager mode
    return opt.apply_gradients([(grad, var)])

def raw_kernel_update(var: tf.Variable, grad: tf.Tensor, learning_rate: float = 0.01):
    """
    Mimics triton_add: A lower-level operation closer to the kernel implementation.
    """
    # Using a raw op to simulate a custom kernel execution
    return tf.raw_ops.ResourceApplyGradientDescent(
        var=var.handle,
        alpha=learning_rate,
        delta=grad,
        use_locking=False
    )

def check_gil_release(func, name, var, grad):
    """
    Helper to check if the GIL is released during the operation execution.
    If GIL is held, the background thread will be blocked and count will be low.
    If GIL is released, the background thread can run concurrently.
    """
    print(f"Running {name}...")
    active_flag = [True]
    bg_count = [0]

    def background_thread():
        while active_flag[0]:
            bg_count[0] += 1
            time.sleep(0.0001)

    t = threading.Thread(target=background_thread)
    t.start()

    # Execute the target function multiple times
    for _ in range(10):
        func(var, grad)

    active_flag[0] = False
    t.join()
    
    print(f"Finished {name}. Background thread iterations: {bg_count[0]}")
    # Note: A high count suggests GIL was released. A low count suggests GIL was held.

def main():
    # Setup variables
    var = tf.Variable([1.0, 2.0, 3.0])
    grad = tf.constant([0.1, 0.1, 0.1])

    # 1. Test Standard Operation
    check_gil_release(manual_update, "Manual Update (Standard)", var, grad)

    # 2. Test Similar API (ProximalGradientDescentOptimizer)
    # This corresponds to the 'torch.compile' part of the original issue
    check_gil_release(proximal_optimizer_update, "ProximalGradientDescentOptimizer (Similar API)", var, grad)

    # 3. Test Raw/Low-level Operation
    check_gil_release(raw_kernel_update, "Raw Kernel Op (Custom)", var, grad)

if __name__ == "__main__":
    main()