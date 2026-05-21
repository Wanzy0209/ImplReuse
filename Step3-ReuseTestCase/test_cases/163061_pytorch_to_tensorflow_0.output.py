import torch
import tensorflow as tf

def tf_add(x: tf.Tensor, y: tf.Tensor):
    return x + y

def tpu_rewrite_add(x: tf.Tensor, y: tf.Tensor):
    """
    Adapts the logic of torch.compile using the similar API 
    tf.compat.v1.tpu.rewrite.
    """
    # tf.compat.v1.tpu.rewrite takes the computation function and a list of inputs
    return tf.compat.v1.tpu.rewrite(tf_add, [x, y])

def main():
    # Initialize TPU system if available, otherwise fallback to CPU/GPU
    # Note: To accurately observe GIL behavior similar to the CUDA context in PyTorch,
    # this should ideally run on TPU hardware.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Running on TPU")
    except ValueError:
        strategy = tf.distribute.get_strategy()
        print("TPU not found, falling back to default strategy")

    with strategy.scope():
        # Create tensors similar to torch.randn(4096, 4096, device='cuda')
        x = tf.random.normal((4096, 4096))
        y = tf.random.normal((4096, 4096))

        for _ in range(10):
            # Standard execution
            _ = tf_add(x, y)
            
            # Rewritten/Compiled execution
            # The rewrite function returns a list of tensors corresponding to the output
            result = tpu_rewrite_add(x, y)
            
            # Access the result to trigger execution
            _ = result[0]

if __name__ == "__main__":
    main()