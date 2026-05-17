import torch
import tensorflow as tf
import numpy as np

def computation(x_shard):
    """
    Replicates the logic of the PyTorch function 'foo'.
    Original PyTorch logic:
        c = torch.tensor(7, dtype=torch.uint8)
        return c+x, torch.neg(c), torch.neg(c)+x
    """
    # Create a uint8 tensor constant
    c = tf.constant(7, dtype=tf.uint8)
    
    # Perform negation
    # Note: In PyTorch, neg(uint8) wraps around (e.g., -7 becomes 249).
    # In TensorFlow, negative(uint8) typically promotes to int32 (e.g., -7 becomes -7).
    neg_c = tf.negative(c)
    
    # Return the three results corresponding to the original test case
    return c + x_shard, neg_c, neg_c + x_shard

def main():
    # Setup input data matching the PyTorch seed (torch.manual_seed(0))
    # PyTorch output: tensor([[ 1.5410, -0.2934], [-2.1788,  0.5684]])
    x_np = np.array([[ 1.5410, -0.2934], [-2.1788,  0.5684]], dtype=np.float32)
    x = tf.constant(x_np)

    print(f"input: {x_np}")

    # tf.compat.v1.tpu.batch_parallel requires a TPU environment.
    # We include the standard boilerplate to initialize the TPU system.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        # Execute the computation using batch_parallel
        # inputs must be a list of lists of tensors.
        # num_shards=1 is used to match the single-device execution context of the original bug report.
        outputs = tf.compat.v1.tpu.batch_parallel(
            computation, 
            inputs=[[x]], 
            num_shards=1
        )

        # Run the graph to get results
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            res = sess.run(outputs)

            print(f"res[0]: {res[0]}")
            print(f"res[1]: {res[1]}")
            print(f"res[2]: {res[2]}")

            # Verification logic:
            # In PyTorch (Eager): neg(uint8) wraps to 249. res[2] is 249 + x.
            # In PyTorch (Inductor Bug): neg(uint8) treated as -7. res[2] is -7 + x.
            # In TensorFlow: negative(uint8) promotes to int32 (-7). res[2] is -7 + x.
            # Therefore, TF's result for res[2] will match PyTorch's *buggy* compiled output,
            # because TF promotes types rather than wrapping for negation.
            
            expected_res_0 = np.array([[8.5410, 6.7066], [4.8212, 7.5684]], dtype=np.float32)
            expected_res_1 = -7 # int32
            expected_res_2 = np.array([[-5.4590, -7.2934], [-9.1788, -6.4316]], dtype=np.float32)

            assert np.allclose(res[0], expected_res_0), f"res[0] mismatch: {res[0]} != {expected_res_0}"
            assert res[1] == expected_res_1, f"res[1] mismatch: {res[1]} != {expected_res_1}"
            assert np.allclose(res[2], expected_res_2), f"res[2] mismatch: {res[2]} != {expected_res_2}"
            
            print("Test passed: TensorFlow behavior matches expected type promotion logic.")

    except (tf.errors.NotFoundError, ValueError) as e:
        print(f"TPU hardware not found or initialization failed: {e}")
        print("This test case requires a TPU runtime to execute tf.compat.v1.tpu.batch_parallel.")

if __name__ == "__main__":
    main()