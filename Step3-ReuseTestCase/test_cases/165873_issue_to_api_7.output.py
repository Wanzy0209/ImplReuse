import tensorflow as tf
import tempfile
import os

# Disable eager execution to use SessionCreator and Saver (TF 1.x behavior)
tf.compat.v1.disable_eager_execution()

def test_session_creator_shape_mismatch():
    """
    Test case reflecting the PyTorch issue where loading a 1D tensor into a scalar
    parameter might not raise an error. This test uses tf.compat.v1.train.SessionCreator
    to verify behavior in TensorFlow.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.ckpt")

        # 1. Create a checkpoint with a 1D tensor (analogous to 'large_tensor' in the issue)
        with tf.compat.v1.Graph().as_default():
            # Create a 1D variable with 5 elements
            v_1d = tf.compat.v1.get_variable("threshold", shape=[5], initializer=tf.compat.v1.ones_initializer())
            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                saver.save(sess, ckpt_path)

        # 2. Define a graph with a scalar variable (analogous to 'SimpleModule' with scalar threshold)
        with tf.compat.v1.Graph().as_default():
            # Initialize a scalar parameter (shape [])
            v_scalar = tf.compat.v1.get_variable("threshold", shape=[], initializer=tf.compat.v1.zeros_initializer())
            saver = tf.compat.v1.train.Saver()

            # 3. Use SessionCreator to load the state
            # ChiefSessionCreator is a concrete implementation of SessionCreator
            # This corresponds to module.load_state_dict(state_dict)
            session_creator = tf.compat.v1.train.ChiefSessionCreator(
                saver=saver,
                checkpoint_filename_with_path=ckpt_path
            )

            # 4. Execute and verify behavior
            # In the PyTorch bug, this loads the first element silently without error.
            # In TensorFlow, we expect an InvalidArgumentError due to shape mismatch.
            error_raised = False
            try:
                sess = session_creator.create_session()
                # If we reach here, TF mimics the PyTorch bug behavior
                val = sess.run(v_scalar)
                # If the bug existed in TF, val would be 1.0 (first element of saved 1D tensor)
                print(f"Value loaded: {val}")
            except tf.errors.InvalidArgumentError as e:
                error_raised = True
                # Verify the error is related to shape or assignment
                assert "shape" in str(e).lower() or "assign" in str(e).lower()
                print(f"Correctly raised error: {e}")

            # Assert that an error was raised, contrasting with the PyTorch bug report
            assert error_raised, "Expected an error for shape mismatch (scalar vs 1D), but load succeeded."

if __name__ == "__main__":
    test_session_creator_shape_mismatch()
    print("Test passed.")