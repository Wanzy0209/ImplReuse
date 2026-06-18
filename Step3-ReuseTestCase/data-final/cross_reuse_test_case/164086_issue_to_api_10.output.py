import torch
import tensorflow as tf
import numpy as np
import tempfile
import shutil
import sys

def test_tf_saved_model_mixed_precision_divergence():
    """
    Test case derived from Issue 164086 (PyTorch) adapted for tf.saved_model.save.
    
    The original issue involves an IncompatibleTypeErrorImpl related to pointer<fp16> 
    and triton.language.float64 during eager/compile divergence with complex math ops.
    
    This test adapts the logic to TensorFlow, wrapping the mixed-precision operations
    in a tf.Module and saving it via tf.saved_model.save to verify that the graph
    serialization and execution handle the type precisions correctly without divergence.
    """
    
    # Define a Module that mimics the logic of the PyTorch function 'foo'
    class MixedPrecisionOpsModel(tf.Module):
        def __init__(self):
            super().__init__()
            # Initialize weights corresponding to arg2 in the original issue
            # size=(46, 128), dtype=float16
            self.w_linear = tf.Variable(
                tf.random.normal([46, 128], dtype=tf.float16), name='w_linear'
            )

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[42, 56], dtype=tf.int64, name='arg0'),
            tf.TensorSpec(shape=[50000, 128], dtype=tf.float16, name='arg1'),
            tf.TensorSpec(shape=[50000, 4, 46], dtype=tf.float16, name='arg3'),
            tf.TensorSpec(shape=[25786, 46], dtype=tf.float16, name='arg4'),
            tf.TensorSpec(shape=[24214, 46], dtype=tf.float16, name='arg5'),
        ])
        def call(self, arg0, arg1, arg3, arg4, arg5):
            # t0 = arg0 (int64)
            # PyTorch allows tanh on int64 (returns float). TF requires float input.
            # We cast to float16 to align with the precision theme of the bug.
            t1 = tf.tanh(tf.cast(arg0, tf.float16))
            
            # t2 = t1.clone(); t2.zero_()
            t2 = tf.zeros_like(t1)
            
            # t3 = arg1
            t3 = arg1
            
            # t4 = self.w_linear
            t4 = self.w_linear
            
            # t5 = torch.nn.functional.linear(t3, t4)
            # In TF: matmul(t3, t4^T)
            t5 = tf.linalg.matmul(t3, t4, transpose_b=True)
            
            # t6 = arg3
            t6 = arg3
            
            # t7 = t6.max(dim=1).values
            t7 = tf.reduce_max(t6, axis=1)
            
            # t8 = arg4, t9 = arg5
            t8 = arg4
            t9 = arg5
            
            # t10 = torch.cat([t8, t9], dim=0)
            t10 = tf.concat([t8, t9], axis=0)
            
            # t11 = torch.pow(torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7)
            # Chained pow operations on float16 tensors
            t11 = tf.math.pow(tf.math.pow(tf.math.pow(tf.math.pow(t5, t7), t10), t5), t7)
            
            # t12 = torch.nn.functional.embedding(torch.clamp(t2, 0, t11.size(0) - 1).to(torch.long), t11)
            # PyTorch: embedding(indices, weights)
            # TF: embedding_lookup(params, ids)
            # t2 is float16, needs to be int32 for indices.
            # Clamp logic: 0 to 49999 (t11.size(0) is 50000)
            indices = tf.clip_by_value(tf.cast(t2, tf.int32), 0, 49999)
            t12 = tf.nn.embedding_lookup(t11, indices)
            
            return t12

    # 1. Setup inputs matching the shapes and dtypes from the bug report
    arg0 = tf.constant(np.random.randint(0, 1000, [42, 56]), dtype=tf.int64)
    arg1 = tf.random.uniform([50000, 128], dtype=tf.float16)
    arg3 = tf.random.uniform([50000, 4, 46], dtype=tf.float16)
    arg4 = tf.random.uniform([25786, 46], dtype=tf.float16)
    arg5 = tf.random.uniform([24214, 46], dtype=tf.float16)

    model = MixedPrecisionOpsModel()

    # 2. Run Eager Mode (Baseline)
    print("Running Eager Mode...")
    try:
        out_eager = model.call(arg0, arg1, arg3, arg4, arg5)
        print("Eager Mode Success! ")
    except Exception as e:
        print(f"Eager Mode Failed: {e}")
        return

    # 3. Save the model using the Similar API: tf.saved_model.save
    # This serializes the tf.function graph.
    export_dir = tempfile.mkdtemp()
    print(f"Saving model to {export_dir}...")
    try:
        tf.saved_model.save(model, export_dir)
        print("Model Saved Successfully! ")
    except Exception as e:
        print(f"Save Failed: {e}")
        shutil.rmtree(export_dir)
        return

    # 4. Load the model (Simulating usage of the saved artifact)
    print("Loading model...")
    try:
        loaded_model = tf.saved_model.load(export_dir)
        print("Model Loaded Successfully! ")
    except Exception as e:
        print(f"Load Failed: {e}")
        shutil.rmtree(export_dir)
        return

    # 5. Run Compiled/Traced Mode (from SavedModel)
    # This tests if the saved graph handles the mixed precision (fp16) and type casting
    # correctly, analogous to the torch.compile divergence check.
    print("Running Loaded Model (Traced)...")
    try:
        out_loaded = loaded_model.call(arg0, arg1, arg3, arg4, arg5)
        print("Loaded Model Execution Success! ")
    except Exception as e:
        print(f"Loaded Model Execution Failed: {e}")
        shutil.rmtree(export_dir)
        return

    # 6. Assertions
    # Check for shape and dtype consistency
    assert out_eager.shape == out_loaded.shape, \
        f"Shape divergence: Eager {out_eager.shape} vs Loaded {out_loaded.shape}"
    assert out_eager.dtype == out_loaded.dtype, \
        f"Dtype divergence: Eager {out_eager.dtype} vs Loaded {out_loaded.dtype}"
    
    # Check numerical consistency (allowing for minor floating point differences)
    np.testing.assert_allclose(out_eager.numpy(), out_loaded.numpy(), rtol=1e-5, atol=1e-5)
    
    print("Test Passed: No divergence between Eager and SavedModel execution.")

if __name__ == '__main__':
    test_tf_saved_model_mixed_precision_divergence()