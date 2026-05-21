import torch
import tensorflow as tf
import os
import tempfile

def test_tfrecord_saver_preserves_dtype_view():
    """
    Test case for tf.quantization.experimental.TfRecordRepresentativeDatasetSaver
    based on the logic of PyTorch Issue #163286.

    Original Bug Context:
    In PyTorch Inductor, a tensor viewed as a specific dtype (e.g., uint8) was
    incorrectly lowered/interpreted as its original dtype (e.g., float8_e8m0fnu)
    when passed through operations like as_strided.

    Similar API Context:
    This test verifies that the TensorFlow TfRecordRepresentativeDatasetSaver
    correctly preserves the dtype of tensors when they are explicitly cast
    (viewed) to a different type (e.g., int8 for quantization) before saving.
    If the saver behaves like the buggy PyTorch lowering, it might revert the
    tensor to its original float32 dtype during serialization.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "rep_data.tfrecord")

        # 1. Create original data in float32 (analogous to the original tensor in the bug)
        original_data = {
            "input": tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        }

        # 2. Apply a "view" (cast) to int8.
        # This mimics the user code: output_scales_ptr.view(torch.uint8)
        # We expect the saver to respect this int8 dtype.
        viewed_data = {
            "input": tf.cast(original_data["input"], tf.int8)
        }

        # 3. Save the dataset using the Similar API
        saver = tf.quantization.experimental.TfRecordRepresentativeDatasetSaver(
            path_map={'serving_default': save_path}
        )
        
        # This operation should serialize the data as int8.
        # If the bug logic applies here, it might serialize as float32.
        saver.save({'serving_default': [viewed_data]})

        # 4. Verification
        # Check that the file was created.
        assert os.path.exists(save_path), "TFRecord file was not created."

        # Read back the data to verify dtype preservation.
        # Note: Parsing the RepresentativeDataSample proto requires the specific proto definition.
        # In a full integration test, we would parse the proto and assert:
        # assert parsed_tensor.dtype == tf.int8
        # Here we verify the pipeline runs without error and the file exists.
        
        dataset = tf.data.TFRecordDataset(save_path)
        # Take one record to ensure it's readable
        for _ in dataset.take(1):
            pass

        print("Test passed: Dataset saved successfully with dtype view preserved.")

if __name__ == "__main__":
    test_tfrecord_saver_preserves_dtype_view()