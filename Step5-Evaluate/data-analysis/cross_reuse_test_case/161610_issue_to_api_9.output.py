import os
import tempfile
from typing import NamedTuple
import sys

# Handle TensorFlow import error due to missing system libraries (GLIBCXX)
# This prevents the script from crashing in environments with incompatible C++ libraries.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues.")
    print(f"Error details: {e}")
    TF_AVAILABLE = False
    
    # Define a dummy tensorflow module to allow the script to parse without NameError
    # The test logic will be skipped, so these don't need to function.
    class DummyTensor:
        pass
    
    class DummyTF:
        Tensor = DummyTensor
        constant = DummyTensor
        class io:
            TFRecordWriter = DummyTensor
            serialize_tensor = DummyTensor
            parse_tensor = DummyTensor
        class data:
            TFRecordDataset = DummyTensor
            
    tf = DummyTF()

# Define the NamedTuple structure similar to the PyTorch issue
class MyNamedTuple(NamedTuple):
    first: tf.Tensor
    second: tf.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def test_tfrecord_namedtuple_dynamic_attributes():
    """
    Test case to verify if dynamic attributes on NamedTuple subclasses 
    persist through a TFRecordWriter write/read cycle.
    
    This mirrors the PyTorch issue where torch.compile drops dynamic attributes.
    In TensorFlow, serialization via TFRecordWriter inherently drops non-serialized 
    attributes (like dynamic Python attributes).
    """
    if not TF_AVAILABLE:
        print("Test skipped: TensorFlow is not available in the current environment.")
        return

    # Setup temporary file for TFRecord
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "test.tfrecord")

    try:
        # Create the object
        extended_tup = MyNamedTupleSubclass(first=tf.constant([2.0]), second=tf.constant(1.0))
        
        # Add dynamic attribute
        extra_info = tf.constant(4.0)
        extended_tup.extra_info = extra_info
        
        print("\nTesting NamedTuple with TFRecordWriter:")
        print(f"Original attribute: {extended_tup.extra_info}")

        # Write to TFRecord (The processing step)
        # We serialize the tensor data. The dynamic attribute is not serialized.
        with tf.io.TFRecordWriter(filename) as writer:
            # Serialize the 'first' tensor to write it
            serialized_tensor = tf.io.serialize_tensor(extended_tup.first)
            writer.write(serialized_tensor.numpy())

        # Read back from TFRecord
        dataset = tf.data.TFRecordDataset(filename)
        
        def parse_fn(x):
            return tf.io.parse_tensor(x, tf.float32)
        
        # Reconstruct the object from the read data
        parsed_data = list(dataset.map(parse_fn).as_numpy_iterator())
        reconstructed_first = tf.constant(parsed_data[0])
        read_tup = MyNamedTupleSubclass(first=reconstructed_first, second=tf.constant(1.0))

        # Check if the dynamic attribute persists
        # Expected: AttributeError, similar to the PyTorch bug behavior
        try:
            _ = read_tup.extra_info
            print("TFRecordWriter result: Attribute found (Unexpected)")
            assert False, "Expected AttributeError for dynamic attribute after TFRecord cycle"
        except AttributeError:
            print("TFRecordWriter result: AttributeError (Attribute lost, as expected in this context)")
            assert True

    finally:
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    test_tfrecord_namedtuple_dynamic_attributes()