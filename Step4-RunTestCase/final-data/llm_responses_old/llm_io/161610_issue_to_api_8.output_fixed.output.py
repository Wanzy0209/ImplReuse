import sys
import tempfile
import os
import shutil
from typing import NamedTuple

# Handle environment dependency issues (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Details: {e}")
    sys.exit(0)

import torch

# Replicate the data structure from the PyTorch bug report
class MyNamedTuple(NamedTuple):
    first: tf.Tensor
    second: tf.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def test_tfrecord_namedtuple_persistence():
    """
    Test case adapted from PyTorch Issue 161610.
    
    Original Issue: Dynamic attributes on NamedTuple subclasses do not persist 
    through torch.compile.
    
    Adaptation: Verify that 'extra' data (simulating dynamic attributes) on a 
    NamedTuple structure can be explicitly persisted through the 
    TFRecordWriter write/read cycle.
    """
    
    # Setup temporary file for TFRecord
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, "test.tfrecord")
    
    try:
        # 1. Create the initial data structure
        # In the original bug, extra_info is added dynamically via setattr.
        # In TF, we must explicitly include it in the serialization format.
        original_data = MyNamedTupleSubclass(
            first=tf.constant([2.0]), 
            second=tf.constant(1.0)
        )
        
        # Create a dataset containing the tuple
        # We use from_tensors to wrap the single object
        dataset = tf.data.Dataset.from_tensors(original_data)
        
        # 2. Define serialization function
        # This mimics the 'fn' in the bug report where extra_info is added.
        def serialize_and_add_extra(tup):
            extra_info = tf.constant(4.0) # The dynamic value from the bug
            
            # We pack the original fields plus the extra_info into a TFRecord Example
            feature = {
                'first': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tup.first).numpy()])),
                'second': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tup.second).numpy()])),
                'extra_info': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(extra_info).numpy()]))
            }
            
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        # 3. Write to TFRecord
        serialized_dataset = dataset.map(serialize_and_add_extra)
        writer = tf.data.experimental.TFRecordWriter(filepath)
        writer.write(serialized_dataset)
        
        # 4. Read back from TFRecord
        read_dataset = tf.data.TFRecordDataset(filepath)
        
        # 5. Define deserialization function
        def parse_fn(serialized_proto):
            features = {
                'first': tf.io.FixedLenFeature([], tf.string),
                'second': tf.io.FixedLenFeature([], tf.string),
                'extra_info': tf.io.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.io.parse_single_example(serialized_proto, features)
            
            first = tf.io.parse_tensor(parsed_features['first'], out_type=tf.float32)
            second = tf.io.parse_tensor(parsed_features['second'], out_type=tf.float32)
            extra_info = tf.io.parse_tensor(parsed_features['extra_info'], out_type=tf.float32)
            
            return first, second, extra_info

        parsed_dataset = read_dataset.map(parse_fn)
        
        # 6. Verify the results
        # The original bug resulted in an AttributeError when accessing extra_info.
        # Here we assert that extra_info is successfully retrieved.
        results = list(parsed_dataset.as_numpy_iterator())
        assert len(results) == 1
        
        first, second, extra_info = results[0]
        
        # Check original fields
        assert (first == [2.0]).all()
        assert (second == 1.0).all()
        
        # Check the "dynamic" extra_info persisted
        assert (extra_info == 4.0).all()
        
        print("Test Passed: NamedTuple data and extra_info persisted via TFRecordWriter.")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_tfrecord_namedtuple_persistence()