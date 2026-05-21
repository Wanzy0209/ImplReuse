import torch
import tensorflow as tf
import tempfile
import shutil
import unittest

class TestSnapshotConstraintViolation(unittest.TestCase):
    def test_dynamic_sequence_snapshot_constraint_violation(self):
        """
        Adapts the PyTorch torch.export.export constraint violation test to 
        tf.data.experimental.snapshot.
        
        Original Bug: torch.export.export fails with 'Constraints violated (seq)' 
        when the sequence length of input data exceeds the guard constraint (e.g., 512).
        
        Adaptation: We use tf.data.experimental.snapshot to persist a dataset of 
        sequences. We define a reader_func that enforces a max sequence length constraint.
        We then verify that reading a snapshot containing a sequence that violates 
        this constraint raises a descriptive error, mimicking the torch._dynamo behavior.
        """
        snapshot_path = tempfile.mkdtemp()
        
        try:
            # 1. Create a dataset with sequences of varying lengths
            # Sequence 1: length 2 (Valid)
            # Sequence 2: length 3 (Violates constraint of <= 2)
            # Using RaggedTensor to represent variable length sequences naturally
            data = tf.ragged.constant([[1, 2], [1, 2, 3]])
            ds = tf.data.Dataset.from_tensor_slices(data)
            
            # 2. Define a reader_func that acts as the "Guard" for sequence length
            def constrained_reader_func(datasets):
                for dataset in datasets:
                    for seq in dataset:
                        seq_len = tf.shape(seq)[0]
                        # Constraint: seq <= 2
                        if seq_len > 2:
                            raise ValueError(
                                f"Constraints violated (seq)! "
                                f"Not all values of seq = {seq_len.numpy()} in the specified range seq <= 2 "
                                f"satisfy the generated guard."
                            )
                        yield seq
            
            # 3. Write the snapshot (The "Export" phase)
            ds_write = ds.snapshot(snapshot_path)
            # Consume to ensure files are written
            list(ds_write.as_numpy_iterator())
            
            # 4. Read the snapshot with the constrained reader (The "Run" phase)
            # This simulates the scenario where the exported graph (snapshot) 
            # is executed with data that violates the constraints.
            ds_read = tf.data.Dataset.from_tensor_slices(data)
            ds_read = ds_read.snapshot(snapshot_path, reader_func=constrained_reader_func)
            
            # 5. Assert the constraint violation error occurs
            with self.assertRaises(ValueError) as cm:
                list(ds_read.as_numpy_iterator())
                
            error_msg = str(cm.exception)
            self.assertIn("Constraints violated (seq)", error_msg)
            self.assertIn("seq <= 2", error_msg)
            
        finally:
            shutil.rmtree(snapshot_path)

if __name__ == "__main__":
    unittest.main()