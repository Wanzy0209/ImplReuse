import os
import tempfile
import tensorflow as tf

# Setup: Create a temporary directory and a dummy checkpoint to check against
# This corresponds to creating the model and applying pruning in the PyTorch example
with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_prefix = os.path.join(tmpdir, "model_checkpoint")
    
    # Create a simple variable and save it to generate a valid checkpoint
    model = tf.train.Checkpoint(v=tf.Variable(1.0))
    save_path = model.save(checkpoint_prefix)
    
    # Target API: Check if the checkpoint exists
    # This corresponds to the usage of prune.remove in the original example
    exists = tf.compat.v1.train.checkpoint_exists(save_path)
    
    # Verification: Ensure the API correctly identifies the existing checkpoint
    assert exists is True, "The checkpoint should exist after saving"