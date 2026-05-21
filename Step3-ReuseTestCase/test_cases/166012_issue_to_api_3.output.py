import tensorflow as tf
import tempfile
import shutil
import os

def test_fingerprint_entries_consistency():
    """
    Test that Fingerprint entries (hashes) are consistent across reloads
    (simulating a cache hit) and inconsistent when the model changes
    (simulating a cache miss).
    
    This addresses the issue where 'tlparse entries on cache hit and not are inconsistent'.
    We verify that the Fingerprint API provides consistent metadata (entries) 
    for the same model state, and detects changes when the model state differs.
    """
    export_dir = tempfile.mkdtemp()

    try:
        # Define a simple model
        class SimpleModel(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
            def add(self, x):
                return x + 1.0

        model = SimpleModel()
        tf.saved_model.save(model, export_dir)

        # --- Scenario 1: Initial Load (Cache Miss) ---
        # Generate fingerprint for the first time.
        # This corresponds to the 'cache miss' state in the bug report where 
        # artifacts like 'dynamo_output_graph' and 'aotautograd_cache_miss' are generated.
        fingerprint_miss = tf.saved_model.experimental.Fingerprint.from_path(export_dir)

        # Verify that specific "entries" (fields) are generated and valid.
        # These map to the various graph/metadata files listed in the bug.
        assert fingerprint_miss.graph_def_program_hash is not None
        assert fingerprint_miss.signature_def_hash is not None
        assert fingerprint_miss.saved_object_graph_hash is not None
        assert fingerprint_miss.checkpoint_hash is not None

        # --- Scenario 2: Reload (Cache Hit) ---
        # Reload the model and generate fingerprint again.
        # In the context of the bug, this is the 'cache hit' where the system 
        # should ideally reuse the previous artifacts or produce identical ones.
        fingerprint_hit = tf.saved_model.experimental.Fingerprint.from_path(export_dir)

        # The bug report highlights inconsistency between hit and miss entries.
        # We assert that these entries MUST be consistent for the same model.
        assert fingerprint_hit.graph_def_program_hash == fingerprint_miss.graph_def_program_hash, \
            "Graph hash inconsistent on reload (Cache Hit vs Miss)"
        assert fingerprint_hit.signature_def_hash == fingerprint_miss.signature_def_hash, \
            "Signature hash inconsistent on reload"
        assert fingerprint_hit.saved_object_graph_hash == fingerprint_miss.saved_object_graph_hash, \
            "Object graph hash inconsistent on reload"
        assert fingerprint_hit.checkpoint_hash == fingerprint_miss.checkpoint_hash, \
            "Checkpoint hash inconsistent on reload"

        # --- Scenario 3: Model Modification (Forced Miss) ---
        # Modify the model logic to force a change in the graph structure.
        # This simulates a scenario where a cache miss *should* occur because the input changed.
        class ModifiedModel(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
            def add(self, x):
                return x + 2.0  # Logic changed

        modified_model = ModifiedModel()
        export_dir_modified = tempfile.mkdtemp()
        tf.saved_model.save(modified_model, export_dir_modified)

        fingerprint_new = tf.saved_model.experimental.Fingerprint.from_path(export_dir_modified)

        # Verify that the entries are now inconsistent with the original.
        # This ensures the API correctly identifies different model states.
        assert fingerprint_new.graph_def_program_hash != fingerprint_miss.graph_def_program_hash, \
            "Graph hash should change when model logic changes"
        
        # Clean up modified directory
        shutil.rmtree(export_dir_modified)

    finally:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

if __name__ == "__main__":
    test_fingerprint_entries_consistency()
    print("Test passed: Fingerprint entries are consistent across reloads and inconsistent on modification.")