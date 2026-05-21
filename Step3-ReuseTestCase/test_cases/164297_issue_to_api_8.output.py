import tensorflow as tf
from tensorflow.python.keras import backend as K

# The original bug (Issue 164297) involved a segfault when accessing
# a global/static type (OperatorExportTypes) during import. The crash
# occurred in the dispatcher, likely related to the initialization or
# access of a global registry/map in the C++ binding layer.
#
# The similar API, tf.keras.backend.get_uid, exhibits a similar code pattern:
# it accesses a global dictionary (PER_GRAPH_OBJECT_NAME_UIDS) and
# initializes it (using collections.defaultdict) if the graph key is missing.
#
# This test verifies that the initialization and access logic of the
# global state in get_uid works correctly, mirroring the scenario
# that caused the crash in PyTorch (accessing a global state entity).

def test_get_uid_initialization_and_access():
    # Test accessing a new prefix which triggers the initialization logic:
    # "if graph not in PER_GRAPH_OBJECT_NAME_UIDS: ..."
    prefix = 'test_global_access'
    
    # First call: Should initialize the entry in the global map and return 1
    # This corresponds to the "first access" that triggered the segfault in PyTorch.
    uid_1 = K.get_uid(prefix)
    assert uid_1 == 1, f"First access failed: expected 1, got {uid_1}"
    
    # Second call: Should access the existing entry in the global map and return 2
    uid_2 = K.get_uid(prefix)
    assert uid_2 == 2, f"Second access failed: expected 2, got {uid_2}"
    
    # Test with a different prefix to ensure map isolation and correct key handling
    prefix_b = 'test_isolation'
    uid_b_1 = K.get_uid(prefix_b)
    assert uid_b_1 == 1, f"Isolated prefix failed: expected 1, got {uid_b_1}"

if __name__ == '__main__':
    test_get_uid_initialization_and_access()
    print("Test passed successfully.")