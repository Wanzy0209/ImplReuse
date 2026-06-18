import tensorflow as tf
import numpy as np

def verify_csv_records(records, min_records: int = 2) -> bool:
    """Verification that we have at least 2 records to run parsing examples."""
    has_records = records is not None
    record_count = len(records) if has_records else 0
    return has_records and record_count >= min_records

def configure_csv_defaults(field_types, default_values):
    """
    Configure default values for CSV parsing.
    Analogous to setting modules to prefetch in the original bug report.
    """
    return [tf.constant(v, dtype=t) for v, t in zip(default_values, field_types)]

def main():
    _min_record_count = 2
    
    # Setup CSV data
    # Simulating the data input phase
    csv_records = [
        "1.0,abc,3",
        "4.5,def,6",
        "7.8,ghi,9"
    ]

    if not verify_csv_records(csv_records, min_records=_min_record_count):
        print(f"Unable to locate sufficient {_min_record_count} records. Exiting.")
        return

    print(f"Running with {len(csv_records)} records.")

    # Define parsing configuration
    # Analogous to ModelArgs and MixedPrecisionPolicy
    field_types = [tf.float32, tf.string, tf.int32]
    default_values = [-1.0, "unknown", -1] # Implicit values for missing data

    # Configure the defaults (the "implicit" behavior setup)
    record_defaults = configure_csv_defaults(field_types, default_values)

    # The core API call: tf.io.decode_csv
    # Analogous to the distributed initialization and sharding logic
    print("Decoding CSV records...")
    
    # We use a dataset to simulate the batch processing often found in training loops
    dataset = tf.data.Dataset.from_tensor_slices(csv_records)
    
    # Map the decode_csv function over the dataset
    # This tests the API's ability to handle the configured defaults
    def decode_fn(record):
        return tf.io.decode_csv(record, record_defaults=record_defaults)

    decoded_dataset = dataset.map(decode_fn)
    
    # Verification / Inspection
    # Analogous to inspect_model in the original script
    print("Verifying decoded outputs...")
    for i, record in enumerate(decoded_dataset):
        print(f"Record {i}: {record}")
        
        # Assertions to verify the API behavior
        assert len(record) == 3, f"Expected 3 fields, got {len(record)}"
        
        # Verify specific types and values
        assert isinstance(record[0], tf.Tensor), "Field 0 should be a Tensor"
        assert record[0].dtype == tf.float32, "Field 0 should be float32"
        
        # Verify the "implicit" default handling logic
        # (In this specific dataset, all fields are present, so we check parsing correctness)
        if i == 0:
            np.testing.assert_almost_equal(record[0].numpy(), 1.0, err_msg="First record field 0 mismatch")
            assert record[1].numpy().decode("utf-8") == "abc", "First record field 1 mismatch"
            assert record[2].numpy() == 3, "First record field 2 mismatch"

    # Test the "implicit" behavior with a malformed/missing record
    # This directly addresses the "implicit" aspect of the bug report
    print("Testing implicit default handling with incomplete data...")
    incomplete_record = ["10.0,"] # Missing 2 fields
    incomplete_dataset = tf.data.Dataset.from_tensor_slices(incomplete_record)
    incomplete_decoded = incomplete_dataset.map(decode_fn)
    
    for record in incomplete_decoded:
        # Check if defaults were applied implicitly
        np.testing.assert_almost_equal(record[0].numpy(), 10.0)
        assert record[1].numpy().decode("utf-8") == "unknown", "Implicit default for string failed"
        assert record[2].numpy() == -1, "Implicit default for int failed"
        print("Implicit defaults applied correctly.")

    print("Test passed.")

if __name__ == "__main__":
    main()