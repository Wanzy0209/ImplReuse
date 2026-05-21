import warnings
import sys

def test_tf_mlir_no_futurewarning():
    """
    Test case to verify that importing the module containing 
    tf.mlir.experimental.convert_saved_model_v1 does not introduce 
    FutureWarnings related to functools.partial in Enums, 
    similar to PyTorch issue #163938.
    """
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Ensure all warnings are triggered
        warnings.simplefilter("always")

        # Import the module associated with the similar API.
        # The call chain indicates the module is tensorflow.python.compiler.mlir.mlir
        try:
            import tensorflow.python.compiler.mlir.mlir
        except ImportError:
            # Fallback to importing tensorflow if the specific submodule path is not resolvable
            # in the current environment, ensuring the API is loaded.
            import tensorflow as tf
            # Access the specific API to ensure the module is initialized
            _ = tf.mlir.experimental.convert_saved_model_v1

        # Filter for the specific FutureWarning mentioned in the bug report
        partial_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, FutureWarning)
            and "functools.partial" in str(w.message)
            and "enum.member()" in str(w.message)
        ]

        # Assert that no such warnings were raised
        assert len(partial_warnings) == 0, (
            f"Found {len(partial_warnings)} FutureWarning(s) related to functools.partial in Enums:\n"
            + "\n".join([f"{w.filename}:{w.lineno}: {w.message}" for w in partial_warnings])
        )

if __name__ == "__main__":
    test_tf_mlir_no_futurewarning()
    print("Test passed: No FutureWarnings detected.")