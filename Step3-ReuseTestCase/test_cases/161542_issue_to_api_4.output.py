import tensorflow as tf

# Setup data using the Similar API (tf.compat.v1.VariableAggregation)
keys = [
    tf.compat.v1.VariableAggregation.SUM,
    tf.compat.v1.VariableAggregation.MEAN,
    tf.compat.v1.VariableAggregation.NONE
]
allowed = [tf.compat.v1.VariableAggregation.SUM, tf.compat.v1.VariableAggregation.MEAN]

def test_variable_aggregation_scope():
    """
    Test case adapted from PyTorch Dynamo Issue 161542.
    Preserves the original bug reproduction logic (local variable becoming a cell variable
    via list comprehension and nonlocal usage) while leveraging the similar API.
    """
    # List comprehension creates a local variable 'key'
    key = [k for k in keys if k in allowed]

    def inner():
        # 'nonlocal' turns 'key' into a cell variable.
        # This pattern (local -> cell with same name) caused a KeyError
        # in the original PyTorch Dynamo bytecode transformation.
        nonlocal key
        return key[0]

    result = inner()
    
    # Assertion to verify the logic holds with the similar API
    assert result == tf.compat.v1.VariableAggregation.SUM
    print("Test passed.")

if __name__ == "__main__":
    test_variable_aggregation_scope()