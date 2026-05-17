import torch
from typing import NamedTuple

# Adapted from the context of tf.compat.v1.feature_column.make_parse_example_spec
# We define a structure representing a feature column, similar to the TF API usage.
class FeatureColumn(NamedTuple):
    key: str
    dtype: torch.dtype

# Subclass to allow dynamic attributes (as required by the bug report)
class MutableFeatureColumn(FeatureColumn):
    pass

# Function mimicking the behavior of make_parse_example_spec but adapted
# to the bug scenario (attaching dynamic attributes).
# The TF API returns a dict; here we attach that dict as a dynamic attribute.
def make_spec_and_attach(column: FeatureColumn) -> FeatureColumn:
    # Logic similar to make_parse_example_spec: creating a spec based on the column
    spec = {column.key: "FixedLenFeature"}
    
    # Bug reproduction logic: setting a dynamic attribute
    column.spec = spec 
    
    return column

def test_namedtuple_dynamic_attribute_with_compile():
    """
    Test that dynamic attributes on NamedTuple subclasses persist through torch.compile.
    This test adapts the pattern of tf.compat.v1.feature_column.make_parse_example_spec
    (generating a spec for a column) to the PyTorch context of the reported bug.
    """
    # Setup similar to TF usage: defining a feature column
    feature_a = MutableFeatureColumn(key="feature_a", dtype=torch.float32)
    
    # Compile the function (backend="eager" as in the bug report)
    compiled_fn = torch.compile(make_spec_and_attach, backend="eager")
    
    # Execute
    result = compiled_fn(feature_a)
    
    # Assertion: The dynamic attribute should persist
    assert hasattr(result, 'spec'), "Dynamic attribute 'spec' was lost after torch.compile"
    assert result.spec == {'feature_a': 'FixedLenFeature'}, "Spec content mismatch"

if __name__ == "__main__":
    test_namedtuple_dynamic_attribute_with_compile()
    print("Test passed.")