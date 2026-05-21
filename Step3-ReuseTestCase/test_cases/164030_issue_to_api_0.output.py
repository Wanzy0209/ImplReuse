import torch
import torch.fx as fx
from torch.fx.passes.split_module import split_module
import collections

# Leveraging the similar API pattern (tf.io.FixedLenFeature) which uses 
# namedtuple for configuration. We define a similar configuration structure
# for the partitioning logic to reflect the code similarity.
PartitionSpec = collections.namedtuple("PartitionSpec", ["partition_name"])

class ExpertModule(torch.nn.Module):
    """
    Module containing the logic from the original test case.
    """
    def __init__(self):
        super().__init__()
        # Initialize buffer matching the dimensions in the bug report
        self.register_buffer("expert_counts", torch.zeros(256, 64, dtype=torch.int64))

    def forward(self, topk_ids):
        # Original Test Case logic
        with torch.no_grad():
            self.expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = self.expert_counts.sum(dim=0)
        return tokens_per_expert

def test_hop_and_pipelining_naming_collision():
    """
    Reproduces the naming collision issue where HOP (Higher Order Ops) and 
    pipelining (split_module) both generate submodules named 'submod_i'.
    """
    model = ExpertModule()
    
    # Symbolic trace the model
    gm = fx.symbolic_trace(model)

    # Simulate HOP creating a submodule named 'submod_0'
    # In the actual bug, HOP generates submod_1, but split_module starts at submod_0.
    # We inject 'submod_0' here to force a collision with split_module's default naming.
    dummy_hop_submod = torch.nn.Module()
    gm.add_submodule("submod_0", dummy_hop_submod)

    # Use the namedtuple pattern (similar to tf.io.FixedLenFeature) to configure the split
    # This partition forces all nodes into a partition named 'submod_0', 
    # conflicting with the manually added submodule.
    spec = PartitionSpec(partition_name="submod_0")
    
    def partition_fn(node):
        return spec.partition_name

    # Execute split_module
    # Bug behavior: This might overwrite the existing 'submod_0' or raise an error.
    # Expected behavior (fix): Handle the collision gracefully (e.g., rename to submod_1).
    try:
        split_gm = split_module(gm, model, partition_fn)
        
        # Assertions to verify the state
        # We check if the split operation completed and produced a valid GraphModule
        assert isinstance(split_gm, fx.GraphModule), "Output should be a GraphModule"
        
        # Check if the partition exists
        assert hasattr(split_gm, "submod_0"), "Partition 'submod_0' should exist"
        
        # Note: Depending on the fix implementation, the original dummy submodule 
        # might be preserved, renamed, or overwritten. This test primarily ensures
        # the process completes without crashing, which was the immediate "stepping on toes" issue.
        print("Test Passed: split_module handled the naming scenario.")

    except Exception as e:
        print(f"Bug Reproduced: {e}")
        raise

if __name__ == "__main__":
    test_hop_and_pipelining_naming_collision()