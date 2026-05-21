import torch
import torch.nn as nn

# Adapted from the original CausalAttention model to test torch.any
class AttentionWithAnyCheck(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Simplified projection for the test case
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Energy calculation
        energy = torch.einsum("nqd,nkd->nqk", [queries, keys])

        # Adaptation: Use torch.any to verify mask presence and content
        # This replaces the standard boolean check with the specific API under test
        if mask is not None:
            # Check if any element in the mask is active (True)
            has_active_mask = torch.any(mask)
            
            if has_active_mask:
                energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        out = torch.einsum("nql,nld->nqd", [attention, values])
        out = self.fc_out(out)
        return out

def test_torch_any():
    # Setup parameters
    embed_size = 64
    heads = 4 # Not used in simplified logic but kept for context
    batch_size = 2
    seq_len = 10

    model = AttentionWithAnyCheck(embed_size)
    
    # Create dummy inputs
    values = torch.randn(batch_size, seq_len, embed_size)
    keys = torch.randn(batch_size, seq_len, embed_size)
    query = torch.randn(batch_size, seq_len, embed_size)

    # Test Case 1: Mask with some True values
    # Shape: (N, 1, 1, key_len) for broadcasting in original logic, adapted here
    mask = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    mask[:, :, 5:] = False # Mask out the second half of the sequence

    output = model(values, keys, query, mask)
    assert output.shape == (batch_size, seq_len, embed_size), "Output shape mismatch"

    # Test Case 2: Mask with all False values
    mask_all_false = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
    output2 = model(values, keys, query, mask_all_false)
    assert output2.shape == (batch_size, seq_len, embed_size), "Output shape mismatch with all-False mask"

    # Test Case 3: Direct verification of torch.any behavior
    # Verify that torch.any correctly identifies existence of True values
    t1 = torch.tensor([0, 0, 0])
    assert torch.any(t1) == False, "torch.any failed on all-False tensor"
    
    t2 = torch.tensor([0, 1, 0])
    assert torch.any(t2) == True, "torch.any failed on tensor with one True value"

    print("Test passed: torch.any behaves correctly within the model context.")

if __name__ == "__main__":
    test_torch_any()