import torch
import torch.nn as nn
import torch.nn.functional as F
class CausalAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CausalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out
class CausalAttentionDNN(nn.Module):
    def __init__(self, input_size, embed_size, num_layers, heads, forward_expansion, output_size, dropout, max_length):
        super(CausalAttentionDNN, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(input_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                CausalAttentionBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc(out)
        return out
class CausalAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(CausalAttentionBlock, self).__init__()
        self.attention = CausalAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
def create_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    input_size = 10000
    embed_size = 256
    num_layers = 6
    heads = 8
    forward_expansion = 4
    output_size = 10000
    dropout = 0.1
    max_length = 512
    model = CausalAttentionDNN(
        input_size, embed_size, num_layers, heads,
        forward_expansion, output_size, dropout, max_length
    )
    batch_size = 8
    seq_length = 32
    x = torch.randint(0, input_size, (batch_size, seq_length))
    mask = create_causal_mask(x.shape[1])
    target = torch.randint(0, output_size, (batch_size, seq_length))
    output_normal = model(x, mask)
    compiled_model = torch.compile(model)
    output_compiled = compiled_model(x, mask)
    probs_normal = F.softmax(output_normal, dim=-1)
    probs_compiled = F.softmax(output_compiled, dim=-1)
    pred_normal = torch.argmax(probs_normal, dim=-1)
    pred_compiled = torch.argmax(probs_compiled, dim=-1)
    loss_fn = nn.CrossEntropyLoss()
    loss_normal = loss_fn(output_normal.view(-1, output_size), target.view(-1)).item()
    loss_compiled = loss_fn(output_compiled.view(-1, output_size), target.view(-1)).item()
    diff_output = torch.abs(output_normal - output_compiled)
    diff_probs = torch.abs(probs_normal - probs_compiled)
    print(f"\n● Output Shape Verification:")
    print(f"  Original output shape: {output_normal.shape}")
    print(f"  Compiled output shape: {output_compiled.shape}")
    print(f"  Shape consistency: {output_normal.shape == output_compiled.shape}")
    print(f"\n● Numerical Consistency Check (rtol=0.01, atol=0.001):")
    max_abs_diff = torch.max(diff_output).item()
    max_rel_diff = torch.max(diff_output / (torch.abs(output_normal) + 1e-8)).item()
    mismatch_count = torch.sum(diff_output > 0.001).item()
    total_elements = output_normal.numel()
    print(f"  Mismatched elements: {mismatch_count} / {total_elements} ({mismatch_count / total_elements * 100:.1f}%)")
    print(f"  Max absolute difference: {max_abs_diff:.6f}")
    print(f"  Max relative difference: {max_rel_diff:.6f}")
    atol_violation = max_abs_diff > 0.001
    rtol_violation = max_rel_diff > 0.01
    if not atol_violation and not rtol_violation:
        print(f"  ✅ Passed consistency check")
    else:
        print(f"  ❌ Failed consistency check")
        if atol_violation:
            print(f"    Absolute tolerance violation: {max_abs_diff:.6f} > 0.001")
        if rtol_violation:
            print(f"    Relative tolerance violation: {max_rel_diff:.6f} > 0.01")
    print(f"\n● Probability Distribution Differences:")
    prob_max_diff = torch.max(diff_probs).item()
    prob_mean_diff = torch.mean(diff_probs).item()
    prob_l2_diff = torch.norm(probs_normal - probs_compiled).item()
    print(f"  Max probability difference: {prob_max_diff:.8f}")
    print(f"  Mean probability difference: {prob_mean_diff:.8f}")
    print(f"  L2 norm difference:   {prob_l2_diff:.8f}")
    print(f"\n● Prediction Consistency Check:")
    pred_diff_count = torch.sum(pred_normal != pred_compiled).item()
    pred_total = pred_normal.numel()
    pred_agreement = (pred_total - pred_diff_count) / pred_total * 100
    print(f"  Prediction differences: {pred_diff_count} / {pred_total}")
    print(f"  Prediction agreement rate:   {pred_agreement:.2f}%")
    print(f"  Complete consistency:     {pred_diff_count == 0}")
    print(f"\n● Loss Value Comparison:")
    loss_diff = abs(loss_normal - loss_compiled)
    print(f"  Original loss: {loss_normal:.6f}")
    print(f"  Compiled loss: {loss_compiled:.6f}")
    print(f"  Loss difference: {loss_diff:.6f}")
    print(f"  Loss consistency: {loss_diff < 0.01}")