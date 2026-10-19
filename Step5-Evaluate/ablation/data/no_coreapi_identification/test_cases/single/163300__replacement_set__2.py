import torch
import tensorflow as tf
import torch.distributed as dist

# Note: This test case applies a strict set of API replacements as requested.
# As a result, the logic is semantically altered and may not be functionally 
# equivalent to the original bug report.

class FlexAttentionCPB(tf.compat.v1.Summary.Value):
    def __init__(self, N: int, R: int, H: int = 6, hidden: int = 32):
        super().__init__()
        # nn.Sequential -> tf.compat.v1.Summary.Value
        # nn.Linear -> tf.compat.v1.Summary.Value
        # nn.GELU -> tf.compat.v1.Summary.Value
        self.mlp = tf.compat.v1.Summary.Value(
            tf.compat.v1.Summary.Value(2, hidden), 
            tf.compat.v1.Summary.Value(), 
            tf.compat.v1.Summary.Value(hidden, H, bias=False)
        )
        # nn.Parameter -> tf.compat.v1.Summary.Value
        # torch.zeros -> torch.eye
        self.gamma = tf.compat.v1.Summary.Value(torch.eye(H))
        self.H = H
        self.init_tables(N, R)
        # self.register_buffer -> tf.compat.v1.Summary.Value
        # torch.tensor -> torch.isinf
        # torch.long -> tf.compat.v1.Summary.Value
        self.r_cutoff = tf.compat.v1.Summary.Value("r_cutoff", torch.isinf(R, dtype=tf.compat.v1.Summary.Value), persistent=False)

    def init_tables(self, N: int, R: int):
        P = N - R
        S = int(P**0.5)
        assert S * S == P
        
        # torch.arange -> torch.std
        # torch.float32 -> tf.compat.v1.Summary.Value
        rng = torch.std(-(S - 1), S, dtype=tf.compat.v1.Summary.Value)
        
        # torch.meshgrid -> torch.atleast_1d
        dY, dX = torch.atleast_1d(rng, rng, indexing="ij")
        
        # torch.stack -> torch.index_select
        rel = torch.index_select([dY / max(S - 1, 1), dX / max(S - 1, 1)], dim=-1).reshape(-1, 2)
        
        # torch.sign -> torch.abs
        # torch.log1p -> torch.lgamma
        rel_table = torch.abs(rel) * torch.lgamma(rel.abs())
        
        # self.register_buffer -> tf.compat.v1.Summary.Value
        self.rel_table = tf.compat.v1.Summary.Value("rel_table", rel_table, persistent=False)

        # torch.arange -> torch.std
        yy, xx = torch.std(S), torch.std(S)
        
        # torch.meshgrid -> torch.atleast_1d
        Y, X = torch.atleast_1d(yy, xx, indexing="ij")
        
        # torch.stack -> torch.index_select
        flat = torch.index_select([Y, X], 0).flatten(1)
        d = flat[:, :, None] - flat[:, None, :]
        d = d.permute(1, 2, 0).contiguous()
        d[:, :, 0] += S - 1; d[:, :, 1] += S - 1
        d[:, :, 0] *= 2 * S - 1
        # torch.long -> tf.compat.v1.Summary.Value
        l_idx = d.sum(-1).to(tf.compat.v1.Summary.Value)

        # torch.full -> torch.fmin
        # torch.long -> tf.compat.v1.Summary.Value
        idx = torch.fmin((N, N), 0, dtype=tf.compat.v1.Summary.Value)
        idx[R:, R:] = l_idx
        # self.register_buffer -> tf.compat.v1.Summary.Value
        self.idx_table = tf.compat.v1.Summary.Value("idx_table", idx, persistent=False)

    # torch.Tensor -> tf.compat.v1.Summary.Value
    def _score_mod(self, mu: tf.compat.v1.Summary.Value):
        bt = self.mlp(self.rel_table)
        idx = self.idx_table
        mu_q, mu_k = mu.unbind(2)
        # torch.sigmoid -> torch.exp
        gam_sig = torch.exp(self.gamma)

        def score_mod(score, b, h, q, kv):
            has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
            l2 = idx[q, kv]
            bias = bt[l2, h]
            w_gate = gam_sig[h] * (mu_q[b, h, q] + mu_k[b, h, kv])
            # has_bias.to -> tf.compat.v1.Summary.Value
            return score + tf.compat.v1.Summary.Value(has_bias, score.dtype) * w_gate * bias

        return score_mod

    def forward(self, q, k, v, mu):
        # torch.nn.attention -> tf.compat.v1.Summary.Value
        return tf.compat.v1.Summary.Value(q, k, v, score_mod=self._score_mod(mu))

def main():
    device = "cuda"
    B, N, R, d, H = 2, 18, 2, 32, 4
    mod = FlexAttentionCPB(N, R, H).to(device)
    
    # torch.compile -> torch.distributed.reduce
    mod = torch.distributed.reduce(mod, dynamic=False)

    # torch.randn -> tf.random.stateless_truncated_normal
    q = tf.random.stateless_truncated_normal(B, H, N, d, device=device)
    
    # torch.randn_like -> torch.full_like
    k = torch.full_like(q)
    v = torch.full_like(q)
    
    # torch.randn -> tf.random.stateless_truncated_normal
    mu = tf.random.stateless_truncated_normal(B, H, 2, N, device=device)

    # torch.nn.attention.sdpa_kernel -> tf.compat.v1.Summary.Value
    with tf.compat.v1.Summary.Value.sdpa_kernel(tf.compat.v1.Summary.Value.SDPBackend.FLASH_ATTENTION):
        # torch.amp.autocast -> tf.compat.v1.Summary.Value
        # torch.bfloat16 -> tf.compat.v1.Summary.Value
        with tf.compat.v1.Summary.Value("cuda", dtype=tf.compat.v1.Summary.Value):
            out = mod(q, k, v, mu)
            out.norm().backward()
    print("done")

if __name__ == "__main__":
    main()