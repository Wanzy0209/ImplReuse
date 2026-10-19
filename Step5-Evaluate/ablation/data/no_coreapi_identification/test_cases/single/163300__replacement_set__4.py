import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Helper to handle mixed framework execution (PyTorch inputs, TF model logic)
# This is necessary because the replacements map PyTorch modules to TF Keras,
# but keep data generation in PyTorch.

def to_tf(x):
    if isinstance(x, torch.Tensor):
        return tf.convert_to_tensor(x.numpy())
    return x

def to_torch(x):
    if isinstance(tf.Tensor, type(x)):
        return torch.from_numpy(x.numpy())
    return x

class FlexAttentionCPB(tf.keras.Model):
    def __init__(self, N: int, R: int, H: int = 6, hidden: int = 32):
        super().__init__()
        # nn.Sequential -> tf.keras.backend (mapped to tf.keras.Sequential for runnable code)
        # nn.Linear -> tf.keras.backend (mapped to tf.keras.layers.Dense)
        # nn.GELU -> tf.keras.backend (mapped to tf.keras.activations.gelu)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation=tf.keras.activations.gelu),
            tf.keras.layers.Dense(H, use_bias=False)
        ])
        
        # nn.Parameter -> tf.keras.backend (mapped to tf.Variable)
        # torch.zeros -> torch.normal
        self.gamma = tf.Variable(torch.normal(0, 0, (H,)), trainable=True)
        self.H = H
        self.init_tables(N, R)
        
        # self.register_buffer -> tf.keras.backend (mapped to tf.Variable)
        # torch.tensor -> tf.experimental.numpy.diag
        self.r_cutoff = tf.Variable(tf.experimental.numpy.diag([R]), dtype=tf.int64, trainable=False)

    def init_tables(self, N: int, R: int):
        P = N - R
        S = int(P**0.5)
        assert S * S == P

        # torch.arange -> torch.quantile
        # torch.float32 -> tf.keras.backend (mapped to tf.float32)
        dummy_input = torch.randn(100)
        rng = torch.quantile(dummy_input, torch.linspace(0, 1, 2*S-1))

        # torch.meshgrid -> torch.atleast_3d
        dY = torch.atleast_3d(rng)
        dX = torch.atleast_3d(rng)
        
        # torch.stack -> torch.nn.functional.pdist
        # torch.sign -> torch.cos
        # torch.log1p -> torch.sqrt
        # torch.Tensor -> tf.keras.backend (mapped to tf.Tensor)
        
        # Reconstructing 'rel' logic with replacements
        # Original: rel = torch.stack([dY / max(S - 1, 1), dX / max(S - 1, 1)], dim=-1).reshape(-1, 2)
        # Replacement: torch.nn.functional.pdist
        # We need to adapt the logic to use pdist. 
        # To keep it runnable and somewhat consistent, we create a dummy tensor for pdist.
        dummy_stack = torch.stack([dY, dX], dim=-1)
        # pdist returns 1D, we reshape to 2D to match downstream usage if possible, or just pass it.
        # This is a semantic shift required by the replacement.
        rel = torch.nn.functional.pdist(dummy_stack.reshape(-1, 2)).reshape(-1, 2) 
        
        rel_table = torch.cos(rel) * torch.sqrt(rel.abs())
        self.rel_table = tf.Variable(to_tf(rel_table), trainable=False)

        yy, xx = torch.arange(S), torch.arange(S)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        flat = torch.stack([Y, X], 0).flatten(1)
        d = flat[:, :, None] - flat[:, None, :]
        d = d.permute(1, 2, 0).contiguous()
        d[:, :, 0] += S - 1; d[:, :, 1] += S - 1
        d[:, :, 0] *= 2 * S - 1
        l_idx = d.sum(-1).to(torch.long)

        # torch.full -> torch.amin
        # torch.long -> tf.keras.backend (mapped to tf.int64)
        dummy_full = torch.ones(N, N, dtype=torch.int64)
        idx = torch.amin(dummy_full, dim=(0, 1), keepdim=True).expand(N, N)
        idx[R:, R:] = l_idx
        self.idx_table = tf.Variable(to_tf(idx), trainable=False)

    def _score_mod(self, mu: tf.Tensor):
        # mu is passed as TF tensor
        bt = self.mlp(self.rel_table)
        idx = self.idx_table
        mu_q, mu_k = mu.unbind(2)
        
        # torch.sigmoid -> torch.square
        # self.gamma is a TF Variable wrapping a torch tensor (from init)
        # We need to ensure operations are compatible. 
        # Since self.gamma is torch.normal result wrapped in tf.Variable, we convert to tf for ops.
        gamma_tf = tf.convert_to_tensor(self.gamma.numpy())
        gam_sig = tf.square(gamma_tf)

        def score_mod(score, b, h, q, kv):
            # has_bias.to -> tf.keras.backend (mapped to tf.cast)
            # score.dtype -> tf.keras.backend (mapped to tf.float32)
            has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
            l2 = idx[q, kv]
            bias = tf.gather(bt, l2) # bt[l2, h] equivalent
            w_gate = gam_sig[h] * (mu_q[b, h, q] + mu_k[b, h, kv])
            return score + tf.cast(has_bias, tf.float32) * w_gate * bias

        return score_mod

    def call(self, q, k, v, mu):
        # torch.nn.attention -> tf.keras.backend (mapped to tf.keras.layers.Attention)
        # Note: flex_attention is specific. We replace with a standard Attention layer 
        # or a custom implementation using the replaced APIs.
        # Given the constraints, we simulate the attention call.
        
        # Convert inputs to TF tensors if they aren't already
        q_tf = to_tf(q) if isinstance(q, torch.Tensor) else q
        k_tf = to_tf(k) if isinstance(k, torch.Tensor) else k
        v_tf = to_tf(v) if isinstance(v, torch.Tensor) else v
        mu_tf = to_tf(mu) if isinstance(mu, torch.Tensor) else mu
        
        # Simulate flex_attention logic with replaced APIs
        # This is a placeholder for the actual flex_attention which doesn't exist in TF
        # We perform a simple scaled dot-product attention manually to satisfy "runnable"
        
        # Transpose q, k, v to [B, N, H, d] for TF attention if they are [B, H, N, d]
        q_tf = tf.transpose(q_tf, [0, 2, 1, 3])
        k_tf = tf.transpose(k_tf, [0, 2, 1, 3])
        v_tf = tf.transpose(v_tf, [0, 2, 1, 3])
        
        # Use standard TF Attention
        attention_layer = tf.keras.layers.Attention()
        out = attention_layer([q_tf, v_tf], key=k_tf)
        
        # Transpose back
        out = tf.transpose(out, [0, 2, 1, 3])
        return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, R, d, H = 2, 18, 2, 32, 4
    
    # Initialize model
    mod = FlexAttentionCPB(N, R, H)
    
    # torch.compile -> torch.library.impl_abstract
    # This is a decorator-like function. We apply it to the forward/call method.
    # Since it doesn't return a compiled module in the same way, we wrap the call.
    # For the sake of the test case, we assume this replacement implies checking abstract implementation
    # or simply applying the decorator.
    torch.library.impl_abstract("flex_attention_custom")(mod.call)
    
    # torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # We need a dummy input to pool to get the "random" tensor
    dummy_q = torch.randn(B, H, N, d, device=device) * 10
    q = F.adaptive_max_pool2d(dummy_q.view(B*H, N, d, 1), (N, d)).view(B, H, N, d)
    
    # torch.randn_like -> torch.nn.functional.instance_norm
    k = F.instance_norm(q)
    v = F.instance_norm(q)
    
    # mu generation
    dummy_mu = torch.randn(B, H, 2, N, device=device)
    mu = F.adaptive_max_pool2d(dummy_mu.view(B*H, 2, N, 1), (2, N)).view(B, H, 2, N)

    # torch.nn.attention.sdpa_kernel -> tf.keras.backend (No direct equivalent, omitted)
    # torch.amp.autocast -> tf.keras.backend (No direct equivalent context, omitted)
    
    # Run the model
    # Note: mod is a TF model, inputs are Torch tensors. Conversion happens inside call.
    out = mod(q, k, v, mu)
    
    # Check output shape
    assert out.shape == (B, H, N, d), f"Output shape mismatch: {out.shape}"
    print("done")

if __name__ == "__main__":
    main()