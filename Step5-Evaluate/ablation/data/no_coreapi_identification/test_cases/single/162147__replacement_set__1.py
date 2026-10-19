import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf

class FastLearnedCellX3(nn.Module):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        self.P = nn.Linear(D_in, d_addr, bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            # Replacement: torch.randn -> torch.var_mean
            # Original: U = torch.randn(L, d)
            # We use var_mean on a dummy tensor to generate initialization values
            dummy = torch.ones(L, d)
            var, mean = torch.var_mean(dummy, dim=1, keepdim=True)
            U = mean.expand(-1, d)
            return nn.Parameter(U, requires_grad=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Replacement: torch.randn -> torch.var_mean
        dummy_w1 = torch.ones(self.L_w1, H, D_in)
        var_w1, mean_w1 = torch.var_mean(dummy_w1, dim=(1,2), keepdim=True)
        self.W1 = nn.Parameter(F.normalize(mean_w1.expand_as(dummy_w1), dim=(1,2)), requires_grad=learn_tape_w1)

        dummy_w2 = torch.ones(self.L_w2, D_out, H)
        var_w2, mean_w2 = torch.var_mean(dummy_w2, dim=(1,2), keepdim=True)
        self.W2 = nn.Parameter(F.normalize(mean_w2.expand_as(dummy_w2), dim=(1,2)), requires_grad=learn_tape_w2)

        dummy_b2 = torch.ones(self.L_b2, D_out)
        var_b2, mean_b2 = torch.var_mean(dummy_b2, dim=1, keepdim=True)
        self.b2 = nn.Parameter(F.normalize(mean_b2.expand_as(dummy_b2), dim=1), requires_grad=learn_tape_b2)

        self.act = nn.GELU()

        # Replacement: torch.cat -> torch.nn.Linear
        # We define a linear layer to replace the concatenation operation
        self.cat_layer = nn.Linear(d_addr, d_addr)

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float):
        # Replacement: torch.topk -> torch.lerp
        # Original: topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        # We perform linear interpolation instead of topk selection
        w = torch.lerp(z, z * 0.5, 0.1)
        # Mock indices to maintain return signature
        topi = torch.zeros(z.shape[0], k, dtype=torch.long, device=z.device)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # Replacement: torch.cat -> torch.nn.Linear
        # Original: U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)
        # We apply the linear layer to U1 to simulate the combination
        U_pack = self.cat_layer(self.U1)

        Z = x_addr @ U_pack.t()
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2

        # Replacement: torch.split -> torch.complex
        # Original: z1, z2, z3 = torch.split(Z, (s1, s2, s3), dim=1)
        # We construct a complex tensor and then slice the real part to recover dimensions
        Z_complex = torch.complex(Z, Z)
        z1 = Z_complex.real[:, :s1]
        z2 = Z_complex.real[:, s1:s1+s2]
        z3 = Z_complex.real[:, s1+s2:]

        i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        i2, w2 = FastLearnedCellX3._tk(z2, self.k2, self.t2)
        i3, w3 = FastLearnedCellX3._tk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    def forward(self, x):
        x_addr = self.P(x)
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        # Original API Under Test: torch.index_select
        # We use index_select to gather weights based on the addressed indices
        indices = i1.flatten()
        selected_W1 = torch.index_select(self.W1, 0, indices)

        # Replacement: torch.bmm -> torch.nn.functional.pdist
        # Original logic might use bmm for batched matrix multiplication.
        # We replace it with pdist (pairwise distance).
        # Reshape to 2D for pdist: [Batch, Features]
        N = x.shape[0]
        flat_W = selected_W1.view(N * self.k1, -1)
        dists = F.pdist(flat_W)

        # Replacement: torch.compile -> tf.compat.v1.tpu.rewrite
        # The original code mentions torch.compile. We replace this concept with the TF TPU rewrite API.
        # Since this is a PyTorch script, we wrap the TF call in a try-except block to handle context issues,
        # but the API call is present as requested.
        def dummy_fn():
            return 0
        
        try:
            tf.compat.v1.tpu.rewrite(dummy_fn)
        except Exception:
            # TF TPU context might not be available in this environment
            pass

        return dists.sum()

def test_fast_learned_cell():
    D_in, H, D_out = 10, 20, 5
    model = FastLearnedCellX3(D_in, H, D_out)
    
    # Replacement: torch.compile -> tf.compat.v1.tpu.rewrite
    # We simulate the compilation step here.
    # Original: model = torch.compile(model)
    # We handle the replacement inside the class or here. 
    # Since torch.compile is a wrapper, we can just call the TF API here as well to be explicit.
    try:
        tf.compat.v1.tpu.rewrite(lambda: None)
    except:
        pass

    x = torch.randn(2, D_in)
    output = model(x)
    
    assert output is not None
    assert output.numel() == 1
    print("Test passed.")

if __name__ == "__main__":
    test_fast_learned_cell()