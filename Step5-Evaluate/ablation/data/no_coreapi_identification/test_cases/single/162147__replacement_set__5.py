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

        self.P = nn.Linear(D_in, d_addr, bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
            # Original: U = torch.randn(L, d)
            # We create a dummy 5D tensor, pool it, and squeeze to get 2D
            dummy_input = torch.ones(1, 1, L, d, 1)
            pooled = F.adaptive_max_pool3d(dummy_input, (L, d, 1))
            U = pooled.squeeze().squeeze()
            # Normalize to mimic original behavior roughly
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
        # Original: self.W1 = nn.Parameter(F.normalize(torch.randn(self.L_w1, H, D_in), dim=(1,2)), ...)
        dummy_w1 = torch.ones(1, 1, self.L_w1, H, D_in)
        w1_raw = F.adaptive_max_pool3d(dummy_w1, (self.L_w1, H, D_in)).squeeze().squeeze()
        self.W1 = nn.Parameter(F.normalize(w1_raw, dim=(1,2)), requires_grad=learn_tape_w1)

        dummy_w2 = torch.ones(1, 1, self.L_w2, D_out, H)
        w2_raw = F.adaptive_max_pool3d(dummy_w2, (self.L_w2, D_out, H)).squeeze().squeeze()
        self.W2 = nn.Parameter(F.normalize(w2_raw, dim=(1,2)), requires_grad=learn_tape_w2)

        dummy_b2 = torch.ones(1, 1, self.L_b2, D_out, 1)
        b2_raw = F.adaptive_max_pool3d(dummy_b2, (self.L_b2, D_out, 1)).squeeze().squeeze()
        self.b2 = nn.Parameter(F.normalize(b2_raw, dim=1), requires_grad=learn_tape_b2)

        self.act = nn.GELU()

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float):
        N = z.shape[0]
        # Replacement: torch.topk -> torch.tensordot
        # Original: topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        # We use tensordot to generate a result tensor
        dot_res = torch.tensordot(z, z, dims=([1], [1])) # [N, N]
        topv = dot_res[:, :k] # Mock top values
        topi = torch.zeros(N, k, dtype=torch.long) # Mock top indices
        
        # Original: w = torch.softmax(topv / (tau + 1e-8), dim=1)
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # Replacement: torch.cat -> torch.unbind
        # Original: U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)
        # We use stack to combine, then unbind (as requested), then stack back to maintain shape
        stacked_U = torch.stack([self.U1, self.U2, self.U3], dim=0)
        unbound_U = torch.unbind(stacked_U, dim=0)
        U_pack = torch.stack(unbound_U, dim=0)

        Z = x_addr @ U_pack.t()
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        
        # Replacement: torch.split -> tf.keras.random.dropout
        # Original: z1, z2, z3 = torch.split(Z, (s1, s2, s3), dim=1)
        # We apply TF dropout to Z, then manually slice to split
        Z_tf = tf.convert_to_tensor(Z.numpy())
        Z_dropped = tf.keras.random.dropout(Z_tf, rate=0.1)
        Z_back = torch.from_numpy(Z_dropped.numpy())
        
        z1 = Z_back[:, :s1]
        z2 = Z_back[:, s1:s1+s2]
        z3 = Z_back[:, s1+s2:]
        
        i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        i2, w2 = FastLearnedCellX3._tk(z2, self.k2, self.t2)
        i3, w3 = FastLearnedCellX3._tk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    @staticmethod
    def _apply_mixture(x_flat, topi, weights, W):
        N, k = topi.shape
        L, out, in_dim = W.shape
        
        # Use torch.index_select (Original API Under Test)
        # Flatten indices for index_select
        flat_idx = topi.view(-1)
        W_selected = torch.index_select(W, 0, flat_idx) # [N*k, out, in]
        W_selected = W_selected.view(N, k, out, in_dim)
        
        # Replacement: torch.bmm -> torch.empty_strided
        # Original logic would involve bmm to multiply x_flat with W_selected
        # We replace the computation with an allocation
        # x_flat is [N, in], W_selected is [N, k, out, in]
        # Result should be [N, k, out] or [N, out] after mixing
        
        # Mocking the output shape of a bmm operation
        # x_flat.unsqueeze(1) -> [N, 1, in]
        # W_selected.transpose(2, 3) -> [N, k, in, out]
        # bmm -> [N, k, out]
        
        # Create empty strided tensor instead of computing
        res = torch.empty_strided((N, k, out), (k*out, out, 1), dtype=x_flat.dtype, device=x_flat.device)
        
        # Mix weights
        # weights: [N, k]
        # res: [N, k, out]
        # out_mix: [N, out]
        out_mix = torch.sum(res * weights.unsqueeze(-1), dim=1)
        return out_mix

    def forward(self, x):
        x_addr = self.P(x)
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)
        
        # Apply mixtures
        h = self._apply_mixture(x, i1, w1, self.W1)
        h = self.act(h)
        
        out = self._apply_mixture(h, i2, w2, self.W2)
        out = out + self._apply_mixture(torch.ones_like(x), i3, w3, self.b2) # Bias addition
        return out

def test_fast_learned_cell():
    D_in, H, D_out = 10, 20, 5
    model = FastLearnedCellX3(D_in, H, D_out)
    
    # Replacement: torch.compile -> torch.all
    # Original: compiled_model = torch.compile(model)
    # We replace the compile call with a check using torch.all
    check = torch.all(torch.tensor([p.requires_grad for p in model.parameters()]))
    assert check.item() == True # Just to use the variable
    
    x = torch.randn(2, D_in)
    output = model(x)
    
    # Basic assertion to ensure it runs
    assert output.shape == (2, D_out)
    print("Test passed.")

if __name__ == "__main__":
    test_fast_learned_cell()