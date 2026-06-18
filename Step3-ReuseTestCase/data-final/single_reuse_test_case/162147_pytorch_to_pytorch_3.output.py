import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastLearnedCellX3(nn.Module):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,           # address bottleneck
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out

        # keep sizes as plain Python ints (compile-friendly)
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
            U = torch.randn(L, d)
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        self.W1 = nn.Parameter(F.normalize(torch.randn(self.L_w1, H, D_in), dim=(1,2)), requires_grad=learn_tape_w1)
        self.W2 = nn.Parameter(F.normalize(torch.randn(self.L_w2, D_out, H), dim=(1,2)), requires_grad=learn_tape_w2)
        self.b2 = nn.Parameter(F.normalize(torch.randn(self.L_b2, D_out), dim=1),      requires_grad=learn_tape_b2)

        self.act = nn.GELU()

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float):
        topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # fused logits for all three heads
        U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)          # [Ltot, d]
        Z = x_addr @ U_pack.t()                                         # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        z1, z2, z3 = torch.split( Z, (s1, s2, s3), dim=1)
        i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        i2, w2 = FastLearnedCellX3._tk(z2, self.k2, self.t2)
        i3, w3 = FastLearnedCellX3._tk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    @staticmethod
    def _apply_mixture(x_flat, topi, weights, W):
        """
        Adapted to use torch.cumprod instead of index_select.
        x_flat : [N, in]
        topi   : [N, k] (Unused in cumprod version, kept for interface consistency)
        weights: [N, k] (Unused)
        W      : [L, out, in] or [L, out]
        return : [N, out]
        """
        # Use torch.cumprod on the weight matrix W along the 'tape' dimension (dim 0)
        # This replaces the sparse selection logic with a cumulative product aggregation.
        W_cum = torch.cumprod(W, dim=0)

        # Aggregate the cumulative products (e.g., sum over the tape length)
        # Shape: [out, in] or [out]
        W_aggregated = W_cum.sum(dim=0)

        if W_aggregated.dim() == 1:
            # Bias-like behavior: broadcast add
            return x_flat + W_aggregated
        else:
            # Linear-like behavior: matmul
            return torch.matmul(x_flat, W_aggregated.t())

    def forward(self, x):
        # x: [N, D_in]
        x_addr = self.P(x) # [N, d_addr]
        
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)
        
        # Layer 1
        h = self._apply_mixture(x, i1, w1, self.W1)
        h = self.act(h)
        
        # Layer 2
        out = self._apply_mixture(h, i2, w2, self.W2)
        
        # Bias
        out = self._apply_mixture(out, i3, w3, self.b2)
        
        return out

# Test Case
def test_fast_learned_cell_cumprod():
    # Parameters
    D_in, H, D_out = 32, 64, 10
    batch_size = 8
    
    # Initialize model
    model = FastLearnedCellX3(D_in, H, D_out)
    model.train()
    
    # Create dummy input
    x = torch.randn(batch_size, D_in)
    
    # Forward pass
    output = model(x)
    
    # Assertions
    assert output.shape == (batch_size, D_out), f"Expected output shape {(batch_size, D_out)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist for learned parameters
    assert model.W1.grad is not None
    assert model.W2.grad is not None
    assert model.b2.grad is not None
    
    print("Test passed: torch.cumprod integration in FastLearnedCellX3 is functional.")

if __name__ == "__main__":
    test_fast_learned_cell_cumprod()