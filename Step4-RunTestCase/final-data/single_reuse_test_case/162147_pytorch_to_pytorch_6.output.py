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
        
        # Adaptation: Replace the original topk/index_select logic with torch.logcumsumexp
        # to test the similar API. 
        # Original: i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        # New: Calculate cumulative log-sum-exp of the logits.
        l1 = torch.logcumsumexp(z1, dim=1)
        l2 = torch.logcumsumexp(z2, dim=1)
        l3 = torch.logcumsumexp(z3, dim=1)
        
        return l1, l2, l3

    def _apply_mixture(self, x_flat, l1, l2, l3):
        """
        Adapted to accept the logcumsumexp results instead of indices/weights.
        For the purpose of testing the API, we simply concatenate the results.
        """
        # In a real scenario, this would use the indices to select weights.
        # Here we just return the processed features to verify the API integration.
        return torch.cat([l1, l2, l3], dim=1)

    def forward(self, x):
        x_addr = self.P(x)
        l1, l2, l3 = self._address(x_addr)
        out = self._apply_mixture(x, l1, l2, l3)
        return out

def test_logcumsumexp_fast_learned_cell():
    # Setup parameters
    D_in, H, D_out = 10, 20, 5
    batch_size = 4
    
    # Instantiate the module
    model = FastLearnedCellX3(D_in, H, D_out)
    
    # Compile the model to test compatibility with torch.compile (as per bug report context)
    # Note: torch.compile is available in PyTorch 2.0+
    try:
        compiled_model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Compilation skipped (requires PyTorch 2.0+): {e}")
        compiled_model = model

    # Create dummy input
    x = torch.randn(batch_size, D_in)
    
    # Run forward pass
    output = compiled_model(x)
    
    # Assertions
    L_tot = model.L_w1 + model.L_w2 + model.L_b2
    assert output.shape == (batch_size, L_tot), f"Expected shape {(batch_size, L_tot)}, got {output.shape}"
    
    # Verify numerical stability (logcumsumexp should be stable)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    
    # Verify monotonicity property of logcumsumexp along dim 1
    # log(cumsum(exp(x))) is monotonically increasing.
    # Since the output is a concatenation of three independent sequences (l1, l2, l3),
    # we must check monotonicity within each segment, not across the boundaries.
    s1, s2, s3 = model.L_w1, model.L_w2, model.L_b2
    l1_out, l2_out, l3_out = torch.split(output, (s1, s2, s3), dim=1)
    
    for name, tensor in [("l1", l1_out), ("l2", l2_out), ("l3", l3_out)]:
        if tensor.size(1) > 1:
            diff = tensor[:, 1:] - tensor[:, :-1]
            assert (diff >= 0).all(), f"{name} output should be monotonically increasing along dim 1"
    
    print("Test passed: torch.logcumsumexp integrated successfully.")

if __name__ == "__main__":
    test_logcumsumexp_fast_learned_cell()