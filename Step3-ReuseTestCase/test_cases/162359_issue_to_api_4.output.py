import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import math

def test_sequentiallr_tensor_lr_base_lr_corruption():
    """
    Regression test for Issue #162359.
    
    Verifies that when using a Tensor learning rate, SequentialLR does not
    corrupt the `initial_lr` and `base_lrs` of chained schedulers.
    
    The bug occurred because SequentialLR.__init__ aliased the optimizer's 
    `lr` tensor with `initial_lr`. The initial step of the first scheduler
    then modified `lr` in-place, inadvertently mutating `initial_lr` and 
    the `base_lrs` of subsequent schedulers.
    """
    # 1. Setup model and optimizer with a Tensor learning rate.
    # Using a Tensor LR is the specific trigger for this bug.
    lr_value = 1.0
    model = nn.Linear(10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=torch.tensor(lr_value))

    # 2. Initialize chained schedulers.
    milestone = 10
    start_factor = 0.2
    
    # Warmup scheduler: Linearly increases LR from start_factor * lr to lr
    warmup_scheduler = LinearLR(
        optimizer, start_factor=start_factor, total_iters=milestone
    )
    
    # Main scheduler: Cosine annealing decay
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # 3. Initialize SequentialLR.
    # In the buggy version, this step would:
    # a) Alias optimizer.param_groups[0]['lr'] with optimizer.param_groups[0]['initial_lr']
    # b) Call warmup_scheduler.step(), which modifies 'lr' in-place to start_factor * lr_value
    # c) This modification corrupts 'initial_lr' and decay_scheduler.base_lrs
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, decay_scheduler], 
        milestones=[milestone]
    )

    # 4. Verify that the optimizer's initial_lr is NOT corrupted.
    # It should remain the original lr_value (1.0), not the scaled value (0.2).
    assert torch.isclose(
        optimizer.param_groups[0]['initial_lr'], 
        torch.tensor(lr_value)
    ), f"Optimizer initial_lr was corrupted to {optimizer.param_groups[0]['initial_lr'].item()}"

    # 5. Verify that the second scheduler's base_lrs are NOT corrupted.
    # They should be 1.0, ensuring the decay starts from the correct baseline.
    assert torch.isclose(
        decay_scheduler.base_lrs[0], 
        torch.tensor(lr_value)
    ), f"Decay scheduler base_lr was corrupted to {decay_scheduler.base_lrs[0].item()}"

    # 6. Verify that the current LR is correctly set by the warmup scheduler.
    expected_current_lr = lr_value * start_factor
    assert torch.isclose(
        optimizer.param_groups[0]['lr'], 
        torch.tensor(expected_current_lr)
    ), f"Current LR is {optimizer.param_groups[0]['lr'].item()}, expected {expected_current_lr}"

    # 7. Verify the schedule behavior over the transition to ensure correctness.
    # Step through the warmup phase
    for _ in range(milestone):
        scheduler.step()

    # At the end of warmup, LR should be back to 1.0
    assert torch.isclose(
        optimizer.param_groups[0]['lr'], 
        torch.tensor(lr_value)
    ), "LR did not reach 1.0 at end of warmup"

    # Step into the decay phase
    scheduler.step()
    
    # Calculate expected LR for CosineAnnealing at step 1 with base_lr=1.0
    # Formula: eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_max)) / 2
    # Here: eta_min=0, base_lr=1.0, t=1, T_max=10
    expected_decay_lr = lr_value * (1 + math.cos(math.pi / 10)) / 2
    
    # If base_lrs were corrupted (0.2), the LR would be approx 0.19.
    # We check it is close to the expected value derived from 1.0.
    assert torch.isclose(
        optimizer.param_groups[0]['lr'], 
        torch.tensor(expected_decay_lr), 
        atol=1e-5
    ), f"LR in decay phase is {optimizer.param_groups[0]['lr'].item()}, expected {expected_decay_lr}"

if __name__ == "__main__":
    test_sequentiallr_tensor_lr_base_lr_corruption()
    print("Test passed!")