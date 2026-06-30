import torch
import torch.nn as nn
from torch import optim

def test_sequential_lr_tensor_aliasing_with_linear():
    """
    Verifies that the SequentialLR bug (aliasing between 'lr' and 'initial_lr')
    is fixed. The bug caused 'base_lrs' to be corrupted for chained schedulers.
    
    This test leverages torch.nn.Linear (the similar API) as the model
    providing parameters to the optimizer.
    """
    # 1. Use a Linear model (leveraging the similar API torch.nn.Linear)
    # instead of a single tensor to reflect standard usage patterns.
    model = nn.Linear(10, 10)
    
    # Use a tensor learning rate to trigger the aliasing bug (if it existed).
    lr = 1.0
    opt = optim.AdamW(model.parameters(), lr=torch.tensor(lr))

    # 2. Initialize our chained schedulers.
    milestone, total_steps = 40, 100
    start_factor, end_factor = 0.2, 1.0
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor, end_factor, total_iters=milestone)
    decay = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps-milestone)

    # Note that each scheduler's base_lr aliases with the optimizer's initial_lr.
    # This is expected behavior: schedulers reference the optimizer's initial_lr.
    assert warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()
    assert decay.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

    # 3. Initialize our SequentialLR.
    # SequentialLR.__init__ should NOT alias our optimizer's lr and initial_lr.
    # It also calls _initial_step() on our LinearLR, setting lr to start_factor * initial_lr.
    scheduler = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, decay], milestones=[milestone])

    # 4. Verify the fix (No aliasing and no corruption)
    
    # SequentialLR.__init__ should NOT alias our optimizer's lr and initial_lr.
    assert opt.param_groups[0]['lr'].data_ptr() != opt.param_groups[0]['initial_lr'].data_ptr(), \
        "Bug detected: lr and initial_lr are aliased!"

    # base_lrs should alias with initial_lr, but lr should be distinct.
    assert (warmup.base_lrs[0].data_ptr() 
            == decay.base_lrs[0].data_ptr()
            == opt.param_groups[0]['initial_lr'].data_ptr())
    
    assert opt.param_groups[0]['lr'].data_ptr() != opt.param_groups[0]['initial_lr'].data_ptr()

    # _initial_step() on LinearLR sets lr to start_factor * initial_lr.
    # initial_lr should remain unchanged.
    expected_lr_value = start_factor * lr
    assert expected_lr_value == opt.param_groups[0]['lr'].item()
    assert lr == opt.param_groups[0]['initial_lr'].item(), \
        f"Bug detected: initial_lr corrupted to {opt.param_groups[0]['initial_lr'].item()}"
    assert lr == warmup.base_lrs[0].item()
    assert lr == decay.base_lrs[0].item()

    # 5. Verify the impact on the schedule
    # Step through the warmup phase and into the decay phase.
    for _ in range(milestone + 1):
        scheduler.step()

    # The CosineAnnealingLR should start decaying from the original lr (1.0).
    # With the fix, it should correctly use the uncorrupted base.
    current_lr = scheduler.get_last_lr()[0].item()
    
    # We assert the correct behavior: the LR is close to the original lr (1.0),
    # not the corrupted value (0.2).
    assert abs(current_lr - 1.0) < 0.01, \
        f"Bug detected: LR is {current_lr}, expected ~1.0 (uncorrupted)"

if __name__ == "__main__":
    test_sequential_lr_tensor_aliasing_with_linear()