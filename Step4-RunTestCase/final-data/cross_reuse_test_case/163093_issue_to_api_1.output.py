import torch
from torch import optim, nn

# Enable logging to capture recompilation events
# Use try-except to handle cases where torch._logging is not available
try:
    torch._logging.set_logs(recompiles=True)
except AttributeError:
    pass

def test_reduce_lr_on_plateau_tensor_lr_recompilation():
    """
    Test case to reproduce the recompilation bug in ReduceLROnPlateau
    when the optimizer uses a tensor learning rate.
    Leverages torch.cat for metric generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(1, 1).to(device)

    # Initialize optimizer with a tensor learning rate.
    # This is the specific condition that triggers the bug.
    lr = torch.tensor(0.1)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler configuration
    # patience=1 means LR reduces after 1 step of no improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=1, min_lr=0.001
    )

    # Compiled function containing the optimizer and scheduler steps
    @torch.compile(fullgraph=False)
    def fn(metric):
        opt.step()
        scheduler.step(metric)

    total_steps = 8
    
    # Use torch.cat to construct the fake metrics tensor
    # This reuses the similar API (torch.cat) as requested
    metrics_part1 = torch.linspace(1.0, 0.6, 3, device=device)
    metrics_part2 = torch.linspace(0.5, 0.0, 5, device=device)
    fake_metrics = torch.cat([metrics_part1, metrics_part2])
    
    # Create a plateau in the metrics to trigger the scheduler's reduction logic
    # Setting indices 3, 4, 5 to the same value (0.5)
    fake_metrics[3:6] = fake_metrics[3]

    print("Running steps...")
    for i, metric in enumerate(fake_metrics):
        print(f"Step {i}, Metric: {metric.item():.2f}")
        fn(metric)

    # Expected behavior (bug):
    # The logs should show a recompilation event triggered by a guard failure
    # related to the type of 'param_groups[0]['lr']' changing from Tensor to float.

if __name__ == "__main__":
    test_reduce_lr_on_plateau_tensor_lr_recompilation()