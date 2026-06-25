import torch
from torch import optim, nn

torch._logging.set_logs(recompiles=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(1, 1).to(device)

lr = torch.tensor(0.1)
opt = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, min_lr=0.001)

@torch.compile(fullgraph=False)
def fn(metric):
    opt.step()
    scheduler.step(metric)

fake_metrics = torch.linspace(1.0, 0.0, 8, device=device)
fake_metrics[3:6] = fake_metrics[3]

for metric in fake_metrics:
    fn(metric)