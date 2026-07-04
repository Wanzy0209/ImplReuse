# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

BATCH_SIZE = 64
CHANNELS, IMG_SIZE = 3, 224
GRID_SIZE = 13

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = torch.nn.Parameter(torch.randn((5000, GRID_SIZE, GRID_SIZE, 2), device='cuda'))
        self.fc = torch.nn.Linear(GRID_SIZE*GRID_SIZE*CHANNELS, 16)

    def forward(self, x):
        per_channel = []
        for i in range(CHANNELS):
            channel = x[:,i,...].expand(5000,-1,-1,-1)
            patch = torch.nn.functional.grid_sample(channel, self.grid, mode="bilinear", align_corners=False, padding_mode="border")
            patch = patch.transpose(0,1).flatten(start_dim=2)
            per_channel.append(patch)
        x = torch.cat(per_channel, axis=2)
        x = self.fc(x)
        return x

def main():
    model = MyModel()
    model = model.cuda()
    model = torch.compile(model)

    for _ in range(10):
        x = torch.randn((BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE), device='cuda')
        x = model(x)
        x = torch.sum(x)
        x.backward()


if __name__ == "__main__":
    main()