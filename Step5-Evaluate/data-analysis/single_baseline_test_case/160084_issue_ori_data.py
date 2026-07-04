# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b
   
model = RegressionModel()
torch_device="cuda"
model.forward = torch.compile(model.forward, backend="inductor")
inputs = torch.randn(4, 10).to(torch_device)
_ = model(inputs)