import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BaseModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X, batch_size=1):
        data_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        self.eval()
        n_samples = X.size(0)
        y_pred = torch.zeros((n_samples,) + self(X[0:1]).size()[1:])
        r = 0
        for batch_data in data_loader:
            X_batch = torch.autograd.Variable(batch_data[0])
            y_batch_pred = self(X_batch).data
            y_pred[r:r + len(y_batch_pred)] = y_batch_pred
            r += len(y_batch_pred)
        return y_pred

class CacheModule(BaseModule):
    def __init__(self, cache: torch.Tensor):
        super().__init__()
        assert cache.ndim == 3
        self.cache = nn.Parameter(cache, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.size(1)
        rolled_cache = torch.roll(self.cache.data, -n_tokens, dims=1)
        rolled_cache[:, -n_tokens:, :] = x
        self.cache.data = rolled_cache
        return self.cache

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x) if self.activation else x

class MyModel(BaseModule):
    INPUT_FEATURES = 5
    CACHE_SEQ_LEN = 10

    def __init__(self):
        super().__init__()
        default_cache = torch.zeros(1, self.CACHE_SEQ_LEN, self.INPUT_FEATURES)
        self.cache_layer = CacheModule(default_cache)
        self.fc1 = LinearBlock(self.INPUT_FEATURES, 10, activation=nn.ReLU())
        self.fc2 = LinearBlock(10, self.INPUT_FEATURES)

    def forward(self, x):
        cached = self.cache_layer(x)
        out = self.fc1(cached)
        out = self.fc2(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, MyModel.INPUT_FEATURES)

if __name__ == "__main__":
    input_tensor = GetInput()
    exported_program = torch.export.export(my_model_function(), (input_tensor,))
    exported_model = exported_program.module()

    compiled_model = torch.compile(exported_model, mode="max-autotune-no-cudagraphs")
    compiled_output = compiled_model(input_tensor)