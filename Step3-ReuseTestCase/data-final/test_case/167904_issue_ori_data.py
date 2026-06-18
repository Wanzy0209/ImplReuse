import torch

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        test_shapes = [torch.tensor([-1])]
        results = []
        for shape in test_shapes:
            reshaped = x.view(*shape.tolist())
            results.append(reshaped)
        return tuple(results)

model = TestModel()
input_tensor = torch.tensor([1,2,3,4,5,6,7,8])

torch.onnx.export(
    model,
    input_tensor,
    "test.onnx",
    opset_version=18,
    dynamo=True,
    report=True
)