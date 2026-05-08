import torch
import onnx

class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)


torch.onnx.export(
    SumModule(),
    (torch.ones(2, 2),),
    "onnx.pb",
    input_names=["x"],
    output_names=["sum"],
    dynamic_axes={
        "x": {0: "my_custom_axis_name"},
        "sum": [0],
    },
)

onnx_model = onnx.load("onnx.pb")
print(onnx_model.graph.input)
print(onnx_model.graph.output)