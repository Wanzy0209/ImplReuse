# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
class PoseWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output["pose"]

model = PoseWrapper(original_model)

input = torch.ones(1, 1, 512, 512)
torch.onnx.export(
    model,
    input,
    output_path,
    export_params=True,
    input_names=["input"],
    output_names=["pose"],
    dynamic_shapes=({0: "batch"},),
    dynamo=True,
)