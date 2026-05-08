import onnxruntime as ort
import torch
import torch.nn as nn


class Net(nn.Module):
    def forward(self, x, y):
        return torch.atan2(x, y)


net = Net()
x = torch.tensor([0.0])
y = torch.tensor([0.0])
torch.onnx.export(
    net, (x, y), "atan2.onnx", input_names=["x", "y"], output_names=["output"]
)

sess = ort.InferenceSession("atan2.onnx")

torch_result = net(x, y)
ort_result = sess.run(["output"], {"x": x.numpy(), "y": y.numpy()})[0]

print("torch_result", torch_result) # tensor([0.])
print("ort_result", ort_result) # [nan]