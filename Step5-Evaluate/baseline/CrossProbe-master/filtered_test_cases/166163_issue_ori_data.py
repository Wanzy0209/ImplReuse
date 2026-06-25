import torch
import torchvision

models = [torchvision.models.efficientnet_b0, torchvision.models.mobilenet_v2, torchvision.models.resnet18]
input_batch = torch.rand(2, 3, 224, 224)

for model_fn in models:
    model = model_fn(weights=None)
    try:
        onnx_program = torch.onnx.export(
            model,
            (input_batch,),
            dynamo=True
        )
        print(f'{model.__class__.__name__}: Success')
    except Exception as e:
        print(f'{model.__class__.__name__}: {type(e).__name__}: {str(e)[:100]}')