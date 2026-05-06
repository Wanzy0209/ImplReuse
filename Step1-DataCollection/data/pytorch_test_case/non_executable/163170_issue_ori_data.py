# export script
import torch
from diffusers import CosmosTransformer3DModel

device = "cuda"
dtype = torch.bfloat16
model = CosmosTransformer3DModel.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Video2World",
    subfolder="transformer",
    use_safetensors=True,
    token=<hf-token>,
    torch_dtype=dtype,
).to(device)

batch_size = 1
image_height = 704
image_width = 1280
latent_height = image_height // 8
latent_width = image_width // 8
latent_frames = 1#24
latent_channels = model.config.in_channels - 1
text_maxlen = 512
sample_input = (
    torch.randn(
        batch_size,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
        dtype=dtype,
        device=device,
    ),
    torch.tensor([1.0] * batch_size, dtype=dtype, device=device),
    torch.randn(
        batch_size, text_maxlen, model.config["text_embed_dim"], dtype=dtype, device=device
    ),
    {
        "fps": torch.tensor([16] * batch_size, dtype=dtype, device=device),
        "padding_mask": torch.ones(1, 1, image_height, image_width, dtype=dtype, device=device),
        "condition_mask": torch.randn(
            batch_size,
            1,
            latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
            device=device,
        )
    },
)

input_names = [
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "padding_mask",
    "fps",
    "condition_mask",
]

output_names = ["latent"]

dynamic_shapes = (
    {0: "B", 2: "latent_frames", 3: "latent_H", 4: "latent_W"},
    {0: "B"},
    {0: "B"},
    {0: "B", 2: "H", 3: "W"},
    {0: "B"},
    {0: "B", 2: "latent_frames", 3: "latent_H", 4: "latent_W"},
)

# run inference for sanity check
out = model(sample_input[0], sample_input[1], sample_input[2], **sample_input[3])

torch.onnx.export(
    model,
    sample_input,
    "cosmos_transformer_vid2world/model.onnx",
    export_params=True,
    opset_version=19,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    verbose=False,
    dynamo=True,
)