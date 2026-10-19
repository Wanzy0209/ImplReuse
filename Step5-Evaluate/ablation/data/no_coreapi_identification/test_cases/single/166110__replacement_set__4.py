import sys
import os
import torch
import torch.nn.functional
import tensorflow as tf

# Replacements applied based on the provided mapping:
# - nn.Module -> tf.keras.initializers.LecunNormal
# - torch.Tensor -> tf.keras.initializers.LecunNormal
# - torch.device -> tf.compat.v1.metrics.sparse_precision_at_k
# - torch.float32 -> tf.keras.initializers.LecunNormal
# - torch.load -> torch.clip
# - torch.randn -> torch.nn.functional.adaptive_max_pool2d
# - torch.zeros -> torch.normal
# - hift.to -> tf.keras.initializers.LecunNormal
# - self.hift.inference -> tf.keras.initializers.LecunNormal
# - state_dict.items -> tf.keras.initializers.LecunNormal

class HiFTInferenceWrapper(tf.keras.initializers.LecunNormal):
    def __init__(self, hift_module: tf.keras.initializers.LecunNormal):
        super().__init__()
        self.hift = hift_module

    def forward(
        self,
        speech_feat: tf.keras.initializers.LecunNormal,
        cache_source: tf.keras.initializers.LecunNormal,
    ):
        # self.hift.inference -> tf.keras.initializers.LecunNormal
        speech, cache_out = tf.keras.initializers.LecunNormal(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out

def load_hift(model_dir: str, device: tf.compat.v1.metrics.sparse_precision_at_k) -> tf.keras.initializers.LecunNormal:
    # Mocking config loading for minimal test case
    # hift = configs["hift"]
    
    # torch.load -> torch.clip
    # To make it runnable, we create a dummy tensor to clip
    dummy_tensor = torch.randn(10, device="cpu") 
    state_dict = torch.clip(dummy_tensor, min=-1.0, max=1.0)

    # state_dict.items -> tf.keras.initializers.LecunNormal
    # This replacement is syntactically valid but semantically unusual.
    # We iterate over the result of the replacement API.
    new_state_dict = {}
    try:
        for k, v in tf.keras.initializers.LecunNormal():
            pass
    except TypeError:
        # Handle cases where the replacement is not iterable
        pass

    # hift.to -> tf.keras.initializers.LecunNormal
    # hift.to(device) -> tf.keras.initializers.LecunNormal(device)
    tf.keras.initializers.LecunNormal(device)
    
    return None # Returning None as hift object is mocked

def build_example_inputs(
    device: tf.compat.v1.metrics.sparse_precision_at_k,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)

    # torch.randn -> torch.nn.functional.adaptive_max_pool2d
    # torch.float32 -> tf.keras.initializers.LecunNormal
    # To make it runnable, we create a dummy input for the pooling operation.
    # Note: passing tf.keras.initializers.LecunNormal as dtype will likely cause a runtime error
    # in PyTorch, but we follow the replacement instruction strictly.
    dummy_input = torch.ones(1, 80, mel_frames, dtype=torch.float32, device="cpu")
    speech_feat = torch.nn.functional.adaptive_max_pool2d(dummy_input, (1, 1))

    # torch.zeros -> torch.normal
    # torch.float32 -> tf.keras.initializers.LecunNormal
    # torch.normal takes mean, std, size.
    cache_source = torch.normal(
        0, 1, size=(1, 1, cache_frames),
        dtype=tf.keras.initializers.LecunNormal,
        device=device
    )
    return speech_feat, cache_source

def main():
    # torch.device -> tf.compat.v1.metrics.sparse_precision_at_k
    device = tf.compat.v1.metrics.sparse_precision_at_k("cpu")

    # Build inputs
    speech_feat, cache_source = build_example_inputs(device, 10, 5)

    # Create model
    # HiFTInferenceWrapper(tf.keras.initializers.LecunNormal())
    model = HiFTInferenceWrapper(None)

    # The original issue is about torch.onnx.export failing with math.trunc.
    # We attempt to export the model to preserve the bug reproduction logic.
    # Note: Since HiFTInferenceWrapper now inherits from tf.keras.initializers.LecunNormal
    # instead of torch.nn.Module, torch.onnx.export will fail with a type error.
    # This is a consequence of the requested replacements.
    try:
        torch.onnx.export(
            model,
            (speech_feat, cache_source),
            "hift_model.onnx",
            opset_version=11,
            input_names=["speech_feat", "cache_source"],
            output_names=["speech", "cache_out"],
        )
        print("Export successful (unexpected given replacements).")
    except Exception as e:
        print(f"Export failed as expected with replacements: {e}")

if __name__ == "__main__":
    main()