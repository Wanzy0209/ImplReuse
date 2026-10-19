```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import sys
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


class HiFTInferenceWrapper(tf.keras.Model):  # Conversion: torch.nn.Module -> tf.keras.Model
    def __init__(self, hift_module):  # Conversion: Remove type hint for TF module
        super().__init__()
        self.hift = hift_module

    def call(  # Conversion: forward -> call
        self,
        speech_feat,  # Conversion: Remove type hint
        cache_source,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str, device: str):  # Conversion: torch.device -> str
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
                "llm": None,
            },
        )
    
    # Conversion: In TensorFlow, we typically load a SavedModel directly.
    # The original code instantiates a model from config and loads state_dict.
    # Here we assume the model is saved as a TF SavedModel in model_dir.
    # hift = configs["hift"] 
    
    # Conversion: torch.load -> tf.saved_model.load
    # Note: TF loads from a directory, not a single .pt file.
    hift = tf.saved_model.load(model_dir)
    
    # Conversion: State dict manipulation is specific to PyTorch checkpoints.
    # TF models loaded via saved_model.load already have weights assigned.
    # state_dict = {
    #     k.replace("generator.", ""): v for k, v in state_dict.items()
    # }
    # hift.load_state_dict(state_dict, strict=True)
    
    # Conversion: .to(device).eval() -> TF handles device placement implicitly or via strategy.
    # .eval() is not needed in TF as inference mode is default unless training=True.
    return hift


def build_example_inputs(
    device: str,  # Conversion: torch.device -> str
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    
    # Conversion: Explicit device placement context
    with tf.device(device):
        # Conversion: torch.randn -> tf.random.normal
        speech_feat = tf.random.normal(
            [1, 80, mel_frames],  # Conversion: Shape as list
            dtype=tf.float32,
        )
        # Conversion: torch.zeros -> tf.zeros
        cache_source = tf.zeros(
            [1, 1, cache_frames],
            dtype=tf.float32,
        )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to SavedModel."  # Conversion: Updated description
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains the SavedModel",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift_savedmodel",
        help="SavedModel output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting",
    )
    # Conversion: ONNX opset version is not applicable to TF SavedModel
    # parser.add_argument("--opset", ...)
    
    parser.add_argument(
        "--mel-frames",
        type=int,
        default=120,
        help="Example mel length (frames)",
    )
    parser.add_argument(
        "--cache-frames",
        type=int,
        default=3840,
        help="Example cache source length (samples)",
    )
    args = parser.parse_args()

    # Conversion: torch.cuda.is_available -> tf.config.list_physical_devices
    gpus = tf.config.list_physical_devices('GPU')
    device_str = '/GPU:0' if (args.device == "cuda" and gpus) else '/CPU:0'
    
    # Conversion: torch.set_grad_enabled(False) -> Not strictly required in TF inference script
    # unless using GradientTape explicitly.

    hift = load_hift(args.model_dir, device_str)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        device=device_str,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: Define input signatures for SavedModel to mimic input_names and dynamic_axes
    # dynamic_axes: "mel": {0: "batch", 2: "mel_frames"} -> shape [None, 80, None]
    # dynamic_axes: "cache_in": {0: "batch", 2: "cache_frames"} -> shape [None, 1, None]
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 80, None], dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=[None, 1, None], dtype=tf.float32, name="cache_in")
    ])
    def serving_fn(mel, cache_in):
        return wrapper(mel, cache_in)

    # Conversion: torch.onnx.export -> tf.saved_model.save
    tf.saved_model.save(
        wrapper,
        args.output,
        signatures={'serving_default': serving_fn}
    )
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```