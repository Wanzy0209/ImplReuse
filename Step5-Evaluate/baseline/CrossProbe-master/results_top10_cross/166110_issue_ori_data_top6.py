```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


class HiFTInferenceWrapper(tf.Module):  # Conversion: torch.nn.Module -> tf.Module
    def __init__(self, hift_module):
        super().__init__()
        self.hift = hift_module

    @tf.function  # Conversion: forward -> __call__ with tf.function decorator for graph mode
    def __call__(
        self,
        speech_feat,  # Conversion: torch.Tensor -> tf.Tensor implied
        cache_source,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str):  # Conversion: Removed device argument
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
                "llm": None,
            },
        )
    hift = configs["hift"]
    
    # Conversion: torch.load -> hift.load_weights
    # Note: Assumes weights are converted to TensorFlow format (e.g., .h5 or .ckpt).
    # The key renaming logic ("generator." prefix removal) is assumed to be handled
    # during the weight conversion process or handled by the model's load_weights method.
    hift.load_weights(os.path.join(model_dir, "hift.h5"))
    
    # Conversion: .to(device).eval() not required in TensorFlow
    return hift


def build_example_inputs(  # Conversion: Removed device argument
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    # Conversion: torch.randn -> tf.random.normal
    speech_feat = tf.random.normal(
        [1, 80, mel_frames],  # Conversion: Shape syntax
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
        description="Export CosyVoice2 HiFT vocoder to SavedModel."  # Conversion: ONNX -> SavedModel
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift.h5",  # Conversion: .pt -> .h5
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift",  # Conversion: .onnx -> directory path
        help="SavedModel output path",
    )
    # Conversion: Removed --device and --opset arguments as they are specific to PyTorch/ONNX
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

    # Conversion: Removed torch.device and torch.set_grad_enabled
    # TensorFlow handles device placement automatically and gradients are not tracked by default during inference.

    hift = load_hift(args.model_dir)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: Removed input_names, output_names, and dynamic_axes.
    # TensorFlow SavedModel handles dynamic shapes automatically and infers input/output names from signatures.

    # Conversion: torch.onnx.export -> tf.saved_model.save
    # We call the wrapper once to ensure the graph is traced before saving, or rely on tf.function.
    _ = wrapper(*example_inputs)
    tf.saved_model.save(wrapper, args.output)
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```