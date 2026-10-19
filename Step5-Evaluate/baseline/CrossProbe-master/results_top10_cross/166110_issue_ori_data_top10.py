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


class HiFTInferenceWrapper(tf.keras.Model):
    def __init__(self, hift_module: tf.keras.Model):
        super().__init__()
        self.hift = hift_module

    # Conversion: PyTorch forward() is equivalent to TensorFlow call()
    @tf.function
    def call(
        self,
        speech_feat: tf.Tensor,
        cache_source: tf.Tensor,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str, device: str) -> tf.keras.Model:
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
    
    # Conversion: torch.load is replaced by tf.keras.models.load_model or load_weights
    # Assuming the model architecture is defined in configs and weights are loaded separately.
    # Note: State dict key manipulation (replace "generator.") is specific to PyTorch.
    # In TensorFlow, we typically load weights directly into the model structure.
    checkpoint_path = os.path.join(model_dir, "hift")
    if os.path.exists(checkpoint_path + ".index"):
        hift.load_weights(checkpoint_path)
    else:
        # Fallback if loading a full SavedModel instead of just weights
        try:
            hift = tf.keras.models.load_model(checkpoint_path)
        except:
            print(f"Warning: Could not load weights from {checkpoint_path}")
            
    # Conversion: .to(device) is handled implicitly or via tf.device context in TF
    # .eval() is not strictly required in TF 2.x for inference, but we ensure training=False if needed
    if hasattr(hift, 'training'):
        hift.training = False
        
    return hift


def build_example_inputs(
    device: str,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    
    # Conversion: Use tf.device context to place tensors on specific device
    with tf.device(device):
        # Conversion: torch.randn -> tf.random.normal
        speech_feat = tf.random.normal(
            shape=[1, 80, mel_frames],
            dtype=tf.float32,
        )
        # Conversion: torch.zeros -> tf.zeros
        cache_source = tf.zeros(
            shape=[1, 1, cache_frames],
            dtype=tf.float32,
        )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to TensorFlow SavedModel."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift_savedmodel",
        help="TensorFlow SavedModel output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting",
    )
    # Conversion: ONNX opset version is not applicable to TF SavedModel export
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (Unused in TF export)",
    )
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

    # Conversion: Handle device selection for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == "cuda" and gpus:
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    # Conversion: torch.set_grad_enabled(False) is not a global switch in TF 2.x.
    # Gradients are only calculated within tf.GradientTape.

    hift = load_hift(args.model_dir, device)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        device=device,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: Define input signatures for the SavedModel
    # This replaces input_names and dynamic_axes from ONNX export
    input_signature = [
        tf.TensorSpec(shape=[None, 80, None], dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=[None, 1, None], dtype=tf.float32, name="cache_in"),
    ]
    
    # Conversion: torch.onnx.export is replaced by tf.saved_model.save
    # We use the wrapper's call method as the signature
    tf.saved_model.save(
        wrapper,
        args.output,
        signatures=wrapper.call.get_concrete_function(*input_signature)
    )
    print(f"HiFT TensorFlow SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```