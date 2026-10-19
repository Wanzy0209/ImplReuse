```python
import sys
# sys.path.append('third_party/Matcha-TTS') # Preserved for structure
import argparse
import os
import yaml # Conversion: Replaced hyperpyyaml with standard yaml
import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


# Conversion: torch.nn.Module -> tf.keras.Model
class HiFTInferenceWrapper(tf.keras.Model):
    def __init__(self, hift_module):
        super(HiFTInferenceWrapper, self).__init__()
        self.hift = hift_module

    # Conversion: forward -> call
    def call(
        self,
        speech_feat: tf.Tensor,
        cache_source: tf.Tensor,
        training=False, # Added for Keras convention
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str, device: str) -> tf.keras.Model:
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        # Conversion: Using yaml.safe_load instead of load_hyperpyyaml
        configs = yaml.safe_load(f)
        # Note: Overrides logic removed for simplicity, would require OmegaConf or similar

    # Conversion: torch.load -> tf.saved_model.load
    # In TensorFlow, we typically load a SavedModel directory rather than a .pt file.
    # We assume the model is saved as a TF SavedModel.
    hift = tf.saved_model.load(os.path.join(model_dir, "hift_saved_model"))

    # Conversion: .to(device) is implicit in TF or handled via tf.device context.
    # .eval() is not strictly required in TF inference.
    return hift


def build_example_inputs(
    device: str,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    with tf.device(device): # Conversion: Explicit device context
        # Conversion: torch.randn -> tf.random.normal
        speech_feat = tf.random.normal(
            shape=(1, 80, mel_frames),
            dtype=tf.float32,
        )
        # Conversion: torch.zeros -> tf.zeros
        cache_source = tf.zeros(
            shape=(1, 1, cache_frames),
            dtype=tf.float32,
        )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to SavedModel." # Conversion: Updated description
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift_saved_model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift_saved_model",
        help="SavedModel output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting",
    )
    parser.add_argument(
        "--opset", # Conversion: ONNX opset is not applicable to TF SavedModel
        type=int,
        default=17,
        help="Ignored in TF export (kept for arg compatibility)",
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

    # Conversion: torch.device -> TF device string
    # Conversion: torch.cuda.is_available -> tf.config.list_physical_devices
    device_str = "/GPU:0" if args.device == "cuda" and tf.config.list_physical_devices('GPU') else "/CPU:0"

    # Conversion: torch.set_grad_enabled(False) -> Not strictly needed in TF inference
    # but we can ensure we are not in training mode.

    hift = load_hift(args.model_dir, device_str)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        device=device_str,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: torch.onnx.export -> tf.saved_model.save
    # We define the signature to handle dynamic axes (equivalent to dynamic_axes in ONNX)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 80, None), dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=(None, 1, None), dtype=tf.float32, name="cache_in")
    ])
    def serving_fn(speech_feat, cache_source):
        return wrapper(speech_feat, cache_source)

    tf.saved_model.save(
        wrapper,
        args.output,
        signatures={'serving_default': serving_fn}
    )
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```