```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import tensorflow as tf
import yaml  # Conversion: Replaced hyperpyyaml with standard yaml for compatibility

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

    # Conversion: PyTorch 'forward' method -> TensorFlow 'call' method
    def call(
        self,
        speech_feat: tf.Tensor,
        cache_source: tf.Tensor,
    ):
        speech, cache_out = self.hift(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str, device: str) -> tf.keras.Model:
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        # Conversion: load_hyperpyyaml -> yaml.safe_load
        # Note: Overrides are handled manually as standard yaml doesn't support them natively
        configs = yaml.safe_load(f)
        if "qwen_pretrain_path" in configs:
            configs["qwen_pretrain_path"] = os.path.join(model_dir, "CosyVoice-BlankEN")
        configs["llm"] = None
    
    # Assuming configs["hift"] returns a constructed Keras model instance
    hift = configs["hift"]
    
    # Conversion: torch.load -> tf.keras.models.load_model or load_weights
    # Note: PyTorch loads .pt files. TensorFlow uses .h5, .pb, or checkpoints.
    # We assume the weights are available in a TF-compatible format (e.g., hift.h5).
    weight_path = os.path.join(model_dir, "hift.h5")
    
    # Conversion: State dict manipulation is specific to PyTorch.
    # In TF, we load weights directly into the model structure.
    if os.path.exists(weight_path):
        hift.load_weights(weight_path)
    else:
        print(f"Warning: Weights file not found at {weight_path}. Model initialized randomly.")
    
    # Conversion: .to(device).eval() -> TF handles device placement via context.
    # Inference mode is handled by training=False in the call method.
    return hift


def build_example_inputs(
    device: str,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    
    # Conversion: Use tf.device context for device placement
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
        description="Export CosyVoice2 HiFT vocoder to SavedModel (TF equivalent of ONNX export)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift.h5",
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
    # Conversion: ONNX opset version is specific to ONNX, not applicable to TF SavedModel export
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

    # Conversion: torch.device and torch.cuda.is_available
    gpus = tf.config.list_physical_devices('GPU')
    device_name = "/GPU:0" if gpus and args.device == "cuda" else "/CPU:0"

    # Conversion: torch.set_grad_enabled(False) -> Not needed in TF inference graph

    hift = load_hift(args.model_dir, device_name)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        device=device_name,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: torch.onnx.export -> tf.saved_model.save
    # We define a concrete function to serve as the signature, handling dynamic axes via None in shape
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 80, None], dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=[None, 1, None], dtype=tf.float32, name="cache_in")
    ])
    def serving_fn(speech_feat, cache_source):
        return wrapper(speech_feat=speech_feat, cache_source=cache_source)

    # Call once to ensure variables are initialized
    wrapper(*example_inputs)

    tf.saved_model.save(
        wrapper,
        args.output,
        signatures={'serving_default': serving_fn}
    )
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```