```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import sys
import tensorflow as tf
import yaml  # Conversion: Replaced hyperpyyaml with standard yaml for TF context

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


# Conversion: Mocking the HiFT model class as the architecture definition is not provided.
# In a real scenario, this would be the actual TensorFlow/Keras model implementation.
class HiFT(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        # Initialize layers based on config here
        self.config = config

    def inference(self, speech_feat, cache_source):
        # Conversion: Implementing the forward pass logic.
        # This is a placeholder to ensure code validity.
        # In the original PyTorch code, this calls self.hift.inference.
        return speech_feat, cache_source


class HiFTInferenceWrapper(tf.keras.Model):
    def __init__(self, hift_module: tf.keras.Model):
        super().__init__()
        self.hift = hift_module

    # Conversion: @tf.function is used for graph optimization and export
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


def load_hift(model_dir: str) -> tf.keras.Model:
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        # Conversion: Using standard yaml loading. 
        # Overrides logic would be handled here if necessary.
        configs = yaml.safe_load(f)
    
    # Conversion: Instantiating the TF model instead of PyTorch module
    hift = HiFT(configs)
    
    # Conversion: Loading weights in TensorFlow.
    # Assuming weights are saved in TF format (e.g., .h5 or .ckpt).
    # The original code loaded a .pt file and manipulated state dict keys.
    checkpoint_path = os.path.join(model_dir, "hift.h5")
    
    if os.path.exists(checkpoint_path):
        hift.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization.")

    return hift


def build_example_inputs(
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    # Conversion: tf.random.normal instead of torch.randn
    # Conversion: Removed explicit device argument, TF handles placement automatically
    speech_feat = tf.random.normal(
        shape=[1, 80, mel_frames],
        dtype=tf.float32,
    )
    # Conversion: tf.zeros instead of torch.zeros
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
        help="Directory that contains cosyvoice2.yaml and model weights",
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
        help="Device for exporting (TF handles this via env vars or context)",
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

    # Conversion: TF handles device placement via context managers or environment variables.
    # We check for GPU availability to inform the user, but explicit device setting 
    # like torch.device is not typically used for model definition in TF.
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == "cuda" and not gpus:
        print("Warning: CUDA requested but no GPUs found.")

    # Conversion: No equivalent to torch.set_grad_enabled(False) needed for inference/export in TF
    # unless we are inside a GradientTape context.

    hift = load_hift(args.model_dir)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: TensorFlow SavedModel export replaces torch.onnx.export
    # Dynamic axes are handled automatically by TF's graph tracing.
    tf.saved_model.save(
        wrapper,
        args.output,
        signatures=wrapper.call.get_concrete_function(
            tf.TensorSpec(shape=[None, 80, None], dtype=tf.float32, name="mel"),
            tf.TensorSpec(shape=[None, 1, None], dtype=tf.float32, name="cache_in"),
        )
    )
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```