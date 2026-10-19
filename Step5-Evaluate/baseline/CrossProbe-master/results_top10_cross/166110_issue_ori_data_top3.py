```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import tensorflow as tf
# Assuming hyperpyyaml is available and compatible or adapted for TF model construction
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


# Conversion: torch.nn.Module -> tf.Module
class HiFTInferenceWrapper(tf.Module):
    def __init__(self, hift_module):
        super().__init__()
        self.hift = hift_module

    # Conversion: forward -> __call__ (standard TF method for inference)
    # Added @tf.function to ensure the model is exported as a graph
    @tf.function
    def __call__(
        self,
        speech_feat,  # Conversion: torch.Tensor -> tf.Tensor (implied)
        cache_source,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str): # Conversion: Removed device arg, TF handles placement via context
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
                "llm": None,
            },
        )
    # Assuming configs['hift'] returns a tf.keras.Model or tf.Module
    hift = configs["hift"]
    
    # Conversion: torch.load -> tf.keras.models.load_model or load_weights
    # Note: PyTorch .pt files cannot be loaded directly into TensorFlow.
    # This assumes the weights have been converted to a TensorFlow checkpoint format.
    checkpoint_path = os.path.join(model_dir, "hift.ckpt")
    if isinstance(hift, tf.keras.Model):
        # Use expect_partial() if the checkpoint contains extra keys (e.g., optimizer state)
        hift.load_weights(checkpoint_path).expect_partial()
    else:
        # Fallback for generic tf.Module saving/loading
        checkpoint = tf.train.Checkpoint(module=hift)
        checkpoint.restore(checkpoint_path).expect_partial()
        
    # Conversion: hift.to(device).eval() -> Not strictly required in TF.
    # Device placement is handled by tf.device context. 
    # Training/inference mode is handled by layer behavior (e.g. Dropout).
    return hift


def build_example_inputs(
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: device argument removed, handled by context
    speech_feat = tf.random.normal(
        [1, 80, mel_frames], # Shape as list/tuple
        dtype=tf.float32,
    )
    # Conversion: torch.zeros -> tf.zeros
    cache_source = tf.zeros(
        [1, 1, cache_frames], # Shape as list/tuple
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
        help="Directory that contains cosyvoice2.yaml and hift.ckpt",
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
    # Conversion: ONNX opset version is not applicable to TF SavedModel export
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

    # Conversion: torch.device & torch.cuda.is_available
    gpus = tf.config.list_physical_devices('GPU')
    device_name = "/GPU:0" if gpus and args.device == "cuda" else "/CPU:0"

    # Conversion: torch.set_grad_enabled(False)
    # In TensorFlow, gradients are only calculated inside tf.GradientTape.
    # For inference/export, we simply avoid using GradientTape.
    
    with tf.device(device_name):
        hift = load_hift(args.model_dir)
        wrapper = HiFTInferenceWrapper(hift)

        example_inputs = build_example_inputs(
            mel_frames=args.mel_frames,
            cache_frames=args.cache_frames,
        )

        # Conversion: torch.onnx.export -> tf.saved_model.save
        # To support dynamic axes (variable batch/sequence length), we define 
        # the signature with None dimensions.
        
        # Define input signatures corresponding to dynamic_axes in the original code
        # mel: {0: "batch", 2: "mel_frames"} -> shape [None, 80, None]
        # cache_in: {0: "batch", 2: "cache_frames"} -> shape [None, 1, None]
        input_signatures = [
            tf.TensorSpec(shape=[None, 80, None], dtype=tf.float32, name="mel"),
            tf.TensorSpec(shape=[None, 1, None], dtype=tf.float32, name="cache_in")
        ]

        # We wrap the call to get a concrete function with the specific signature
        # This mimics the tracing behavior of torch.onnx.export
        @tf.function(input_signature=input_signatures)
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