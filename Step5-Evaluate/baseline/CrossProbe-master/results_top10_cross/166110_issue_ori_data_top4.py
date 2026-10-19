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


# Conversion: torch.nn.Module -> tf.keras.Model
class HiFTInferenceWrapper(tf.keras.Model):
    def __init__(self, hift_module):
        super().__init__()
        self.hift = hift_module

    # Conversion: forward -> call
    @tf.function
    def call(
        self,
        speech_feat,
        cache_source,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


# Conversion: Removed device argument as TF handles placement via context or config
def load_hift(model_dir: str):
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
    
    # Conversion: torch.load + load_state_dict -> tf.train.Checkpoint
    # Assuming the weights are stored as a TF checkpoint.
    checkpoint = tf.train.Checkpoint(model=hift)
    checkpoint.restore(os.path.join(model_dir, "hift.ckpt")).expect_partial()
    return hift


def build_example_inputs(
    device: str,
    mel_frames: int,
    cache_frames: int,
):
    cache_frames = max(cache_frames, 1)
    # Conversion: Explicit device placement using tf.device context
    with tf.device(device):
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
        description="Export CosyVoice2 HiFT vocoder to SavedModel." # Conversion: ONNX -> SavedModel
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift.ckpt", # Conversion: .pt -> .ckpt
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
    # Conversion: Removed --opset as TF SavedModel does not use ONNX opsets
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

    # Conversion: torch.device + is_available -> tf.config.list_physical_devices
    gpus = tf.config.list_physical_devices('GPU')
    device = "/GPU:0" if args.device == "cuda" and gpus else "/CPU:0"

    # Conversion: torch.set_grad_enabled(False) -> Not strictly needed in TF eager inference 
    # unless using GradientTape, but good practice to ensure no training ops are active.
    
    hift = load_hift(args.model_dir)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        device=device,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: torch.onnx.export -> tf.saved_model.save
    # TF SavedModel handles dynamic axes automatically.
    # We use the wrapper's call signature.
    tf.saved_model.save(
        wrapper,
        args.output,
        signatures=wrapper.call.get_concrete_function(
            tf.TensorSpec(shape=(None, 80, None), dtype=tf.float32, name="mel"),
            tf.TensorSpec(shape=(None, 1, None), dtype=tf.float32, name="cache_in")
        )
    )
    print(f"HiFT SavedModel exported to {args.output}")


if __name__ == "__main__":
    main()
```