```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import tensorflow as tf
import tf2onnx
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
    
    # Conversion: torch.load is PyTorch specific. 
    # In TensorFlow, we assume the model is saved as a checkpoint or SavedModel.
    # Here we simulate loading weights, assuming a converted checkpoint exists.
    # Original: state_dict = torch.load(os.path.join(model_dir, "hift.pt"), map_location=device)
    # Original: state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}
    # Original: hift.load_state_dict(state_dict, strict=True)
    
    checkpoint_path = os.path.join(model_dir, "hift_checkpoint")
    if os.path.exists(checkpoint_path):
        hift.load_weights(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Model initialized with random weights.")

    # Conversion: .to(device) and .eval() are handled by TF context and training flag.
    # We ensure the model is built by calling it once or building it based on input shape if necessary.
    return hift


def build_example_inputs(
    device: str,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    
    # Conversion: Use tf.device context for placement
    with tf.device(device):
        # Conversion: torch.randn -> tf.random.normal
        speech_feat = tf.random.normal(
            (1, 80, mel_frames),
            dtype=tf.float32,
        )
        # Conversion: torch.zeros -> tf.zeros
        cache_source = tf.zeros(
            (1, 1, cache_frames),
            dtype=tf.float32,
        )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to ONNX."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift.onnx",
        help="ONNX output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
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

    # Conversion: Check for GPU availability and set device string
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus and args.device == "cuda" else '/CPU:0'

    # Conversion: torch.set_grad_enabled(False) is not needed in TF for inference/export
    # unless explicitly using GradientTape. tf2onnx handles the graph in inference mode.

    hift = load_hift(args.model_dir, device)
    wrapper = HiFTInferenceWrapper(hift)

    # Build the model by calling it once to ensure shapes are defined
    example_inputs = build_example_inputs(
        device=device,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )
    _ = wrapper(*example_inputs)

    # Conversion: Define input signatures for tf2onnx to handle dynamic axes
    # dynamic_axes in PyTorch maps to None dimensions in tf.TensorSpec
    input_signature = [
        tf.TensorSpec(shape=(None, 80, None), dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=(None, 1, None), dtype=tf.float32, name="cache_in"),
    ]

    output_names = ["speech", "cache_out"]
    
    # Conversion: torch.onnx.export -> tf2onnx.convert.from_keras
    model_proto, _ = tf2onnx.convert.from_keras(
        wrapper,
        input_signature=input_signature,
        output_path=args.output,
        opset=args.opset,
        custom_opset=None,
        custom_ops=None,
        extra_opset=None,
        output_names=output_names
    )
    
    print(f"HiFT ONNX exported to {args.output}")


if __name__ == "__main__":
    main()
```