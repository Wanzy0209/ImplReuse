# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import sys
import torch
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


class HiFTInferenceWrapper(torch.nn.Module):
    def __init__(self, hift_module: torch.nn.Module):
        super().__init__()
        self.hift = hift_module

    def forward(
        self,
        speech_feat: torch.Tensor,
        cache_source: torch.Tensor,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str, device: torch.device) -> torch.nn.Module:
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
    state_dict = torch.load(
        os.path.join(model_dir, "hift.pt"),
        map_location=device,
    )
    state_dict = {
        k.replace("generator.", ""): v for k, v in state_dict.items()
    }
    hift.load_state_dict(state_dict, strict=True)
    hift.to(device).eval()
    return hift


def build_example_inputs(
    device: torch.device,
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    speech_feat = torch.randn(
        1,
        80,
        mel_frames,
        dtype=torch.float32,
        device=device,
    )
    cache_source = torch.zeros(
        1,
        1,
        cache_frames,
        dtype=torch.float32,
        device=device,
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    hift = load_hift(args.model_dir, device)
    wrapper = HiFTInferenceWrapper(hift).to(device)

    example_inputs = build_example_inputs(
        device=device,
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    input_names = ["mel", "cache_in"]
    output_names = ["speech", "cache_out"]
    dynamic_axes = {
        "mel": {0: "batch", 2: "mel_frames"},
        "cache_in": {0: "batch", 2: "cache_frames"},
        "speech": {0: "batch", 2: "audio_samples"},
        "cache_out": {0: "batch", 2: "cache_frames_out"},
    }

    torch.onnx.export(
        wrapper,
        example_inputs,
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        report=True
    )
    print(f"HiFT ONNX exported to {args.output}")


if __name__ == "__main__":
    main()