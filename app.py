"""Gradio inference demo for Retinexformer (no dataset or BasicSR install needed)."""

import argparse
import importlib.util
from inspect import signature
from pathlib import Path
from threading import Lock
from time import perf_counter

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent


def load_architecture():
    # Import only the standalone architecture: BasicSR's package imports also
    # initialize training datasets, metrics and optional compiled extensions.
    spec = importlib.util.spec_from_file_location(
        "retinexformer_demo_arch", ROOT / "basicsr/models/archs/RetinexFormer_arch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RetinexFormer


def select_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use auto, cpu, cuda, or cuda:N for --device.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; launch with --cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist.")
    return device


@torch.inference_mode()
def forward_image(model, tensor, ensemble=False):
    """Match test_from_dataset.py's padding and eight-way self-ensemble."""
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    # Reflection requires padding smaller than the input dimension.
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    if not ensemble:
        output = model(tensor)
    else:
        output = torch.zeros_like(tensor)
        for hflip in (False, True):
            for vflip in (False, True):
                for rotate in (False, True):
                    sample = tensor
                    if hflip:
                        sample = sample.flip(-2)
                    if vflip:
                        sample = sample.flip(-1)
                    if rotate:
                        sample = torch.rot90(sample, 1, (-2, -1))
                    prediction = model(sample)
                    if rotate:
                        prediction = torch.rot90(prediction, 3, (-2, -1))
                    if vflip:
                        prediction = prediction.flip(-1)
                    if hflip:
                        prediction = prediction.flip(-2)
                    output.add_(prediction)
        output.div_(8)
    return output[..., :height, :width].clamp(0, 1)


class Enhancer:
    def __init__(self, weights_dir, config, device, initial_weights=None):
        self.weights_dir = Path(weights_dir).expanduser().resolve()
        self.device = select_device(device)
        with Path(config).expanduser().open(encoding="utf-8") as stream:
            network = dict(yaml.safe_load(stream)["network_g"])
        if network.pop("type") != "RetinexFormer":
            raise ValueError("The demo supports RetinexFormer configurations only.")
        self.network = network
        self.model_class = load_architecture()
        self.model = None
        self.loaded_path = None
        self.loaded_stamp = None
        self.lock = Lock()
        self.checkpoints = {}
        self.refresh()
        self.initial_weights = None
        if initial_weights:
            path = Path(initial_weights).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            self.initial_weights = str(path)
            self.checkpoints[str(path)] = path

    def refresh(self):
        self.checkpoints = {
            str(path.relative_to(self.weights_dir)): path
            for path in sorted(self.weights_dir.rglob("*.pth"))
            if "mst" not in str(path.relative_to(self.weights_dir)).lower()
        }
        if getattr(self, "initial_weights", None):
            self.checkpoints[self.initial_weights] = Path(self.initial_weights)
        return list(self.checkpoints)

    def default_weight(self):
        return self.initial_weights or next(
            (key for key in self.checkpoints if Path(key).name == "LOL_v1.pth"),
            next(iter(self.checkpoints), None),
        )

    def load_model(self, path):
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if self.loaded_path == path and self.loaded_stamp == stamp:
            return self.model
        # Use CPU while deserializing and strict loading to catch wrong models.
        load_options = {"map_location": "cpu"}
        # PyTorch 1.11 predates weights_only; newer releases can restrict loading
        # to tensor state dictionaries. Both accept the official checkpoints.
        if "weights_only" in signature(torch.load).parameters:
            load_options["weights_only"] = True
        checkpoint = torch.load(path, **load_options)
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected a state dictionary checkpoint.")
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model = self.model_class(**self.network)
        model.load_state_dict(state, strict=True)
        # Release the previous model before moving the new one to the GPU.
        self.model = None
        self.loaded_path = None
        self.model = model.eval().to(self.device)
        self.loaded_path, self.loaded_stamp = path, stamp
        return self.model

    @torch.inference_mode()
    def process(self, image, weights, gamma=1.0, max_size=1280, ensemble=False):
        if image is None:
            raise gr.Error("Upload a low-light image first.")
        if not weights or weights not in self.checkpoints:
            raise gr.Error("Select a .pth checkpoint; add it to pretrained_weights and click Refresh weights.")
        if not 0.1 <= float(gamma) <= 5:
            raise gr.Error("Gamma must be between 0.1 and 5.")
        if max_size is None or int(max_size) < 0:
            raise gr.Error("Maximum image size must be zero or positive.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        max_size = int(max_size)
        if max_size and max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        try:
            with self.lock:
                started = perf_counter()
                model = self.load_model(self.checkpoints[weights])
                tensor = tensor.to(self.device).pow(float(gamma))
                output = forward_image(model, tensor, bool(ensemble))
                array = output[0].permute(1, 2, 0).cpu().numpy()
                elapsed = perf_counter() - started
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            # PyTorch 1.11 reports CUDA OOM as RuntimeError rather than exposing
            # the newer torch.cuda.OutOfMemoryError exception class.
            if self.device.type == "cuda" and "out of memory" in str(error).lower():
                raise gr.Error("GPU memory is full. Reduce Maximum image size or restart with --cpu.") from error
            raise gr.Error(f"Could not run this checkpoint. Check --opt and the weights file. {error}") from error
        result = Image.fromarray(np.rint(array * 255).astype(np.uint8))
        info = (
            f"{Path(weights).name} | {self.device} | {elapsed:.2f} s (including weight loading on first run)\n"
            f"Input: {original_size[0]} × {original_size[1]} → Output: {result.width} × {result.height} | "
            f"Gamma: {gamma:g} | Self-ensemble: {'on (8 passes)' if ensemble else 'off'}"
        )
        return result, info


def apply_brightness(image, brightness, details=""):
    """Adjust a cached model result so slider changes never compound or rerun it."""
    if not 0.1 <= float(brightness) <= 3.0:
        raise gr.Error("Output brightness must be between 0.1 and 3.0.")
    if image is None:
        return None, details
    result = ImageEnhance.Brightness(image).enhance(float(brightness))
    return result, f"{details}\nOutput brightness: {float(brightness):.2f}×"


def build_demo(enhancer):
    with gr.Blocks(
        title="Retinexformer · Low-Light Image Enhancement",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=".gradio-container {max-width: 1240px !important;} footer {display: none !important;}",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Retinexformer\n"
            "### Low-Light Image Enhancement\n"
            "Upload a dark photo, choose a checkpoint, and reveal the details. "
            "Compare the input and enhanced image side by side, then download the result."
        )
        base_image = gr.State(value=None)
        base_details = gr.State(value="")
        with gr.Row():
            source = gr.Image(label="Low-light image", type="pil", sources=["upload"], height=380)
            result = gr.Image(label="Enhanced result", type="pil", format="png", height=380,
                              interactive=False, show_download_button=True)
        example_paths = sorted((ROOT / "demo/examples").glob("*.png"))
        if example_paths:
            gr.Examples(
                examples=[[str(path)] for path in example_paths], inputs=[source],
                label="Example images — click a thumbnail, then Enhance image",
                examples_per_page=5, cache_examples=False,
            )
        with gr.Row():
            weights = gr.Dropdown(choices=list(enhancer.checkpoints), value=enhancer.default_weight(),
                                  label="Model weights", scale=4,
                                  info="Retinexformer .pth files in pretrained_weights.")
            refresh = gr.Button("Refresh weights", scale=1)
        brightness = gr.Slider(
            0.1, 3.0, value=1.0, step=0.01, label="Output brightness",
            info="1 = model output; >1 brighter, <1 darker. After enhancement, release to update instantly.",
        )
        with gr.Row():
            gamma = gr.Slider(0.1, 5, value=1, step=0.01, label="Gamma curve",
                              info="1 = original input; lower values brighten the input.")
            max_size = gr.Dropdown(choices=[("Original resolution", 0), ("512 px", 512),
                                           ("1024 px", 1024), ("1280 px", 1280), ("2048 px", 2048)],
                                   value=1280, label="Maximum image size",
                                   info="Longest edge. Larger photos are resized before inference.")
            ensemble = gr.Checkbox(value=False, label="Self-ensemble (8 passes)",
                                   info="Average flipped / rotated predictions; takes longer.")
        with gr.Row():
            run = gr.Button("Enhance image", variant="primary", scale=3)
            clear = gr.ClearButton([source, result], value="Clear images", scale=1)
        info = gr.Textbox(
            label="Inference details", interactive=False, lines=3,
            info="Selected weights, CPU/GPU, elapsed time, input/output dimensions and enhancement settings.",
        )
        gr.Markdown(
            f"**Device:** `{enhancer.device}` · **Tip:** Start with Gamma = 1.0. "
            "Use the download icon on the result to save a PNG.\n\n"
            "[Paper](https://arxiv.org/abs/2303.06705) · "
            "[Code](https://github.com/caiyuanhao1998/Retinexformer) · "
            "[Pretrained models](https://huggingface.co/CaiYuanhao/Retinexformer)"
        )

        def refresh_weights():
            choices = enhancer.refresh()
            return gr.Dropdown(choices=choices, value=enhancer.default_weight())

        def enhance(image, weight, gamma_value, size, use_ensemble, brightness_value):
            original, details = enhancer.process(image, weight, gamma_value, size, use_ensemble)
            adjusted, displayed_details = apply_brightness(original, brightness_value, details)
            return adjusted, displayed_details, original, details

        source.change(lambda: (None, None, "", ""),
                      outputs=[result, base_image, base_details, info], queue=False, api_name=False)
        refresh.click(refresh_weights, outputs=weights, queue=False)
        clear.click(lambda: (None, "", ""), outputs=[base_image, base_details, info], queue=False)
        run.click(enhance, [source, weights, gamma, max_size, ensemble, brightness],
                  [result, info, base_image, base_details], api_name="enhance", concurrency_limit=1)
        brightness.release(apply_brightness, [base_image, brightness, base_details], [result, info],
                           queue=False, api_name=False)
    return demo.queue(max_size=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "pretrained_weights")
    parser.add_argument("--weights", type=Path, help="Optional initial checkpoint, including outside weights-dir")
    parser.add_argument("--opt", type=Path, default=ROOT / "Options/RetinexFormer_LOL_v1.yml",
                        help="YAML containing the matching RetinexFormer network_g architecture")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU; overrides --device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio sharing link")
    args = parser.parse_args()
    try:
        enhancer = Enhancer(args.weights_dir, args.opt, "cpu" if args.cpu else args.device, args.weights)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if not enhancer.checkpoints:
        print("No weights found. Download a Retinexformer .pth into pretrained_weights, then click Refresh weights.")
    build_demo(enhancer).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
