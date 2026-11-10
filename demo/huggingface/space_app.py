"""Hosted entry point; deployed alongside the repository's shared app.py."""

import os

# Import spaces before app/torch to enable ZeroGPU's CUDA emulation at startup.
import spaces
import torch
from huggingface_hub import snapshot_download

from app import ROOT, Enhancer, build_demo
from space_success_stats import SuccessStats, add_usage_panel

MODEL_REPO = "CaiYuanhao/Retinexformer"
MODEL_REVISION = "51d9f18d5bd8fefc85d9dabf95caf2fa2a5f2018"
CHECKPOINTS = (
    "LOL_v1.pth", "LOL_v2_real.pth", "LOL_v2_synthetic.pth", "FiveK.pth",
    "SID.pth", "SMID.pth", "SDSD_indoor.pth", "SDSD_outdoor.pth", "NTIRE.pth",
)


class SpaceEnhancer(Enhancer):
    def __init__(self, weights_dir, config):
        self.models = {}
        super().__init__(weights_dir, config, "auto")
        self.max_size = 2048 if self.device.type == "cuda" else 1024
        if self.device.type == "cuda":
            # Only ~56 MiB for all nine models. Place them on CUDA at startup so
            # ZeroGPU can restore them without transferring weights every request.
            for path in self.checkpoints.values():
                self.load_model(path)

    def load_model(self, path):
        if path in self.models:
            return self.models[path]
        model = super().load_model(path)
        if self.device.type == "cuda":
            self.models[path] = model
        return model

    def process(self, image, weights, gamma=1.0, max_size=1024, ensemble=False):
        if max_size is None or not 1 <= int(max_size) <= self.max_size:
            raise ValueError(f"Choose an image size between 1 and {self.max_size} pixels.")
        return super().process(image, weights, gamma, max_size, ensemble)


def make_demo():
    torch.set_num_threads(2)
    stats = SuccessStats.from_env()
    stats.start("retinexformer")
    weights_dir = ROOT / "pretrained_weights"
    snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION,
        allow_patterns=list(CHECKPOINTS), local_dir=weights_dir,
    )
    enhancer = SpaceEnhancer(weights_dir, ROOT / "Options/RetinexFormer_LOL_v1.yml")
    process_image = enhancer.process

    # Only image/settings cross the ZeroGPU process boundary. Decorating a
    # bound method would also try to pickle the enhancer, models and its lock.
    @spaces.GPU(duration=60)
    def process(image, weights, gamma=1.0, max_size=1024, ensemble=False):
        return process_image(image, weights, gamma, max_size, ensemble)

    enhancer.process = process
    sizes = [512, 1024, 1280, 2048] if enhancer.device.type == "cuda" else [512, 768, 1024]
    demo = build_demo(
        enhancer, max_size_choices=[(f"{size} px", size) for size in sizes],
        default_max_size=1280 if enhancer.device.type == "cuda" else 1024,
        on_success=lambda: stats.record("retinexformer"),
    )
    add_usage_panel(demo, stats, "retinexformer")
    return demo


if __name__ == "__main__":
    make_demo().launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="20mb",
    )
