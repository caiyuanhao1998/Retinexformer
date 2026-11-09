# Publish the online demo

From the repository root, with the demo dependencies installed:

```bash
hf auth login
python scripts/deploy_hf_space.py
```

This creates a public `<your-account>/Retinexformer` Gradio Space. It requests free ZeroGPU and falls back to free CPU Basic if the account cannot use ZeroGPU. It never requests paid hardware. Use `--hardware cpu-basic` to select CPU explicitly.

To deploy later updates:

```bash
python scripts/deploy_hf_space.py --repo-id CaiYuanhao/Retinexformer --update
```

The script packages only the shared demo, architecture, model configuration, example images, screenshot and hosting files. It does not upload tokens, local environments, uploaded photos or checkpoint files. The Space downloads the nine official checkpoints at a pinned revision on startup.

`space_app.py` adds ZeroGPU execution and preloads GPU weights at startup. The local `app.py` remains usable independently. CPU hosting limits the longest edge to 1024 pixels; GPU hosting offers up to 2048 pixels.

To inspect the upload directory without publishing:

```bash
python scripts/deploy_hf_space.py --prepare-only /tmp/retinexformer-space
```
