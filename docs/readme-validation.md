# README command validation

Validated on 2026-11-08 (America/Los_Angeles), using an isolated checkout and two initially clean Conda environments. The audit found failures in the previous README; the installation commands and source fixes described below were applied and retested.

## Tested environments

| Component | ICCV environment | PyTorch 2 environment |
| --- | --- | --- |
| Python | 3.10.22 | 3.10.22 |
| PyTorch / torchvision | 1.11.0 / 0.12.0 | 2.5.1 / 0.20.1 |
| CUDA runtime | 11.3 | 11.8 |
| NumPy / MKL | 1.26.4 / 2023.1.0 | 1.26.4 / 2023.1.0 |
| Gradio / huggingface_hub | 5.50.0 / 1.33.0 | 5.50.0 / 1.33.0 |
| setuptools | 72.1.0 | 72.1.0 |

Hardware: NVIDIA RTX A5000, 24 GB VRAM per GPU, on a shared server. Installation, BasicSR development installation, CUDA tensor operations and `pip check` passed. The final pinned Conda recipes were also solved from scratch with `--dry-run`.

## Command coverage

| README workflow | Verification | Result |
| --- | --- | --- |
| Sections 1.1 and 1.2 installation | Both Conda/PyTorch installations, pip dependency commands and `python setup.py develop --no_cuda_ext` | Passed after compatibility fixes |
| Model download | `hf download ... --include "*.pth" --local-dir pretrained_weights` in both environments | All 11 checkpoints downloaded: nine Retinexformer and two MST++ |
| Gradio demo | CPU and CUDA startup in both environments, nine model choices, actual enhancement and PNG download | All four combinations passed |
| Benchmark commands | All 14 documented evaluation commands in each environment, including the three `--GT_mean` commands | 28 command runs passed; see dataset scope below |
| Self-ensemble | LOL-v1 and SID with `--self_ensemble`, in both environments | Passed |
| High-resolution split-and-test | 3001 × 65 input, with and without self-ensemble, in both environments | Passed after padding fix; original dimensions preserved |
| Params/FLOPs snippet | Execute the README Python snippet in both environments | 1,605,701 parameters; fvcore reports 15.567627 GMac at 256 × 256 |
| Single-GPU training | Nine Retinexformer configurations per environment: one optimizer step, validation and checkpoint saving | 18 runs passed |
| Training resume | Resume LOL-v1 in PyTorch 1.11 and AMP NTIRE in PyTorch 2 from iteration 1 to 2 | Passed |
| Distributed launchers | Retinexformer NTIRE and both MST++ NTIRE configurations: two optimizer steps, validation and model/state checkpoint saving | Passed on two GPUs with `NCCL_P2P_DISABLE=1` |
| Original NTIRE per-GPU patch sizes | 2000 (Retinexformer, AMP), 1800 (MST++, AMP), 1150 (MST++, FP32), batch size 1 | CUDA out of memory on this 24 GB server; defaults are not directly runnable here |
| Reduced NTIRE per-GPU patch size | All three configurations at 512 × 512, batch size 1, original AMP settings; one training step, validation and checkpoint save | All three passed |
| CLI GPU selection | Evaluate with `--gpus 1,6` in both environments | Passed |
| Invalid data | Empty input directory and unequal input/target counts | Now fail with explicit errors |
| Regression tests | `python -m unittest discover -s tests -v` in both CUDA environments | Passed |

The FLOPs helper uses fvcore's supported-operation count; fvcore still prints warnings for operations it does not count.

## Full LOL-v1 test-set results

These evaluations used all 15 original LOL-v1 test pairs and the official `LOL_v1.pth`, without self-ensemble.

| Environment | Default PSNR / SSIM | With `--GT_mean` PSNR / SSIM |
| --- | --- | --- |
| PyTorch 1.11 | 25.155017 / 0.845461 | 27.182371 / 0.850052 |
| PyTorch 2.5.1 | 25.156158 / 0.845456 | 27.182254 / 0.850043 |

For each environment, all 15 normal-resolution output PNGs were byte-identical before and after the split-and-test fix.

## Fixes found by the audit

- Pin NumPy and MKL in Conda before pip installation. The previous unconstrained recipes selected MKL 2025 and NumPy 2, causing a PyTorch `iJIT_NotifyEvent` import error and incompatible NumPy array/ufunc behavior.
- Pin setuptools to 72.1.0 and install Cython. Setuptools 84.2.1 redirected `setup.py develop` to an isolated editable build that could not import PyTorch.
- Remove the redundant unbounded `pip install -U huggingface_hub`: its current resolution upgrades to 2.4.0, outside Gradio 5.50.0's `<2` requirement. The environment installation already supplies the compatible `hf` CLI.
- Install fvcore, add the missing FLOPs helper imports, and use a valid Retinexformer constructor and import path in the README snippet.
- Match the HF download layout for MST++ weights; fix the nonexistent four-GPU training configuration name and use four GPU IDs in that example.
- Correct the SMID split-list location to `data/SMID/test_list.txt` and the NTIRE validation folder to `data/NTIRE/mini_val`.
- Correct the FiveK training input path from `train/target` to `train/input`.
- Pad the split width to a multiple of eight so both halves remain valid multiples of four. The previous high-resolution path crashed on the odd-width regression input.
- Reject empty and mismatched image sets. The previous evaluator could exit successfully with `PSNR: nan` or silently evaluate only the shorter list.
- Preserve comma-separated GPU IDs instead of inserting commas between every character.

The server's NCCL P2P initialization hung in all three distributed configurations. A minimal all-reduce and the full short training runs passed with `NCCL_P2P_DISABLE=1`; README now documents this observed PCIe-system workaround.

## Original-size training memory checks

One-step training was also attempted with the original NTIRE per-GPU batch size and patch sizes, using one GPU and the original AMP setting. All three configurations failed with CUDA out of memory. The Retinexformer run used approximately 23.34 GiB when it failed on a nearly idle card; the shared-server MST++ runs also had roughly 1 GiB occupied by another process. These observations do not establish a minimum VRAM requirement on other systems.

The README documents reducing both `gt_size` and `gt_sizes` when memory is insufficient. The original paper configurations remain at their original patch sizes. Retesting all three at 512 × 512 with batch size 1 passed. This changes the training setup and is not a reproduction of the original training recipe.

## Scope and limits

Only LOL-v1 used a complete original benchmark test split. The other evaluations used small format fixtures derived from a LOL-v1 pair, including NPY sequences and split lists, to exercise their actual loaders, networks, metrics and output saving. Their smoke metrics are not benchmark results.

The main training smoke matrix used audit-only copies of the YAMLs: 32-pixel patches, reduced batch/worker counts and one or two iterations. Separate original-size and 512-pixel memory checks are recorded above. Model architectures were unchanged. Saved checkpoints were additionally checked for finite weights and actual optimizer-step counters. These tests exercised real backward passes, validation, optimizer/scaler state saving and resume; they did not reproduce full 150,000-iteration training.

Distributed tests used two available GPUs and small patches, not full four/eight-GPU training. Original-size per-GPU attempts failed as recorded above. Full datasets and paper-scale training remain unverified. GPU IDs, ports and environment prefixes were adapted to the validation server. The SSH forwarding example contains user/server placeholders and was not executed literally.

## Re-run the regression checks

After installing either README environment, select an available CUDA GPU and run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests -v
```

The regression checks generate their own temporary data and random weights; they need no benchmark downloads. They test command behavior and output dimensions, not model quality. They are skipped when CUDA is unavailable.
