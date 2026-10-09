# DAX-Net: degradation-aware enhancement of drone imagery

Code for the paper *Improving Drone Image Quality for Environmental Resource Analysis with a Degradation-Aware Enhancement Network* (U. Kumaran, M. Karthikeyan, Signal, Image and Video Processing, under review).

DAX-Net restores drone tiles affected by low light, defocus, motion blur and sensor noise. An ImageNet-pretrained MobileNetV2 encoder and an auxiliary branch built on illumination and gradient priors exchange information through bidirectional cross-attention; an SE-gated decoder predicts a residual correction. Loss weights and the learning rate are selected with the dingo optimisation algorithm (DOA). Every number in the paper is produced by the scripts in this folder.

## Results (2,080 degraded SDD test tiles)

All learned models were trained with the same data and budget (3,000 iterations, batch 8, 96 × 96 crops) on two CPU cores. GMACs at 256 × 256 input; time in CPU seconds per tile; Acc and F1 averaged over four environmental tasks.

| Method | Params (M) | GMACs | Time (s) | PSNR (dB) | SSIM | GMSD | NIQE | Acc (%) | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Degraded input | – | – | – | 16.05 | 0.459 | 0.139 | 10.26 | 84.3 | 0.716 |
| CLAHE | – | – | 0.002 | 16.82 | 0.509 | 0.124 | 10.79 | 78.5 | 0.655 |
| Zero-DCE | 0.08 | 5.19 | 0.093 | 13.48 | 0.416 | 0.201 | 11.86 | 77.7 | 0.581 |
| SRCNN | 0.07 | 4.53 | 0.052 | 20.08 | 0.584 | 0.123 | 9.23 | 85.5 | 0.739 |
| ESRGAN-C | 1.10 | 17.93 | 0.200 | 18.51 | 0.555 | 0.130 | 9.01 | 85.8 | 0.738 |
| U-Net-C | 7.77 | 12.08 | 0.127 | 20.74 | 0.590 | 0.121 | 8.80 | 85.1 | 0.733 |
| NAFNet-C | 1.72 | 1.51 | 0.071 | 21.56 | 0.607 | 0.109 | 7.77 | 85.8 | 0.739 |
| Restormer-C | 2.65 | 12.14 | 0.800 | 22.01 | 0.611 | 0.108 | 8.58 | 86.1 | 0.745 |
| **DAX-Net** | 2.59 | 2.60 | 0.053 | **22.88** | **0.671** | **0.098** | 8.34 | **88.5** | **0.789** |
| Clean tiles | – | – | – | – | – | – | 6.44 | 94.3 | 0.911 |

"-C" marks compact configurations of the published architectures (Table S2 of the paper). DAX-Net exceeds Restormer-C on 59.2% of tiles (paired Wilcoxon p = 5.5 × 10⁻¹⁹). `reference_results_summary.json` holds every value reported in the paper, including the ablation, encoder, robustness and per-task results.

Ablation summary (reference: DOA weights with λ3 = 0.5, 22.38 dB): without low-light synthesis −6.23 dB; without the SSIM term −0.49 dB; uniform loss weights instead of DOA −0.29 dB; without the auxiliary branch −0.31 dB (p = 0.058); concatenation instead of cross-attention +0.27 dB; without SE gates +0.18 dB; without the VGG term +0.50 dB. The final model (`abl_novgg`) keeps the DOA values of λ2 and η0 and sets λ3 = 0, a choice made on the validation split.

## Files

| File | Role |
|---|---|
| `prep.py` | ×0.25 area down-sampling, 256-px tiles with stride 224, image-level 280/60/60 split |
| `degradation.py` | camera-response exposure model, Gaussian and motion kernels |
| `dax_net.py` | auxiliary branch, cross-attention, SE gate and decoder blocks |
| `baselines.py` | SRCNN, U-Net, RRDB generator (ESRGAN), Zero-DCE network |
| `restormer_arch.py`, `NAFNet_arch.py`, `arch_util.py`, `local_arch.py` | published Restormer and NAFNet architectures (MIT, see `THIRD_PARTY_NOTICES.md`) |
| `doa.py` | dingo optimisation algorithm |
| `metrics.py` | macro precision, recall and F1 |
| `exp.py` | experiment driver: degradation draw, DOA, training, evaluation, downstream tasks, robustness, complexity |
| `convert_weights.py` | converts ONNX-zoo SqueezeNet 1.1 and VGG-16 to PyTorch state dicts |
| `summarize.py` | collects all paper numbers and Wilcoxon tests into `results/summary.json` |
| `make_result_figs.py` | result and supplementary figures |
| `run_all.sh` | full pipeline |

## Configuration names

| Name in code | Model in the paper |
|---|---|
| `abl_novgg` | final DAX-Net (Table 2) |
| `daxnet` | DAX-Net with the DOA weights, λ3 = 0.5 (ablation reference, MobileNetV2 row of Table S6) |
| `enc_squeezenet`, `enc_resnet18`, `enc_resnet50` | encoder study (Table S6) |
| `abl_noca`, `abl_noaux`, `abl_nose`, `abl_nossim`, `abl_nolowlight`, `abl_nodoa` | ablation (Table 3) |
| `srcnn`, `zerodce`, `esrgan`, `unet`, `nafnet`, `restormer` | baselines (Table 2) |

## Setup

```bash
pip install -r requirements.txt
mkdir -p weights && cd weights
wget https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
wget https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.onnx
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
cd ..
```

Download the Semantic Drone Dataset (training set, 400 images) from http://dronedataset.icg.tugraz.at and point the scripts at the folder that contains `original_images/` and `label_images_semantic/`. The weight folder can be moved by setting `DAX_WEIGHTS`. NIQE and GMSD come from IQA-PyTorch (`pyiqa`), which downloads its NIQE parameters on first use.

## Reproduce

```bash
bash run_all.sh /path/to/semantic_drone_dataset
```

The pipeline needs about 12 hours on two CPU cores; single steps can be run separately, for example `python exp.py train abl_novgg` followed by `python exp.py evaluate abl_novgg`. Results are written to `results/`, checkpoints to `ckpt/`, enhanced test tiles to `outputs/` and figures to `figures/`.

Tested with Python 3.13, PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30, pyiqa 0.1.16 and scikit-image 0.26.

## Citation

```
@article{kumaran_daxnet,
  author  = {Kumaran, U. and Karthikeyan, M.},
  title   = {Improving Drone Image Quality for Environmental Resource Analysis with a Degradation-Aware Enhancement Network},
  journal = {Signal, Image and Video Processing},
  note    = {under review}
}
```

## Licence

MIT (see `LICENSE`). Third-party files keep their own MIT notices.
