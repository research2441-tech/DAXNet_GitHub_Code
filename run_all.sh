#!/usr/bin/env bash
# Reproduces every table and figure of the paper. Usage: bash run_all.sh /path/to/semantic_drone_dataset
# Expects ./weights to hold the five ImageNet weight files listed in README.md.
set -euo pipefail
cd "$(dirname "$0")"
SDD=${1:-sdd_data}
python convert_weights.py                      # ONNX-zoo SqueezeNet 1.1 / VGG-16 -> PyTorch
python prep.py "$SDD"                          # x0.25 area down-sampling, 256-px tiles, 280/60/60 split
python exp.py prepare                          # one fixed degradation draw per validation/test tile
python exp.py doa                              # DOA search of (lambda2, lambda3, eta0)
for n in daxnet srcnn zerodce unet nafnet esrgan restormer \
         enc_squeezenet enc_resnet18 enc_resnet50 \
         abl_noca abl_noaux abl_nose abl_novgg abl_nossim abl_nolowlight abl_nodoa; do
  python exp.py train "$n"
done
for n in input clean clahe daxnet abl_novgg srcnn zerodce unet nafnet esrgan restormer; do
  python exp.py evaluate "$n"                  # full evaluation, outputs kept for the downstream study
done
for n in enc_squeezenet enc_resnet18 enc_resnet50 abl_noca abl_noaux abl_nose abl_nossim abl_nolowlight abl_nodoa; do
  python exp.py evaluate "$n" lite             # quality metrics only
done
python exp.py downstream                       # fixed classifier, four environmental tasks
python exp.py robustness                       # extra test-time noise: SRCNN and DOA-weight model
python exp.py robustness_final                 # extra test-time noise: final model
python exp.py complexity                       # parameters and GMACs at 256 x 256
python summarize.py                            # results/summary.json (all numbers in the paper)
python make_result_figs.py figures             # result figures and supplementary figures
