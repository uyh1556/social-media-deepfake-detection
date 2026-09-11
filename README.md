# Deepfake Detection Research

Research code for evaluating the robustness and generalization of image-based deepfake detection models.

This public repository provides selected implementation code from an ongoing master's thesis project. Detailed hypotheses, dataset manifests, intermediate results, checkpoints, and unpublished experimental findings are intentionally not included.

## Included

- FaceForensics++ data preparation and leakage-aware split utilities;
- Xception training and checkpoint evaluation code;
- Standard, full-frame letterbox, and face-ROI preprocessing implementations;
- paired-image similarity and prediction-analysis utilities;
- Instagram transformation calibration and analysis utilities.

## Repository structure

```text
.
├── scripts/                    # data, training, evaluation, and analysis code
├── requirements-colab.txt     # training and evaluation dependencies
└── requirements-instagram.txt # Instagram pipeline dependencies
```

## Data and responsible use

No datasets, face images, social-media downloads, credentials, trained checkpoints, or unpublished result files are distributed in this repository. External datasets must be obtained directly from their maintainers and used under their respective terms.

The code is intended solely for academic deepfake-detection research and defensive evaluation. It is not intended for deepfake generation or for the development of systems that could harm individuals or institutions.

## License

No license has yet been granted for the original code in this repository. Third-party datasets and code remain subject to their respective licenses and terms.
