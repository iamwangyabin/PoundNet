# PoundNet

Official code release for **Penny-Wise and Pound-Foolish in AI-Generated Image Detection**.

This repository contains training and evaluation code for PoundNet, a CLIP-based detector built around asymmetric prompt learning for binary real/fake classification and category-aware supervision.

## Highlights

- CLIP ViT-L/14 backbone with learnable text and vision prompts
- PyTorch Lightning training pipeline
- Evaluation scripts for multiple deepfake detection benchmarks
- Hugging Face Arrow-based dataset loading
- Released checkpoints for the main reported runs

## Paper

If this repository is useful in your work, please cite:

```bibtex
@article{wang2026pennywise,
  title={Penny-Wise and Pound-Foolish in AI-Generated Image Detection},
  author={Wang, Yabin and Huang, Zhiwu and Su, Zhou and Prugel-Bennett, Adam and Hong, Xiaopeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  pages={1--14},
  year={2026},
  doi={10.1109/TPAMI.2026.3664388}
}
```

## Environment Setup

### 1. Create environment

Install PyTorch first from [pytorch.org](https://pytorch.org/) according to your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 2. Notes on dependencies

- `requirements.txt` installs both `timm` and OpenAI CLIP from GitHub.
- The codebase uses `lightning`, `hydra-core`, `datasets`, `albumentations`, `opencv-python`, and `scikit-learn`.
- GPU inference/training is assumed by default.

## Download Checkpoints

Create a weights directory and download the released checkpoints:

```bash
mkdir -p weights

wget -O ./weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt \
  https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240506_23_30_25/last.ckpt

wget -O ./weights/poundnet_ViTL_Progan_20240804_21_16_47.ckpt \
  https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240804_21_16_47/last.ckpt

wget -O ./weights/poundnet_ViTL_Progan_20240805_10_31_08.ckpt \
  https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240805_10_31_08/last.ckpt
```

The default evaluation configs point to these three checkpoints:

- `cfgs/poundnet.yaml`
- `cfgs/poundnet2.yaml`
- `cfgs/poundnet3.yaml`

## Dataset Format

PoundNet expects datasets saved in Hugging Face Arrow format and loaded with `datasets.load_from_disk(...)`.

Each dataset directory should contain:

- Arrow shard files such as `data-00000-of-xxxxx.arrow`
- `dataset_info.json`
- `mapping.json`
- `state.json`
- A split file such as `train_binary.json` or `test.json`

The split JSON is expected to map subset names to `{relative_image_path: label}` pairs.

## Download Benchmark Data

The repository provides a helper script:

```bash
bash download_data.sh
```

By default, this script downloads:

- `DiffusionForensics`
- `Ojha`
- `DIF`

Important:

- The script downloads into the current directory.
- The default test configs use `datasets.base_path: "/root/autodl-tmp/data"`, so you will usually want to override this from the command line.
- The official training config also expects `ForenSynths`, which is not downloaded by `download_data.sh` and must be prepared separately in the same Arrow format.

Example local layout:

```text
/path/to/DF-arrow
├── DIF
├── DiffusionForensics
├── ForenSynths
└── Ojha
```

## Quick Start

### Evaluate a released model

Run evaluation with a local dataset root override:

```bash
python test.py --cfg cfgs/poundnet.yaml \
  datasets.base_path=/path/to/DF-arrow
```

To test the other released checkpoints:

```bash
python test.py --cfg cfgs/poundnet2.yaml datasets.base_path=/path/to/DF-arrow
python test.py --cfg cfgs/poundnet3.yaml datasets.base_path=/path/to/DF-arrow
```

If needed, you can also override the checkpoint path directly:

```bash
python test.py --cfg cfgs/poundnet.yaml \
  datasets.base_path=/path/to/DF-arrow \
  resume.path=./weights/your_model.ckpt
```

Evaluation prints metrics for each benchmark subset and writes:

- `<test_name>_results.csv`
- `<test_name>.pkl`

Reported metrics include:

- `AP`
- `AUC`
- `F1`
- `ACC`
- `R_ACC`
- `F_ACC`

### Train PoundNet

The official training config is:

```bash
python train.py --cfg cfgs/train/poundnet_official.yaml \
  datasets.base_path=/path/to/DF-arrow
```

Useful overrides:

```bash
python train.py --cfg cfgs/train/poundnet_official.yaml \
  datasets.base_path=/path/to/DF-arrow \
  train.gpu_ids=1 \
  train.train_epochs=10 \
  datasets.train.batch_size=64 \
  datasets.val.batch_size=64
```

Training outputs are saved under `logs/<run_name>/`, including:

- Lightning checkpoints
- `last.ckpt`
- archived source snapshot
- logger outputs

If `logger.use_wandb=true`, the training script will use Weights & Biases; otherwise it falls back to CSV logging.

## Configuration

Configs are managed with Hydra/OmegaConf. The main files are:

- `cfgs/train/poundnet_official.yaml`: official training recipe
- `cfgs/poundnet.yaml`: evaluation config for checkpoint 1
- `cfgs/poundnet2.yaml`: evaluation config for checkpoint 2
- `cfgs/poundnet3.yaml`: evaluation config for checkpoint 3

You can override any field from the command line, for example:

```bash
python test.py --cfg cfgs/poundnet.yaml \
  datasets.base_path=/path/to/DF-arrow \
  datasets.batch_size=32 \
  datasets.loader_workers=8
```

## Repository Structure

```text
.
├── cfgs/                  # training and evaluation configs
├── data/                  # dataset wrappers and augmentations
├── engine/                # Lightning training pipeline
├── networks/              # PoundNet and CLIP backbone code
├── utils/                 # validation, checkpoint loading, config helpers
├── train.py               # training entrypoint
├── test.py                # evaluation entrypoint
└── download_data.sh       # benchmark download helper
```

## Known Assumptions

- The current code assumes CUDA is available for both testing and training.
- Checkpoint loading expects Lightning checkpoints with weights stored under `state_dict`.
- Evaluation uses binary labels where `0` is real and `1` is fake.
- The training config currently defaults to `train_epochs: 1`; increase this for actual training runs.

## Acknowledgments

This repository borrows partially from [CNNDetection](https://github.com/PeterWang512/CNNDetection).

## License

This project is released under the Apache 2.0 License. See `LICENSE` for details.
