<img width="2045" height="521" alt="4f8ddb27024234ca1f1774081d4c6fc7681ef8192eaabd4f775aa8173b42682a" src="https://github.com/user-attachments/assets/4ca34bb6-df38-4eb6-a10c-9299657a13b0" />

# Penny-Wise and Pound-Foolish in AI-Generated Image Detection

Official code for **Penny-Wise and Pound-Foolish in AI-Generated Image Detection**.

This repository contains training and evaluation code for PoundNet, a CLIP-based detector built around asymmetric prompt learning for binary real/fake classification and category-aware supervision.

## 1. Environment Setup

Install PyTorch first from [pytorch.org](https://pytorch.org/) according to your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 2. Download Datasets

PoundNet expects datasets saved in Hugging Face Arrow format and loaded with `datasets.load_from_disk(...)`.
Each dataset directory should contain:

- Arrow shard files such as `data-00000-of-xxxxx.arrow`
- `dataset_info.json`
- `mapping.json`
- `state.json`
- A split file such as `train_binary.json` or `test.json`

The repository provides a helper script:

```bash
bash download_data.sh
```

Example local layout:

```text
/path/to/DF-arrow
├── DIF
├── DiffusionForensics
├── ForenSynths
└── Ojha
```

## 3. Quick Start

### Evaluate

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

```bash
python test.py --cfg cfgs/poundnet.yaml \
  datasets.base_path=/path/to/DF-arrow
```

### Train

The official training config is:

```bash
python train.py --cfg cfgs/train/poundnet_official.yaml \
  datasets.base_path=/path/to/DF-arrow
```


## 4. Acknowledgments

This repository borrows partially from [CNNDetection](https://github.com/PeterWang512/CNNDetection).

## 5. Citation

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
