# Penny-Wise and Pound-Foolish in Deepfake Detection 

Under construction...

Start cleaning the training code.


## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Download model weights

All checkpoints are trained with different seeds, but in the main paper, we primarily report the performance metrics of the first model (poundnet_ViTL_Progan_20240506_23_30_25).

wget -O ./weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240506_23_30_25/last.ckpt

wget -O ./weights/poundnet_ViTL_Progan_20240804_21_16_47.ckpt https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240804_21_16_47/last.ckpt

wget -O ./weights/poundnet_ViTL_Progan_20240805_10_31_08.ckpt  https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240805_10_31_08/last.ckpt

wget -O ./weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt  https://huggingface.co/nebula/PoundNet/resolve/main/poundnet_ViTL_Progan_20240506_23_30_25/last.ckpt

### Download benchmark data

bash download_data.sh

You can also use, huggingface-cli, to download data.


All datasets are used in Arrow format. If anyone is interested in PoundNet's performance on a publicly available dataset and does not want to build the dataset, you can submit an issue, and we will consider running the model on that dataset and providing the results, even uploading the data to huggingface if possible.

## (2) Test your models

python test.py --cfg cfg/poundnet.yaml

## (A) Acknowledgments

This repository borrows partially from the [CNNDetection](https://github.com/PeterWang512/CNNDetection).


## (B) Cite
````
@misc{wang2024pennywisepoundfoolishdeepfakedetection,
      title={Penny-Wise and Pound-Foolish in Deepfake Detection}, 
      author={Yabin Wang and Zhiwu Huang and Su Zhou and Adam Prugel-Bennett and Xiaopeng Hong},
      year={2024},
      eprint={2408.08412},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08412}, 
}
`````






