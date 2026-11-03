# LiDSeg

**Linear DehazeFormer Guidance for Efficient Semantic Segmentation in Adverse Weather Conditions**

BMVC 2026 · Yobsan Bayisa Leta, Christos Kyrkou, Maria K. Michael, Theocharis Theocharides · University of Cyprus

Official **LiDSeg (PIDNet-L)** model submitted to the [ACDC benchmark](https://acdc.vision.ee.ethz.ch/submissions/6ac8c8bb3041fc0c1fd88257).

![LiDSeg architecture](docs/architecture.png)

## Overview

LiDSeg improves semantic segmentation in adverse weather through a lightweight guidance encoder and Lightweight Gated Fusion (LGF). Auxiliary image reconstruction is used during pretraining; its decoder is removed at inference.

PIDNet-L is the backbone used in this release. LiDSeg supports other segmentation backbones through its model and feature registries: replace PIDNet by registering the backbone and its 1/8, 1/16 and 1/32 feature taps in `models/segmodels/`.

## Installation

```bash
conda create -n lidseg python=3.12 -y
conda activate lidseg
pip install -r requirements.txt
cp configs/paths.env.example configs/paths.env
```

## Datasets

| Dataset | Use | Download |
|---|---|---|
| Cityscapes + Foggy Cityscapes | Paired pretraining | [Cityscapes](https://www.cityscapes-dataset.com/) |
| ACDC | Fine-tuning and benchmark evaluation | [ACDC](https://acdc.vision.ee.ethz.ch/) |

Set the dataset roots in `configs/paths.env`; the expected layout is described in its template. Labels use the 19 Cityscapes trainIds, with 255 ignored. Datasets are downloaded separately.

## Model and inference

Download the [official trained model](https://github.com/yobsanb/LiDSeg/releases/download/pidnetl-acdc/lidseg_pidnetl_uber_tta_model.pth) into `pretrained/`.

```python
import torch
from deploy.lidseg_infer import load_model, load_image

model = load_model("pretrained/lidseg_pidnetl_uber_tta_model.pth", device="cuda")
with torch.inference_mode():
    prediction = model.predict(load_image("image.png").to("cuda")).argmax(1)
```

The output contains Cityscapes trainIds. See the [inference guide](deploy/README.md) for dataset evaluation.

## Training

Pretrain on Foggy Cityscapes, fine-tune on ACDC, then generate the benchmark submission:

```bash
python tools/prepare_pidnetl_weights.py
GPUS=0,1,2,3 scripts/step1_foggy_cityscapes.sh
GPUS=0,1 scripts/step2_acdc.sh
GPUS=0,1 scripts/step3_acdc_submission.sh
```

The stage configurations are in `configs/`. Submission inference uses scales 0.5–1.75 and horizontal flipping.

## Results

### Foggy Cityscapes

Validation mIoU (%).

| Model | Clear | Light | Medium | Dense |
|---|---:|---:|---:|---:|
| LiDSeg (PIDNet-L) | 80.35 | 80.40 | 80.17 | 78.84 |

### ACDC

Official test result from the [ACDC benchmark leaderboard](https://acdc.vision.ee.ethz.ch/submissions/6ac8c8bb3041fc0c1fd88257), task *all*.

| Model | mIoU (%) |
|---|---:|
| **LiDSeg (PIDNet-L)** | **74.73** |

## Citation

```bibtex
@inproceedings{leta2026lidseg,
  title     = {LiDSeg: Linear DehazeFormer Guidance for Efficient Semantic
               Segmentation in Adverse Weather Conditions},
  author    = {Leta, Yobsan Bayisa and Kyrkou, Christos and
               Michael, Maria K. and Theocharides, Theocharis},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## License

LiDSeg is released under the MIT License. The vendored backbone ports in `models/segmodels/` and the dataloader base classes in `dataset/` retain their upstream licenses (MIT and Apache-2.0); each file header names its source, copyright holder, and license.

Datasets and third-party pretrained checkpoints are obtained from their original sources.

## Acknowledgements

This work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 101169378 (TIRAMISU Doctoral Network), by the European Union's Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE), and from the Government of the Republic of Cyprus through the Cyprus Deputy Ministry of Research, Innovation and Digital Policy.

LiDSeg builds on [BiSeNetV2](https://github.com/CoinCheung/BiSeNet), [DDRNet](https://github.com/ydhongHIT/DDRNet), [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg), [PIDNet](https://github.com/XuJiacong/PIDNet), [SCTNet](https://github.com/xzz777/SCTNet), [panoptic-deeplab](https://github.com/bowenc0221/panoptic-deeplab), [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), [DehazeFormer](https://github.com/IDKiro/DehazeFormer), [Restormer](https://github.com/swz30/Restormer), and [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).
