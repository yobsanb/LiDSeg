# LiDSeg

**Linear DehazeFormer Guidance for Efficient Semantic Segmentation in Adverse Weather Conditions**

BMVC 2026 · Yobsan Bayisa Leta, Christos Kyrkou, Maria K. Michael, Theocharis Theocharides · University of Cyprus

This repository releases **LiDSeg (PIDNet-L)** model submitted to the [ACDC benchmark](https://acdc.vision.ee.ethz.ch/submissions/6ac8c8bb3041fc0c1fd88257): **74.73 mIoU** on the ACDC test set (task *all*).

![LiDSeg architecture](docs/architecture.png)

## Overview

LiDSeg improves semantic segmentation under adverse weather using a lightweight guidance encoder supervised by auxiliary image reconstruction. At inference, the reconstruction decoder is removed, while the guidance encoder and Lightweight Gated Fusion (LGF) remain active.

This release uses PIDNet-L as the segmentation backbone. PIDNet-L can be replaced: the framework supports other segmentation networks that expose features at 1/8, 1/16 and 1/32 resolution.

### Core components

- **Guidance encoder** (`models/dehazeformer.py`): gated linear attention at 1/8, 1/16, and 1/32 scales with 32/64/128 channels.
- **Reconstruction decoder** (`models/dehazeformer.py`): training-only Restormer-style decoder with PixelShuffle upsampling and residual reconstruction.
- **LGF** (`models/lgf.py`): channel-wise gated fusion at s8/s16/s32 with learnable per-stage scaling.
- **Losses** (`models/loss.py`): OHEM cross-entropy, Lovász-Softmax, auxiliary segmentation loss, Charbonnier reconstruction loss, and uncertainty weighting.
- **Wrapper** (`models/lidseg.py`): connects the segmentation, guidance branch, and LGF. Set `guidance=False` for the unguided baseline.

## Installation

```bash
conda create -n lidseg python=3.12 -y
conda activate lidseg
pip install -r requirements.txt
```

For a different CUDA runtime, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) first.

## Datasets

Datasets are not redistributed. Obtain them from the official sources.

| Dataset | Use | Source |
|---|---|---|
| Cityscapes + Foggy Cityscapes | Paired pretraining | [Cityscapes](https://www.cityscapes-dataset.com/) |
| ACDC | Fine-tuning and evaluation | [ACDC](https://acdc.vision.ee.ethz.ch/) |

Labels use Cityscapes `trainIds` (19 classes, `255` ignore). For Cityscapes, generate them with [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) (`csCreateTrainIdLabelImgs`). Configure paths once:

```bash
cp configs/paths.env.example configs/paths.env
# edit configs/paths.env
```

Expected roots:

```text
$CITYSCAPES_ROOT/
  leftImg8bit/{train,val}/...
  gtFine/{train,val}/...
  foggy_0.00/leftImg8bit/...
  Foggy_{0.005,0.01,0.02}/leftImg8bit/...

$ACDC_ROOT/
  rgb_anon/<condition>/{train,val,test}/...
  gt/<condition>/{train,val}/...
```

## Pretrained Model

Download [`lidseg_pidnetl_uber.pth`](https://github.com/yobsanb/LiDSeg/releases/download/pidnetl-acdc/lidseg_pidnetl_uber.pth) from the [release page](https://github.com/yobsanb/LiDSeg/releases/tag/pidnetl-acdc) into `pretrained/`.

```bash
python tools/infer.py --input image.png --output predictions/ --color
```

Predictions are saved as Cityscapes `trainIds`; `--color` also saves color maps. `--input` accepts an image or a folder.

## Training

### Stage 1: Foggy Cityscapes

Paired clear/foggy training with the progressive fog curriculum `0.005 -> 0.01 -> 0.02`

```bash
python tools/prepare_pidnetl_weights.py
GPUS=0,1,2,3 scripts/step1_foggy_cityscapes.sh
```

Output goes to `output/cityscapes/pidnetl_guided`.

### Stage 2: ACDC

The reconstruction decoder and reconstruction loss are removed. Fine-tuning uses segmentation labels from all four conditions.

```bash
GPUS=0,1 scripts/step2_acdc.sh
```

Output goes to `output/acdc/pidnetl_norecon_uber_bblr0.1`.

### Stage 3: ACDC benchmark

Generates the ACDC test-set predictions and the released model file:

```bash
GPUS=0,1 scripts/step3_acdc_benchmark.sh
```

Output goes to `output/acdc_benchmark`.

## Evaluation

```bash
# Foggy Cityscapes
python evaluation/eval_cs_foggy.py --exp_dir output/cityscapes/pidnetl_guided

# ACDC val, released model
python tools/acdc_predict.py val --weights pretrained/lidseg_pidnetl_uber.pth
```

## Results

### Foggy Cityscapes

Validation mIoU (%).

| Model | Clear | Light | Medium | Dense | Drop |
|---|---:|---:|---:|---:|---:|
| **LiDSeg (PIDNet-L)** | 80.35 | 80.40 | 80.17 | 78.84 | -1.51 |

### ACDC

Test per-class IoU (%), all conditions, from the [official ACDC leaderboard](https://acdc.vision.ee.ethz.ch/submissions/6ac8c8bb3041fc0c1fd88257).

| Model | Road | Sidew. | Build. | Wall | Fence | Pole | Light | Sign | Veg. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | Motor. | Bike | **mIoU** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LiDSeg (PIDNet-L)** | 94.87 | 79.81 | 90.05 | 53.82 | 53.09 | 65.43 | 80.13 | 76.47 | 89.37 | 66.01 | 96.35 | 75.44 | 53.88 | 90.66 | 66.93 | 79.13 | 89.01 | 53.69 | 65.70 | **74.73** |

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

LiDSeg is released under the MIT License. The PIDNet port in `models/segmodels/` (MIT) and the dataloader classes in `dataset/` (Apache-2.0) retain their upstream licenses; each file header names its source, copyright holder, and license.

Datasets are not redistributed and remain subject to their own licenses.

## Acknowledgements

This work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 101169378 (TIRAMISU Doctoral Network), by the European Union's Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE), and from the Government of the Republic of Cyprus through the Cyprus Deputy Ministry of Research, Innovation and Digital Policy.

LiDSeg builds on [BiSeNetV2](https://github.com/CoinCheung/BiSeNet), [DDRNet-23s](https://github.com/ydhongHIT/DDRNet), [STDC1-Seg](https://github.com/MichaelFan01/STDC-Seg), [PIDNet-L](https://github.com/XuJiacong/PIDNet), [TopFormer-B](https://github.com/hustvl/Topformer), and [SeaFormer-B](https://github.com/fudan-zvg/SeaFormer). The implementation utilized [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Panoptic-DeepLab](https://github.com/bowenc0221/panoptic-deeplab), [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), [DehazeFormer](https://github.com/IDKiro/DehazeFormer), [Restormer](https://github.com/swz30/Restormer), and [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).
