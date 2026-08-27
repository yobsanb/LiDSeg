# LiDSeg

**Linear DehazeFormer Guidance for Efficient Semantic Segmentation in Adverse Weather Conditions**

BMVC 2026 · Yobsan Bayisa Leta, Christos Kyrkou, Maria K. Michael, Theocharis Theocharides · University of Cyprus

![LiDSeg architecture](docs/architecture.png)

## Overview

LiDSeg improves semantic segmentation under adverse weather using a lightweight guidance encoder supervised by auxiliary image reconstruction. At inference, the reconstruction decoder is removed, while the guidance encoder and Lightweight Gated Fusion (LGF) remain active. The design is plug-and-play, lightweight, and suitable for real-time edge deployment.

### Core components

- **Guidance encoder** (`models/dehazeformer.py`): gated linear attention at 1/8, 1/16, and 1/32 scales with 32/64/128 channels.
- **Reconstruction decoder** (`models/dehazeformer.py`): training-only Restormer-style decoder with PixelShuffle upsampling and residual reconstruction.
- **LGF** (`models/lgf.py`): channel-wise gated fusion at s8/s16/s32 with learnable per-stage scaling.
- **Losses** (`models/loss.py`): cross-entropy, Lovász-Softmax, auxiliary segmentation loss, Charbonnier reconstruction loss, and uncertainty weighting.
- **Wrapper** (`models/lidseg.py`): connects the backbone, guidance branch, and LGF. Set `guidance=False` for the unguided baseline.

## Installation

```bash
conda create -n lidseg python=3.10 -y
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
| Foggy Zurich v2 | Zero-shot evaluation | [Foggy Zurich v2](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/) |
| Foggy Driving | Zero-shot evaluation | [Foggy Driving](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) |

Labels must use Cityscapes `trainIds` (19 classes, `255` ignore). For ACDC conversion when needed:

```bash
python tools/utils/convert_acdc_labels.py --input /path/to/ACDC
```

Configure paths once:

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

## Pretrained Backbones

Inspect available checkpoints:

```bash
python tools/fetch_pretrained.py --list
```

```bash
NGPUS=4 scripts/train_cityscapes.sh bisenetv2 clear
```

This writes `output/pretrain/bisenetv2/`, whose `best.pth` is a valid `PRETRAIN_FROM` for Stage 1.

## Training

### Stage 1: Foggy Cityscapes

Paired clear/foggy training with the progressive fog curriculum `0.005 -> 0.01 -> 0.02`:

```bash
NGPUS=4 scripts/train_cityscapes.sh <backbone> [guided|baseline|norecon|clear]
```

```bash
NGPUS=4 scripts/train_cityscapes.sh bisenetv2            # guided, the full method
NGPUS=4 scripts/train_cityscapes.sh ddrnet baseline      # no guidance
NGPUS=4 scripts/train_cityscapes.sh stdc norecon         # guidance without L_rec
```

Output goes to `output/cityscapes/<backbone>_<variant>`. The final dense-fog checkpoint is `best_stage_4.pth`, also published as `best.pth`.

### Stage 2: ACDC

The reconstruction decoder and reconstruction loss are removed. Fine-tuning uses segmentation labels only.

```bash
NGPUS=4 scripts/train_acdc.sh <backbone> [uber|fog|night|rain|snow]
```

```bash
CKPT=output/cityscapes/bisenetv2_guided/best_stage_4.pth
PRETRAIN_FROM=$CKPT NGPUS=4 scripts/train_acdc.sh bisenetv2         # Uber, all conditions
PRETRAIN_FROM=$CKPT NGPUS=4 scripts/train_acdc.sh bisenetv2 night   # condition Expert
```

Output goes to `output/acdc/<backbone>_norecon_<target>`.

Supported backbones: `bisenetv2`, `ddrnet`, `stdc`, `pidnets`, `pidnetm`, `pidnetl`, `sctnets`, `sctnetb`.

### Configs

Three configs cover every run. `--variant` sets `guidance`/`use_decoder`/`lambda_reconstruction`; `--conditions` sets the ACDC conditions.

| Config | Used by |
|---|---|
| `configs/cityscapes_foggy.yml` | Stage 1, all three variants |
| `configs/cityscapes_clear.yml` | clear-Cityscapes backbone pretraining |
| `configs/acdc.yml` | Stage 2, Uber and per-condition Experts |

Custom runs:

```bash
scripts/train.sh <config> <output_dir> [backbone] [n_gpus] [extra args]
```

## Evaluation

```bash
# Foggy Cityscapes
python evaluation/eval_cs_foggy.py   --exp_dir output/cityscapes/bisenetv2_guided

# ACDC
python evaluation/eval_acdc.py   --exp_dir output/acdc/bisenetv2_norecon_uber --split val

# Zero-shot
scripts/eval_zeroshot.sh output/acdc/bisenetv2_norecon_uber/best.pth bisenetv2 foggy_zurich
scripts/eval_zeroshot.sh output/acdc/bisenetv2_norecon_uber/best.pth bisenetv2 foggy_driving
```

## Results

### Foggy Cityscapes

Validation mIoU (%).

| Model | Clear | Light | Medium | Dense | Drop |
|---|---:|---:|---:|---:|---:|
| BiSeNetV2 | 69.1 | 60.9 | 58.8 | 54.2 | -14.9 |
| DDRNet-23s | 77.8 | 66.1 | 62.2 | 56.1 | -21.7 |
| STDC1-Seg | 71.8 | 67.5 | 62.5 | 56.1 | -15.7 |
| FreD-ViT (Swin-B) | 82.8 | 79.5 | 78.0 | 74.8 | -8.0 |
| **LiDSeg (BiSeNetV2)** | 76.7 | 76.3 | 75.7 | **74.2** | **-2.5** |
| **LiDSeg (DDRNet-23s)** | 76.9 | 76.2 | 75.6 | **74.1** | **-2.8** |
| **LiDSeg (STDC1-Seg)** | 76.9 | 76.2 | 75.5 | **74.3** | **-2.6** |
| **LiDSeg (PIDNet-L)** | 76.8 | 76.2 | 75.5 | **74.2** | **-2.6** |

### ACDC

Test mIoU (%).

| Model | Protocol | Fog | Night | Rain | Snow | Pooled All |
|---|---|---:|---:|---:|---:|---:|
| DeepLabV3+ | Supervised | 69.1 | 60.9 | 74.1 | 69.6 | 70.0 |
| HRNet-W48 | Supervised | 74.7 | 65.3 | 77.7 | 76.3 | 75.0 |
| MIC | UDA | – | – | – | – | 70.4 |
| WA2Net | UDA | – | – | – | – | 73.8 |
| **LiDSeg (PIDNet-L)** | Sup. fine-tune | 73.99 | 64.58 | 78.58 | 77.18 | **75.07** |

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

No datasets or pretrained weights are redistributed.

## Acknowledgements

This work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 101169378 (TIRAMISU Doctoral Network), by the European Union's Horizon 2020 research and innovation programme under grant agreement No. 739551 (KIOS CoE), and from the Government of the Republic of Cyprus through the Cyprus Deputy Ministry of Research, Innovation and Digital Policy.

LiDSeg builds on [BiSeNetV2](https://github.com/CoinCheung/BiSeNet), [DDRNet](https://github.com/ydhongHIT/DDRNet), [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg), [PIDNet](https://github.com/XuJiacong/PIDNet), [SCTNet](https://github.com/xzz777/SCTNet), [panoptic-deeplab](https://github.com/bowenc0221/panoptic-deeplab), [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), [DehazeFormer](https://github.com/IDKiro/DehazeFormer), [Restormer](https://github.com/swz30/Restormer), and [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).
