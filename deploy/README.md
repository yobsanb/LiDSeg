# LiDSeg PIDNet-L on ACDC: model for fault injection

In a source checkout, `deploy/` contains templates for the locally generated handoff bundle.
Build it from the repository root with `GPU=0 scripts/make_fault_injection_bundle.sh` after
step 2. Then run the instructions below inside `output/handoff/lidseg_pidnetl_acdc/`.
The weights, dataset and reference files described below belong to that generated folder and
are excluded from the GitHub repository.

A trained semantic-segmentation network for driving scenes in adverse weather (fog, night,
rain, snow). This folder is self-contained: the model, the evaluation data and the scripts to
reproduce its fault-free ("golden") accuracy on the ACDC validation set.

The model is one weights file, `lidseg_pidnetl_acdc.pth`: our own trained model, the single
checkpoint chosen for ACDC (validation mIoU 74.49). Nothing else has to be downloaded, and the
only software needed is Python 3.9+ with PyTorch, NumPy and Pillow.

## Quick start: reproduce the result

```bash
cd lidseg_pidnetl_acdc
bash run_eval.sh            # checks the files, evaluates, compares with the reference
```

If PyTorch, NumPy or Pillow are missing, run `bash setup_env.sh` once first: it creates a
virtual environment in `.venv`, which `run_eval.sh` then uses. Expected output:

```
[1/2] checking files against SHA256SUMS
      all files OK
[2/2] evaluating on ACDC val
  fog: mIoU  79.54  (100 images)
night: mIoU  55.26  (106 images)
 rain: mIoU  75.47  (100 images)
 snow: mIoU  79.19  (100 images)
  all: mIoU  74.49  (406 images, 53.4 ms/forward)
-> golden_acdc_val.json
comparison with reference_acdc_val.json:
    fog:  79.54 vs  79.54 (+0.00)
  night:  55.26 vs  55.26 (+0.00)
   rain:  75.47 vs  75.47 (+0.00)
   snow:  79.19 vs  79.19 (+0.00)
    all:  74.49 vs  74.49 (+0.00)
identical label maps: 406 of 406 (all on the reference GPU; elsewhere a few pixels may differ)
PASS: every mIoU is within 0.05 of the reference
```

`PASS` means every mIoU is within 0.05 of `reference_acdc_val.json`; the run is saved to
`golden_acdc_val.json`. It takes about a minute on a GPU. On a CPU it is much slower: about
0.8 s per image with 16 free cores (about 6 minutes), and longer on a busy machine. Choose the
device with `CUDA_VISIBLE_DEVICES=1 bash run_eval.sh` or `DEVICE=cpu bash run_eval.sh`.

## Contents

```
lidseg_pidnetl_acdc/
├── README.md
├── run_eval.sh                 one-step reproduction (checksums, evaluation, comparison)
├── setup_env.sh                creates .venv with requirements.txt, if packages are missing
├── requirements.txt            torch, numpy, Pillow
├── SHA256SUMS                  SHA-256 of every other file
├── lidseg_pidnetl_acdc.pth     trained weights (fp32, 159 MB)
├── lidseg_infer.py             load_model(), load_image(); writes the layer table
├── eval_acdc.py                evaluation on ACDC val, writes a JSON report
├── reference_acdc_val.json     our golden run: the results to reproduce
├── layers.csv                  every module: type, parameters, calls per forward pass, output shape
├── models/                     network definition, unchanged from the training repository
│   ├── lidseg.py               LiDSegModel: guidance encoder and fusion around the backbone
│   ├── dehazeformer.py         guidance encoder (and the reconstruction decoder, not used here)
│   ├── lgf.py                  Lightweight Gated Fusion
│   └── segmodels/
│       ├── pidnet.py           PIDNet backbone (this model uses PIDNet-L)
│       ├── feature_registry.py where the fusion taps into the backbone
│       └── model_registry.py   backbone name to class
└── data/ACDC/                  ACDC validation split (745 MB)
    ├── README.md, License.pdf  the dataset's own description and license
    ├── rgb_anon/{fog,night,rain,snow}/val/<seq>/<frame>_rgb_anon.png
    └── gt/{fog,night,rain,snow}/val/<seq>/<frame>_gt_labelTrainIds.png
```

`models/` and `models/segmodels/` also hold an `__init__.py` each.

## The model

| | |
|---|---|
| Task | Semantic segmentation into the 19 Cityscapes classes |
| Network | PIDNet-L backbone, plus a lightweight guidance encoder (gated linear attention) whose features are fused into the backbone at 1/8, 1/16 and 1/32 resolution by Lightweight Gated Fusion (LGF) |
| Input | `[N, 3, H, W]` float32 RGB, normalized as in "Evaluation protocol"; ACDC images are 1080 x 1920 |
| Output | `[N, 19, H, W]` float32 logits at input resolution; `argmax` over dim 1 gives the class |
| Precision | fp32 weights and inference; not quantized; BatchNorm not folded into the convolutions |
| Weights | `lidseg_pidnetl_acdc.pth`: a dict with `state_dict` (751 tensors), `architecture` and `meta`; loads with `torch.load(..., weights_only=True)` |
| Training | Trained by us on Foggy Cityscapes (synthetic fog of increasing density), then on the ACDC training split (1600 images, all four conditions); the checkpoint with the best ACDC val mIoU was chosen |

| Part | Parameters | Used at inference |
|---|---:|---:|
| `backbone` (PIDNet-L) | 37,314,183 | 36,939,443 |
| `guidance` (guidance encoder) | 1,845,840 | 1,845,840 |
| `lgf` (fusion at 3 scales) | 386,307 | 386,307 |
| **total** | **39,546,330** | **39,171,590** |

By module type: 200 `Conv2d` hold 39,488,023 parameters (99.85 %), 95 `BatchNorm2d` hold 53,184,
32 `ChannelLayerNorm` hold 5,120 and the three LGF modules hold one scalar each. There are no
`nn.Linear` layers: the attention projections are 1x1 convolutions.

Classes by index (Cityscapes trainId): 0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole,
6 traffic light, 7 traffic sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider, 13 car,
14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle.

## Dataset: ACDC validation split

ACDC, the Adverse Conditions Dataset with Correspondences (C. Sakaridis, D. Dai, L. Van Gool,
ICCV 2021, ETH Zurich), has driving images in fog, night, rain and snow with pixel-level labels
for the 19 Cityscapes classes. Its test labels are withheld for the online benchmark, so the
labelled **validation split** is the evaluation set: 406 images (fog 100, night 106, rain 100,
snow 100), all 1080 x 1920. The model was not trained on these images, but they were used to
select the checkpoint, so the scores are this model's reference, not an unbiased test estimate.

`data/ACDC/` holds exactly the files the evaluation reads, in the dataset's own layout:

- images: `rgb_anon/<condition>/val/<seq>/<frame>_rgb_anon.png`, the anonymized RGB images;
- labels: `gt/<condition>/val/<seq>/<frame>_gt_labelTrainIds.png`, one trainId (0-18) per pixel,
  255 where a pixel is not labelled.

The rest of ACDC is not used and not included: the training and test splits, the
normal-condition reference images (`*_ref`), the invalid masks (`*_invIds.png`) and the other
label encodings (`labelIds`, `labelColor`). ACDC belongs to ETH Zurich and comes with its own
license, `data/ACDC/License.pdf` (non-commercial research use; cite the ACDC paper). To evaluate
on another copy of ACDC with the same layout, set `ACDC_ROOT=<folder>` for `run_eval.sh` or
pass `--acdc_root <folder>` to `eval_acdc.py`.

## Evaluation protocol

This is exactly what `eval_acdc.py` does:

1. **Order.** Conditions fog, night, rain, snow; within a condition, images sorted by path.
   This is the order of `images` in the report.
2. **Preprocessing.** Read as 8-bit RGB, divide by 255, subtract the ImageNet mean
   `(0.485, 0.456, 0.406)` and divide by the std `(0.229, 0.224, 0.225)`, per channel. No
   resizing or cropping.
3. **Inference.** One forward pass per image, batch size 1, in eval mode; fp32 with TF32
   disabled and deterministic cuDNN kernels. No test-time augmentation (no extra scales, no
   flipping, no sliding window). Inside the model, the input is padded by edge replication to a
   multiple of 32 (1080 to 1088 rows) and the output is cropped back; the network upsamples its
   1/8-resolution logits bilinearly to full resolution.
4. **Prediction.** Per pixel, the class with the largest logit.
5. **Scores.** Pixels labelled 255 are skipped. For each condition a 19 x 19 confusion matrix is
   summed over its images; per class, IoU = TP / (TP + FP + FN), and mIoU is the mean over the 19
   classes. **all** sums the confusion matrices of all 406 images. It is not the average of the
   four condition mIoUs, which would be 72.37.
6. **Per image** (for classifying fault outcomes): the pixel accuracy, the mIoU over the classes
   present in the label or the prediction, and `pred_md5`, the MD5 of the predicted label map as
   1080 x 1920 uint8.

## Reference results

ACDC val mIoU (%), one forward pass, fp32, from `reference_acdc_val.json`:

| fog | night | rain | snow | all (406 images) |
|---:|---:|---:|---:|---:|
| 79.54 | 55.26 | 75.47 | 79.19 | **74.49** |

Made with Python 3.12.4, torch 2.9.0+cu128, numpy 1.26.4 and Pillow 10.3.0 on a Tesla
V100-SXM2-32GB: 53 ms per forward pass and 0.7 GB peak GPU memory.

**Determinism.** Two runs on the same machine give bit-identical predictions (all 406
`pred_md5` match). Across devices they can differ slightly: run on the CPU, 311 of the 406 label
maps matched the V100 bit for bit, the others differed in a few pixels (per-image mIoU within
0.0004), and every condition's mIoU matched to within 0.0001. Record a golden run on the machine
and software used for the campaign and compare faulty runs against that one. A faulty image with the same `pred_md5` is masked; otherwise its
per-image mIoU and pixel accuracy show how much the output degraded.

## Running the evaluation directly

`run_eval.sh` calls `eval_acdc.py`, which can also be run on its own (defaults: the weights
and `data/ACDC/` in this folder, output `golden_acdc_val.json`):

```bash
python eval_acdc.py --reference reference_acdc_val.json       # same as run_eval.sh, step 2
python eval_acdc.py --conditions night --limit 10             # first 10 night images only
python eval_acdc.py --save_pred predictions                   # also write label-map PNGs
python eval_acdc.py --device cpu --out_json golden_cpu.json   # on the CPU
```

`python lidseg_infer.py --layers layers.csv` rewrites the layer table and prints the parameter
counts.

## Using the model in your own code

```python
import torch
from lidseg_infer import load_model, load_image

model = load_model("lidseg_pidnetl_acdc.pth", device="cuda")    # eval mode
x = load_image("data/ACDC/rgb_anon/night/val/GOPR0351/GOPR0351_frame_000033_rgb_anon.png").to("cuda")
with torch.no_grad():
    logits = model(x)["logits"]      # [1, 19, 1080, 1920]; model.predict(x) returns the same tensor
pred = logits.argmax(1)              # [1, 1080, 1920] class indices
```

`model(x)` returns a dict. For a tool that expects the forward pass to return a tensor, wrap
it (module names then gain a `net.` prefix):

```python
class Logits(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["logits"]
```

## Notes for fault injection

`layers.csv` has one row per module that owns parameters or has no submodules: its name, type,
own parameter count, how many times it runs in one forward pass (`calls`) and the shape of its
first output for a 1080 x 1920 image.

1. **Training-only heads.** `backbone.p_head`, `backbone.p_cls_seg`, `backbone.d_head` and
   `backbone.d_cls_seg` (374,740 parameters, `calls` = 0) are PIDNet's auxiliary training heads
   and never run at inference. A fault in their weights is always masked, so leave them out of
   the sampling space or report them separately.
2. **The fusion uses forward hooks.** The model's own forward hooks replace the outputs of
   `backbone.p_branch_layers.2` (1/8), `backbone.i_branch_layers.0` (1/16) and
   `backbone.i_branch_layers.1` (1/32) with the fused features `lgf.s8/s16/s32(f, g)`. The
   guidance encoder runs first, on the same input. PyTorch runs hooks in registration order,
   so a hook you add to one of those three modules sees the fused output.
3. **Activations.** Most ReLUs are functional calls, not modules, and `backbone.relu` is one
   module used 6 times per forward pass (a hook on it fires 6 times). To inject into
   activations, hook the outputs of the `Conv2d` or `BatchNorm2d` modules.
4. **Shapes.** Because of the padding (protocol step 3) the network runs on 1088 x 1920, so the
   shapes in `layers.csv` are, e.g., 544 x 960 after the stem and 136 x 240 at 1/8.
5. **Eval mode.** BatchNorm uses its stored running statistics. `model.train()` would change the
   computation and make the backbone also return the auxiliary heads' outputs.
6. **Numeric settings.** Use the same settings for golden and faulty runs (fp32, TF32 off,
   deterministic cuDNN, as `eval_acdc.py` sets them), so that any difference comes from the fault.

## Source

LiDSeg: Linear DehazeFormer Guidance for Efficient Semantic Segmentation in Adverse Weather
Conditions. Yobsan Bayisa Leta, Christos Kyrkou, Maria K. Michael, Theocharis Theocharides,
University of Cyprus. Please do not redistribute these weights outside the collaboration without
asking the authors first.
