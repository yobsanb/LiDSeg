# LiDSeg PIDNet-L inference

Download `lidseg_pidnetl_uber_tta_model.pth` into `pretrained/` from the
[official model release](https://github.com/yobsanb/LiDSeg/releases/tag/pidnetl-acdc).
Run the examples below from the repository root.

## Predict

```python
from deploy.lidseg_infer import load_image, load_model

model = load_model("pretrained/lidseg_pidnetl_uber_tta_model.pth", device="cuda")
image = load_image("image.png").to("cuda")
prediction = model.predict(image).argmax(1)
```

Predictions use the 19 Cityscapes trainIds. Set `device="cpu"` and move the
image to `"cpu"` for CPU inference.

## Evaluate

Obtain ACDC separately and provide its local root:

```bash
python deploy/eval_acdc.py --weights pretrained/lidseg_pidnetl_uber_tta_model.pth \
    --acdc_root /path/to/ACDC --out_json output/acdc_val.json
```

Or use the wrapper:

```bash
WEIGHTS=pretrained/lidseg_pidnetl_uber_tta_model.pth ACDC_ROOT=/path/to/ACDC \
    OUT_JSON=output/acdc_val.json bash deploy/run_eval.sh
```

Evaluation uses native resolution, single-scale fp32 inference and pooled mIoU.
`--reference report.json` optionally compares an existing evaluation report.
