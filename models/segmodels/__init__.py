from .model_registry import model_hub, aux_models, register_model

from . import bisenetv2        # BiSeNetV2       (IJCV 2021)
from . import ddrnet_official  # DDRNet-23-slim  (CVPR 2021)
from . import stdc_official    # STDC1/2-Seg     (CVPR 2021)
from . import pidnet           # PIDNet-S/M/L    (CVPR 2023)
from . import sctnet           # SCTNet-S/B      (AAAI 2024)

__all__ = ["model_hub", "aux_models", "register_model"]
