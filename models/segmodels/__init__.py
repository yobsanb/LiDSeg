from .model_registry import model_hub, aux_models, register_model

from . import pidnet  # noqa: F401  (registers PIDNet-S/M/L, CVPR 2023)

__all__ = ["model_hub", "aux_models", "register_model"]
