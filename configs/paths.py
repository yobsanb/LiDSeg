"""Dataset and output roots, read from configs/paths.env."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

_HERE = Path(__file__).resolve().parent
_ENV_FILE = _HERE / "paths.env"
_EXAMPLE_FILE = _HERE / "paths.env.example"

_REQUIRED = ("CITYSCAPES_ROOT", "ACDC_ROOT")

_OPTIONAL = ("CITYSCAPES_CLEAR_ROOT", "FOGGY_DRIVING_ROOT", "FOGGY_ZURICH_ROOT",
             "LIDSEG_REPO", "LIDSEG_OUTPUT")

_CACHE: Optional[Dict[str, str]] = None


def _parse_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load(force: bool = False) -> Dict[str, str]:
    """Parse configs/paths.env once and return the resolved key/value map."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    if not _ENV_FILE.exists():
        raise FileNotFoundError(
            f"{_ENV_FILE} not found.\n"
            f"Copy the template and point it at your datasets:\n"
            f"    cp {_EXAMPLE_FILE.relative_to(_HERE.parent)} "
            f"{_ENV_FILE.relative_to(_HERE.parent)}"
        )

    env = _parse_env(_ENV_FILE)
    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        raise KeyError(f"{_ENV_FILE} is missing required keys: {missing}")

    env.setdefault("CITYSCAPES_CLEAR_ROOT", env["CITYSCAPES_ROOT"])
    env.setdefault("FOGGY_DRIVING_ROOT", "")
    env.setdefault("FOGGY_ZURICH_ROOT", "")
    env.setdefault("LIDSEG_REPO", str(_HERE.parent))
    env.setdefault("LIDSEG_OUTPUT", str(Path(env["LIDSEG_REPO"]) / "output"))

    _CACHE = env
    return _CACHE


class _LazyPaths:
    """``paths["KEY"]`` and ``paths.KEY`` access, resolved on first use."""

    def __getitem__(self, key: str) -> str:
        return load()[key]

    def __getattr__(self, key: str) -> str:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return load()[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def get(self, key: str, default=None):
        return load().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in load()

    def __iter__(self) -> Iterator[str]:
        return iter(load())

    def keys(self):
        return load().keys()

    def items(self):
        return load().items()

    def __repr__(self) -> str:
        state = "unresolved" if _CACHE is None else f"{len(_CACHE)} keys"
        return f"<LiDSeg paths ({state}) from {_ENV_FILE}>"


paths = _LazyPaths()


def __getattr__(name: str) -> str:
    """Module-level ``configs.paths.CITYSCAPES_ROOT`` style access (PEP 562)."""
    if name in _REQUIRED or name in _OPTIONAL:
        return load()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def assert_exists() -> None:
    """Validate required dataset roots."""
    env = load()
    for key in _REQUIRED:
        if not Path(env[key]).exists():
            raise FileNotFoundError(
                f"{key}={env[key]} (from {_ENV_FILE}) does not exist")


__all__ = [
    "CITYSCAPES_ROOT",
    "CITYSCAPES_CLEAR_ROOT",
    "ACDC_ROOT",
    "FOGGY_DRIVING_ROOT",
    "FOGGY_ZURICH_ROOT",
    "LIDSEG_REPO",
    "LIDSEG_OUTPUT",
    "assert_exists",
    "load",
    "paths",
]
