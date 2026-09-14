"""HE/thor — Liberate/THOR complex-path adapters (Branch A)."""
from .path_setup import ensure_thor_path, diagnose
from .geometry import ThorComplexGeometry, bert_base_complex, real_vs_complex_table
from .bootstrap_hook import BootstrapHook
from .engine import (
    THOR_DEFAULT_PARAMS,
    SMOKE_PARAMS,
    create_engine,
    create_keys,
    load_thor_keys,
    status,
)

ensure_thor_path()

__all__ = [
    "ensure_thor_path",
    "diagnose",
    "status",
    "ThorComplexGeometry",
    "bert_base_complex",
    "real_vs_complex_table",
    "BootstrapHook",
    "THOR_DEFAULT_PARAMS",
    "SMOKE_PARAMS",
    "create_engine",
    "create_keys",
    "load_thor_keys",
]
