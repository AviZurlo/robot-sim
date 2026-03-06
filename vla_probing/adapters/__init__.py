"""VLA model adapters."""

from vla_probing.adapters.openvla import OpenVLAAdapter
from vla_probing.adapters.openvla_oft import OpenVLAOFTAdapter
from vla_probing.adapters.pi0 import Pi0Adapter

# CUDA-only models — conditional imports
try:
    from vla_probing.adapters.cosmos_policy import CosmosPolicyAdapter
except ImportError:
    CosmosPolicyAdapter = None

try:
    from vla_probing.adapters.groot import GR00TAdapter
except ImportError:
    GR00TAdapter = None

__all__ = ["OpenVLAAdapter", "OpenVLAOFTAdapter", "Pi0Adapter", "CosmosPolicyAdapter", "GR00TAdapter"]
