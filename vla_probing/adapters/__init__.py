"""VLA model adapters."""

from vla_probing.adapters.openvla import OpenVLAAdapter
from vla_probing.adapters.openvla_oft import OpenVLAOFTAdapter
from vla_probing.adapters.pi0 import Pi0Adapter

# Cosmos Policy requires CUDA — import is conditional to avoid
# breaking MPS-only environments
try:
    from vla_probing.adapters.cosmos_policy import CosmosPolicyAdapter
except ImportError:
    CosmosPolicyAdapter = None

__all__ = ["OpenVLAAdapter", "OpenVLAOFTAdapter", "Pi0Adapter", "CosmosPolicyAdapter"]
