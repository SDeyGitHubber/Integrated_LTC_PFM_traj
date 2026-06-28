"""
================================================================================
UTILS MODULE - Custom Neural Network Components
================================================================================

This module provides tweakable implementations of:
1. CfC (Closed-form Continuous-time) cells and wrappers
2. LTC (Liquid Time-Constant) cells and wrappers

These implementations provide full access to the internal ODE dynamics,
allowing for experimentation and modification of the core equations.

Usage:
------
```python
from utils.cfc_cell import CfCCell, CfC
from utils.ltc_cell import LTCCell, LTC, FullyConnectedWiring

# Create a CfC cell with custom parameters
cell = CfCCell(
    input_size=64,
    hidden_size=64,
    mode="default",
    backbone_activation="lecun_tanh"
)

# Create an LTC cell with ODE integration
ltc_cell = LTCCell(
    wiring=FullyConnectedWiring(64),
    in_features=32,
    ode_unfolds=6
)
```
================================================================================
"""

from .cfc_cell import CfCCell, CfC, LeCunTanh
from .ltc_cell import LTCCell, LTC, FullyConnectedWiring

__all__ = [
    'CfCCell',
    'CfC',
    'LeCunTanh',
    'LTCCell',
    'LTC', 
    'FullyConnectedWiring'
]
