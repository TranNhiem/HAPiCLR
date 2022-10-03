# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from HAPiCLR.methods.barlow_twins import BarlowTwins
from HAPiCLR.methods.base import BaseMethod
from HAPiCLR.methods.byol import BYOL
from HAPiCLR.methods.deepclusterv2 import DeepClusterV2
from HAPiCLR.methods.dino import DINO
from HAPiCLR.methods.linear import LinearModel
from HAPiCLR.methods.mocov2plus import MoCoV2Plus
from HAPiCLR.methods.nnbyol import NNBYOL
from HAPiCLR.methods.nnclr import NNCLR
from HAPiCLR.methods.nnsiam import NNSiam
from HAPiCLR.methods.ressl import ReSSL
from HAPiCLR.methods.simclr import SimCLR
from HAPiCLR.methods.simsiam import SimSiam
from HAPiCLR.methods.supcon import SupCon
from HAPiCLR.methods.swav import SwAV
from HAPiCLR.methods.vibcreg import VIbCReg
from HAPiCLR.methods.vicreg import VICReg
from HAPiCLR.methods.wmse import WMSE
from HAPiCLR.methods.mncrl import MNCRL
#from solo.methods.hapiclr_simple import HAPiCLR_Unified
from HAPiCLR.methods.hapiclr import HAPiCLR

# from solo.methods.mncrl_edit import MNCRL_edit
from HAPiCLR.methods.mscrl import MSCRL
from HAPiCLR.methods.mocov2plus_MPLCL import MoCoV2Plus_MPLCL
METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    #hapiclr': HAPiCLR_Unified, 
    'hapiclr_simple': HAPiCLR,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
    "mncrl": MNCRL,
    # "mncrl_edit": MNCRL_edit,
    "mscrl": MSCRL,
    "moco_MPLCL": MoCoV2Plus_MPLCL,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
    "mncrl",
    # "mncrl_edit",
    "mscrl",
    "hapiclr",
   # "hapiclr_simple",
    "moco_MPLCL",
]

try:
    from HAPiCLR.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
