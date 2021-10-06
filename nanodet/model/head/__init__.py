import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_simota_head import NanoDetSimOTAHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetSimOTAHead":
        return NanoDetSimOTAHead(**head_cfg)
    else:
        raise NotImplementedError
