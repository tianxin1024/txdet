

from .yolox.yolox import tx_YOLOX
from .yolox.yolox_backbone import tx_CSPDarknet
from .yolox.yolox_pafpn import tx_YOLOXPAFPN
from .yolox.yolox_head import tx_YOLOXHead


__all__ = ['tx_YOLOX', 'tx_CSPDarknet', 'tx_YOLOXPAFPN', 'tx_YOLOXHead']

print("----" * 20)
