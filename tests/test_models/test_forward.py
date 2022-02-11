import copy
import torch
from os.path import dirname, exists, join, abspath

import sys
sys.path.append("/home/tianxin/2022/myself/txdet/")

from models import *

import ipdb



def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(abspath(__file__))))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def test_models_forward():
    model = _get_detector_cfg("/home/tianxin/2022/myself/txdet/configs/my_yolox/my_yolox.py")
    # model = _get_detector_cfg("/home/tianxin/2022/myself/txdet/configs/yolox/yolox_s_8x8_300e_coco.py")
    # model = _get_detector_cfg("/home/tianxin/2022/myself/txdet/configs/my_rpn/my_rpn_r50_fpn_coco.py")
    print(model)

    from mmdet.models import build_detector
    detector = build_detector(model)
    # ipdb.set_trace()
    print(detector)

    # input_shape = (1, 3, 640, 640)
    # input = torch.randn(1, 3, 640, 640)
    # output = detector(input)


if __name__ == "__main__":
    test_models_forward()
