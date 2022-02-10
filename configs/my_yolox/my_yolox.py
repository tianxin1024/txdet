
img_scale = (640, 640)

# model settings
model = dict(
    type='tx_YOLOX',
    input_size = img_scale,
    random_size_range = (15, 25),
    random_size_interval=10,
    backbone = dict(type='tx_CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    nect=dict(
        type='tx_YOLOXPAFPN',
        in_channels = [128, 256, 512],
        out_channels = 128,
        num_csp_blocks=1),
    bbox_head = dict(
        type='tx_YOLOXHead', num_classes=80, in_channels=128, feat_channels=128),

    train_cfg = dict(assigner=dict(type='tx_SimOTAAssigner', center_radius=2.5)),
    test_cfg = dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)
