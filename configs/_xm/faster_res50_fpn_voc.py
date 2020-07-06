_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './my_voc.py',
    './my_runtime.py'
]
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    roi_head=dict(bbox_head=dict(num_classes=13))
    )
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35))
# learning policy
# actual epoch = 
lr_config = dict(policy='step',
                warmup='linear',
                warmup_iters=500,
                warmup_ratio=0.001,
                step=[5, 8, 10], gamma=0.2)
# runtime settings  0.01  0.002 0.0004  0.00008
total_epochs = 12  # actual epoch = 12 * 2 = 24
