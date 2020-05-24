_base_ = './faster_res50_fpn_voc.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))))
