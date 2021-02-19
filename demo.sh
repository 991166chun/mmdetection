CUDA_VISIBLE_DEVICES=0 python3 -u tools/test.py configs/_xm/cascade_x101_fpn.py  work_dirs/cascade_res50_fpn/crcnn_101x_final.pth --show-dir output/Cascade

CUDA_VISIBLE_DEVICES=0 python3 -u tools/test.py configs/_xm/faster_rcnn_r50_fpn_giou.py  work_dirs/faster_rcnn_r50_fpn_giou/epoch_8.pth --show-dir output/Giou

CUDA_VISIBLE_DEVICES=0 python3 -u tools/test.py configs/_xm/faster_res50_fpn_voc.py  work_dirs/faster_res50_fpn_voc/frcnn_final.pth --show-dir output/frcnn
