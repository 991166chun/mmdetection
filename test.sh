CUDA_VISIBLE_DEVICES=1 python3 -u tools/test.py configs/_xm/faster_rcnn_r50_fpn.py  work_dirs/faster_res50_fpn_voc/frcnn_final.pth  --eval mAP --show-dir output/fig12/frcnn
# python3 tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
# CUDA_VISIBLE_DEVICES=1 python3 -u tools/test.py configs/_xm/faster_rcnn_r50_fpn_diou.py  work_dirs/faster_rcnn_r50_fpn_diou/epoch_4.pth --get_roi --eval mAP
