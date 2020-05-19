'''
000
1. 修改 mmdetection/mmdet/core/evaluation 裡面 class_names.py 的 voc_classes

2. 修改 mmdetection/mmdet/datasets/voc.py 裡的類別   [如果只有一類面要加逗號]) 

3. 修改 config.py
	
	3.1 num_classes 要改 (不用另外+1了，直接輸入要train的數量)
 
	3.2  config.py  裡 dataset_type, dataroot, ann_file, img_prefix

	lr set 4 GPU * 2 img/gpu = 8 img in mb --> lr = 0.01
	       16    * 4         = 64              lr = 0.08
	       1     * 2         = 2               lr = 0.0025

	3.3 gradient clip (避免loss爆炸，可加可不加)

	optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

	3.4 ****重要**** 有直的圖片cv2會讀成直的 要加UNCHANGE參數

	dict(type='LoadImageFromFile', color_type='unchanged')



python3 tools/train.py ${CONFIG_FILE} --work_dir ${YOUR_WORK_DIR} [optional arguments]

# single-gpu testing
python3 tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# Plot the classification and regression loss of some run, and save the figure to a pdf.
python3 tools/analyze_logs.py plot_curve log.json --title ${TITLE} --keys loss_cls loss_reg --legend loss_cls loss_reg --out losses.pdf

# Compare the bbox mAP of two runs in the same figure.
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
'''
