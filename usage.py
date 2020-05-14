'''

python3 tools/train.py ${CONFIG_FILE} --work_dir ${YOUR_WORK_DIR} [optional arguments]

# single-gpu testing
python3 tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]




# Plot the classification and regression loss of some run, and save the figure to a pdf.
python3 tools/analyze_logs.py plot_curve log.json --title ${TITLE} --keys loss_cls loss_reg --legend loss_cls loss_reg --out losses.pdf

# Compare the bbox mAP of two runs in the same figure.
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
'''
