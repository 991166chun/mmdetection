import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmcv.image import  imwrite

color = {   
            'brownblight':[0.75, 0.25, 0],
            'blister':[0.5, 0.5, 0],
            'algal':[0, 0, 0.5],
            'fungi_early':[0.5, 0, 0.5],
            'miner':[0, 0.5, 0.58],
            'thrips':[0.5, 0.5, 0.5],
            'mosquito_early':[0.75, 0, 0],
            'mosquito_late':[0.25, 0, 0],
            'moth':[0.25, 0.5, 0],
            'tortrix':[0.75, 0.5, 0],
            'flushworm':[0.25, 0, 0.5],
            'roller':[0.75, 0, 0.5],
            'other':[0.25, 0.5, 0.5],

            'cls 0':[0.75, 0.5, 0.5],
            'cls 1':[0, 0.25, 0],
            'cls 2':[0.5, 0.25, 0],
            'cls 3':[0, 0.75, 0],
            'cls 4':[0.5, 0.75, 0],
            'cls 5':[0, 0.25, 0.5]
            }
ignoer_list = ['other',]

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    # print(img)
    # img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # bbox_color = color_val(bbox_color)
    # text_color = color_val(text_color)
    
    fs = int(img.shape[0]/100)
    lw = int(img.shape[0]/300)
    fig = plt.figure(frameon=False)
    dpi = 200
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img[:, :, ::-1])

    for bbox, label in zip(bboxes, labels):
        name = class_names[label]
        if name in ignoer_list:
            continue            
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        ax.add_patch(
            plt.Rectangle(left_top,
                          right_bottom[0] - left_top[0],
                          right_bottom[1] - left_top[1],
                          fill=False,
                          edgecolor=tuple(color[name]),
                          linewidth=lw, alpha=0.7))
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f' {bbox[-1]:.02f}'
        ax.text(
                left_top[0], left_top[1] - 2,
                label_text,
                fontsize=fs,
                family='serif',
                bbox=dict(
                    facecolor=tuple(color[name]),
                    alpha=0.6, pad=0, edgecolor='none'),
                color='white')

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        fig.savefig(out_file, dpi=dpi)
        plt.close('all')
        # imwrite(img, out_file)

            