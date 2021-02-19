import pickle as pkl 
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mmdet.core import BboxOverlaps2D, bbox_overlaps

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], **cbar_kw)
    cbar.mappable.set_clim(0,1)
    cbar.ax.set_yticklabels(['0','20','40','60','80','100'])
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def valfmt(x, pos):

    x = x*100

    if x < 0.1:
        return "0"
    
    return '{:.1f}'.format(x)


def get_iou(pr, gt, iou_thr=0.5):

    pr_box = t.FloatTensor(pr)[:, :-1]
    pr_cls = pr[:, -1].astype(int).tolist()
    
    gt_box = t.FloatTensor(gt)[:, 1:]
    gt_cls = gt[:, 0].tolist()

    # print((pr_box, gt_box))
    ious = bbox_overlaps(pr_box, gt_box)    
    v, idx = t.max(ious, dim=1)
    # print(v)
    idx[v < iou_thr] = -1

    true_cls = []
    for id in idx.tolist():
        if id == -1:
            true_cls.append(-1)
        else:
            true_cls.append(int(gt_cls[id]))

    # print(pr_cls , true_cls)
    return zip(pr_cls , true_cls)

def add_id(count_table, ids):
    ids = list(ids)

    for id in ids:
        # print(id)
        count_table[id] +=1

    return count_table

def sc_thr(det, score_thr=0.05):

    n_cls = len(det)
    cc = np.zeros(n_cls) # class count
    out = np.empty((0,5))
    for i, d in enumerate(det):
        
        if d.size == 0:
            continue
        
        num = d.shape[0]
        box = d[d[:,4] > score_thr]

        cc[i] += box.shape[0]

        box[:,4] = i
        out = np.vstack((out, box))

    return out, cc



if __name__ == '__main__':

    boxfile = 'xmTool/result_cascade.pkl'

    with open(boxfile, 'rb') as f:
        boxdict =pkl.load(f)

    gtfile = 'xmTool/gt_origin.pkl'

    with open(gtfile, 'rb') as f1:
        gtdict =pkl.load(f1)

    gts = gtdict['test']

    n_cls = len(boxdict[0])-1

    class_total = np.zeros(n_cls)
    count_table = np.zeros((n_cls, n_cls+1))
    print(boxdict[0].__len__())
    for (i, gt) in enumerate(gts):

        det  = boxdict[i][:-1]
        det, cc = sc_thr(det, score_thr=0.7)

        class_total += cc
        if det.size > 0:
            id = get_iou(det, gt, iou_thr=0.5)
            
            count_table = add_id(count_table, id)
    print(class_total)
    print(count_table)
    class_total.resize(class_total.size, 1)
    cfmatrix = count_table/class_total

    print(cfmatrix)

    fig, ax = plt.subplots(figsize=(8,6))
    pr_CLASSES = ['brown', 'blister', 'algal', 
                'fungi', 'miner', 'thrips', 
                'mos_e', 'mos_l',
                 'tor_w', 'tor_r', 'flush', 'ori_w']
    gt_CLASSES = ['brown', 'blister', 'algal', 
                'fungi', 'miner', 'thrips', 
                'mos_e', 'mos_l',
                 'tor_w', 'tor_r', 'flush', 'ori_w', 'back']
    im, cbar = heatmap(cfmatrix.T, gt_CLASSES, pr_CLASSES, ax=ax,
                    cmap="YlGn", cbarlabel="accuracy")

    texts = annotate_heatmap(im, valfmt=valfmt)

    fig.tight_layout()

    plt.savefig('xmTool/cnofusion_mat1.png', dpi=200)
    plt.close()
        
    




    
    
