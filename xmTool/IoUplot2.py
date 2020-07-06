import sys
import pickle as pkl 
from mmdet.core import BboxOverlaps2D, bbox_overlaps
import torch as t

import itertools
from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler


def get_iou(pr,gt):
    if pr.shape[1] == 5:
        pr = t.FloatTensor(pr)[:,1:]
    else:
        pr = t.FloatTensor(pr)

    gt = t.FloatTensor(gt)[:,1:]
    ious = bbox_overlaps(pr,gt)
    v, idx = t.max(ious, dim=1)

    return v.view(-1)
    

def y_fmt(tick_val, pos):
    if tick_val > 1000000:
        val = int(tick_val)/1000000
        return '{:d} M'.format(int(val))
    elif tick_val > 1000:
        val = int(tick_val) / 1000
        return '{:d} k'.format(int(val))
    else:
        return tick_val

def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Extra kwargs are passed through to `fill_between`

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : scalar or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)
    if orientation not in 'hv':
        raise ValueError("orientation must be in {{'h', 'v'}} "
                         "not {o}".format(o=orientation))

    kwargs.setdefault('step', 'post')
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError('Must provide one more bin edge than value not: '
                         'len(edges): {lb} len(values): {lv}'.format(
                             lb=len(edges), lv=len(values)))

    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)

    values = np.r_[values, values[-1]]
    bottoms = np.r_[bottoms, bottoms[-1]]
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")

def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (N, M) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, optional
        The initial positions of the bottoms, defaults to 0

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If stacked_data is a mapping, and labels is None, default to the keys
        (which may come out in a random order).

        If stacked_data is a mapping and labels is given then only
        the columns listed by be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra kwargs to pass through to the plotting function.  This
        will be the same for all calls to the plotting function and will
        over-ride the values in cycle.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = 'dflt set {n}'.format(n=j)
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        
        bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    # ax.legend(fontsize=10)
    return arts




def draw(ax, tag, stack_data):
    # set up histogram function to fixed bins
    edges = np.linspace(0.1, 1, 20, endpoint=True)
    hist_func = partial(np.histogram, bins=edges)

    # set up style cycles
    color_cycle = cycler(facecolor=['#f0fc03','#0356fc'])
    label_cycle = cycler(label=tag)
    alpha_cycle = cycler(alpha=[0.7, 0.3])
    # Fixing random state for reproducibility
    
    dict_data = OrderedDict(zip((c['label'] for c in label_cycle), stack_data))

    arts = stack_hist(ax, stack_data, color_cycle + label_cycle + alpha_cycle,
                hist_func=hist_func)
    
    ax.set_xlim([0.1, 1])

def one_hist(ax, boxfile):

    with open(boxfile, 'rb') as f:
        boxdict =pkl.load(f)

    gtfile = 'xmTool/gt_test.pkl'

    with open(gtfile, 'rb') as f1:
        gtdict =pkl.load(f1)

    tag1 = ['rpn', 'reg']

    pr = boxdict[tag1[0]]
    pr2 = boxdict[tag1[1]]
    
    gt = gtdict['test']
    
    count = t.tensor([])
    count2 = t.tensor([])

    for i in range(len(pr)):

        ious = get_iou(pr[i], gt[i])
        count = t.cat((count, ious))

        ious2 = get_iou(pr2[i], gt[i])
        count2 = t.cat((count2, ious2))
    
    count = count.numpy()
    count2 = count2.numpy()
    
    stack_data = np.stack((count,count2))
    
    draw(ax, tag1, stack_data)
    plt.tight_layout()


if __name__ == '__main__':
    if sys.argv[1] == 'iou'
        boxfile1 = 'xmTool/frcnn.pkl'
        boxfile2 = 'xmTool/giou.pkl'
        boxfile3 = 'xmTool/diou.pkl'

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharey=True)
        
        ax1.set_ylabel('counts',fontsize=18)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
        plt.ylim((0,45000))

        ax1.set_title('L1 norm loss',fontsize=18)
        one_hist(ax1, boxfile1)
        plt.tight_layout()

        ax2.set_title('g-IoU loss',fontsize=18)
        one_hist(ax2, boxfile2)
        ax2.set_xlabel('IoU',fontsize=18)
        plt.tight_layout()

        ax3.set_title('d-IoU loss',fontsize=18)
        one_hist(ax3, boxfile3)
        plt.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, -0.19),fontsize=14)
        plt.tight_layout()
        
        plt.savefig('xmTool/3iou' , dpi=200)
        plt.close()

    if argv[1] == 'cas':
        
    
    


