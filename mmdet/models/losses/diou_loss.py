import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def mdiou_loss(pred, target, eps=1e-6):
  """ D-IoU loss

  Computing the distance-IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
  """
  #  center point
  x_p = (pred[:, 2] + pred[:, 0]) / 2
  y_p = (pred[:, 3] + pred[:, 1]) / 2
  x_g = (target[:, 2] + target[:, 0]) / 2
  y_g = (target[:, 3] + target[:, 1]) / 2


  # overlap
  i_x1y1 = torch.max(pred[:, :2], target[:, :2])
  i_x2y2 = torch.min(pred[:, 2:], target[:, 2:])
  i_wh = (i_x2y2 - i_x1y1).clamp(min=0)
  overlap = i_wh[:, 0] * i_wh[:, 1]

  # union
  ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
  ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
  union = ap + ag - overlap + eps

  # IoU
  ious = overlap / union

  # enclose box each (n, 2)
  c_x1y1 = torch.min(pred[:, :2], target[:, :2])
  c_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
  
  # diagonal length ^2 of enclose box
  c = ((c_x2y2[:, 0] - c_x1y1[:, 0]) ** 2) + ((c_x2y2[:, 1] - c_x1y1[:, 1]) ** 2) + eps

  # center distance ^2
  d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

  # u = d / c
  r = d / c

  loss = 1 - ious + r

  return loss
  

@LOSSES.register_module()
class mDIoULoss(nn.Module):

  def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
    super(DIoULoss, self).__init__()
    self.eps = eps
    self.reduction = reduction
    self.loss_weight = loss_weight

  def forward(self,
              pred,
              target,
              weight=None,
              avg_factor=None,
              reduction_override=None,
              **kwargs):

    if weight is not None and not torch.any(weight > 0):
      return (pred * weight).sum() # 0
    assert reduction_override in (None, 'none', 'mean', 'sum')
    reduction = (reduction_override if reduction_override 
                                        else self.reduction)
    if weight is not None and weight.dim() > 1:
      assert weight.shape == pred.shape
      weight = weight.mean(-1)

    loss = self.loss_weight * diou_loss(
        pred, target, weight,
        eps=self.eps,
        reduction=reduction,
        avg_factor=avg_factor,
        **kwargs
    )
    return loss