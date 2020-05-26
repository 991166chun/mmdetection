import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def diou_loss(pred, target, eps=1e-6):
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
  ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

  loss = -ious.log()
  return loss


@LOSSES.register_module()
class DIoULoss(nn.Module):

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