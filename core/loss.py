import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class SegmentationLoss(nn.Module):
    """
    soft-Dice loss and soft-IoU loss calculation.
    """

    def __init__(self, loss_type='iou'):
        """
        Macro strategy used for multiclass calculation.
        Args:
            loss_type: iou, dice or both
        """
        super().__init__()
        self.loss_type = loss_type

    def loss(self, predicted_mask: Tensor, target_mask: Tensor, batch_first: bool = False):
        """
        Args:
            predicted_mask: predicted mask
            target_mask: ground truth mask
            batch_first: set True if you use batches
        Returns:
            soft-Dice loss or soft-IoU loss
        """

        assert predicted_mask.shape == target_mask.shape, (
            'Incompatible shape between prediction: {pred_shape} and target: {target_shape}'.format(
                pred_shape=predicted_mask.shape, target_shape=target_mask.shape)
        )

        assert predicted_mask.dim() == 3 and batch_first, (
            'Incompatible shape in target and prediction. Expected dim value: 3, but got: {value}'.format(
                value=predicted_mask.dim())
        )

        assert self.loss_type in ['dice', 'iou', 'both'], '{loss_type} mode is not supported.'.format(loss_type=self.loss_type)
        sum_dims = (-1, -2, -3) if batch_first else (-1, -2)

        intersection = torch.sum(predicted_mask * target_mask, dim=sum_dims)
        all_pixels = torch.sum(predicted_mask, dim=sum_dims) + torch.sum(target_mask, dim=sum_dims)
        union = all_pixels - intersection

        dice_loss = (2.0 * intersection) / (all_pixels + 1e-8)
        dice_loss = torch.mean(dice_loss)  # Average by batch

        iou_loss = intersection / (union + 1e-8)
        iou_loss = torch.mean(iou_loss)  # Average by batch

        if self.loss_type == 'dice':
            return 1.0 - dice_loss
        if self.loss_type == 'iou':
            return 1.0 - iou_loss
        if self.loss_type == 'both':
            return 1.0 - dice_loss, 1.0 - iou_loss

    def __call__(self, 
                 predicted_mask: Tensor, 
                 target_mask: Tensor, 
                 num_classes: int = None,
                 batch_first: bool = True, 
                 ignore_last_index: bool = False
    ):
        """
        Call function for self.loss()
        Args:
            predicted_mask: predicted mask
            target_mask: ground truth mask
            num_classes: use this argument if your target is not prepared for binary format
            batch_first: set True if you use batches
            ignore_first: set True if you use ignore label
            ignore_last_index: ignore last index label for loss 
        Returns:
            soft-Dice loss or soft-IoU loss
        """

        if ignore_last_index:
            num_classes += 1

        if num_classes:
            target_mask = F.one_hot(target_mask, num_classes=num_classes)
            target_mask = target_mask.permute(0, 3, 1, 2)

        if batch_first:
            if ignore_last_index:
                target_mask = target_mask[:, :num_classes-1, :, :]
            predicted_mask = predicted_mask.flatten(start_dim=0, end_dim=1)
            target_mask = target_mask.flatten(start_dim=0, end_dim=1)
        elif ignore_last_index:
            target_mask = target_mask[:num_classes-1, :, :]

        result_loss = self.loss(predicted_mask, target_mask, batch_first=True)
        return result_loss
