import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix


class MeanIoU:
    """
    Class for mean IoU computation.
    Available metrics:
        micro IoU through standard mode (class by class)
        micro IoU through confusion matrix (same result)

        macro IoU through standard mode
        macro IoU through confusion matrix (same result) 
    """
    def __init__(self, num_classes, ignore_class):
        """
        Args:
            num_classes: total relevant classes without ignore class
            ignore_class: number of ignore class
        Note:
            ignore class must be last and is not included in num_classes
        """
        self.num_classes = num_classes
        self.ignore_class = ignore_class


    def __call__(self, predicted_mask: Tensor, target_mask: Tensor, average='micro', mode='confusion_matrix',):
        """
        Args:
            predicted_mask: predicted mask in the argmax format, not binary or softmax
            target_mask: target mask in format like predicted mask
            average: 'micro' or 'macro'
            mode: 'standard' or 'confusion_matrix'
                  confusion matrix calculated using the helpful function from scikit-learn
        Returns:
            mean Intersection Over Union by average format
        Note:
            'micro' average compute all unions and intersections for all relevant classes.
            And then they are summed up
            'macro' average means a division into the corresponding classes and averaging over them.

            'standard' mode and 'confusion_matrix' mode fully compatible.                  
        """
        assert average in ['micro', 'macro'], ('{average_type} average type is not supported.').format(average_type=average)
        assert mode in ['standard', 'confusion_matrix'], ('{mode_type} mode for calculation is not supported. '
                'Expected mode: standard or confusion_matrix').format(mode_type=average)

        assert predicted_mask.dtype == torch.int64, (
            'Unsupported dtype of predicted tensor. Expected dtype: torch.int64, but got: {pred.dtype}'.format(
                pred=predicted_mask.dtype)
        )
        assert predicted_mask.shape == target_mask.shape, (
            'Incompatible shape between prediction: {pred_shape} and target: {target_shape}'.format(
                pred_shape=predicted_mask.shape, target_shape=target_mask.shape)
        )

        assert predicted_mask.dim() in [2, 3], (
            'Incompatible shape in target and prediction. Expected dim value: 2 or 3, but got: {value}'.format(
                value=predicted_mask.dim())
        )

        if mode == 'standard':
            with torch.no_grad():
                predicted_mask = predicted_mask.view(-1)
                target_mask = target_mask.view(-1)
                predicted_mask = torch.where(target_mask == self.ignore_class, self.ignore_class, predicted_mask)

                intersection = torch.zeros(self.num_classes, dtype=torch.float32)
                union = torch.zeros(self.num_classes, dtype=torch.float32)
                for class_idx in range(self.num_classes):
                    true_predicted_mask = predicted_mask == class_idx
                    true_target_mask = target_mask == class_idx
                    
                    union[class_idx] = torch.sum(torch.logical_or(true_predicted_mask, true_target_mask), dim=0)
                    intersection[class_idx] = torch.sum(torch.logical_and(true_predicted_mask, true_target_mask), dim=0)
                
                relevant_classes = union > 0    
            
            if average == 'micro':
                IoU = intersection[relevant_classes] / union[relevant_classes]
                mIoU = torch.mean(IoU)
                return mIoU
            elif average == 'macro':
                mIoU = torch.sum(intersection[relevant_classes]) / torch.sum(union[relevant_classes])
                return mIoU

        if mode == 'confusion_matrix':
            with torch.no_grad():
                current_device = predicted_mask.device

                y_pred = predicted_mask.flatten().cpu()
                y_true = target_mask.flatten().cpu()

                confusion_tensor = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
                confusion_tensor = torch.tensor(confusion_tensor, device=current_device)

                intersection = torch.diag(confusion_tensor)
                ground_truth = torch.sum(confusion_tensor, dim=1)
                predicted = torch.sum(confusion_tensor, dim=0)
                
                union = ground_truth + predicted - intersection
                relevant_classes = union > 0
            
            if average == 'micro':
                IoU = intersection[relevant_classes] / union[relevant_classes]
                mIoU = torch.mean(IoU)
                return mIoU
            elif average == 'macro':
                mIoU = torch.sum(intersection[relevant_classes]) / torch.sum(union[relevant_classes])
                return mIoU
