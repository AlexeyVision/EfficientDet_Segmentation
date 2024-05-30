import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from core.metrics import MeanIoU
from terminaltables import AsciiTable


def evaluate(model, criterion, val_dataloader, device, num_classes, ignore_class):
    """
    Args:
        model: trainable model
        criterion: optimization loss function.
                   Can be a single loss or a tuple of loss (distributed loss, region loss).
                   Available loss functions: Cross Entropy, Dice, IoU.
                   If you want to use BCE loss see details in your dataset structure and loss inputs.
        val_dataloader: validation dataloader
        device: working device.
                Can be either 'cpu' or 'cuda'
        num_classes: number of classes in the dataset,
        ignore_class: ignore class label for loss functions and metrics.
    Returns:
        {'distribution': distribution loss, 'region', region loss, 'score': score}
    """
    model.eval()
    mean_iou = MeanIoU(num_classes=num_classes, ignore_class=ignore_class)
    loss_val_info = {'distribution': 0.0, 'region': 0.0, 'score': 0.0,}

    region_loss_type = None
    distribution_loss_type = None
    if isinstance(criterion, tuple):
        if isinstance(criterion[0], torch.nn.CrossEntropyLoss):
            distribution_loss_type = 'CE'
        elif isinstance(criterion[0], torch.nn.BCEWithLogitsLoss):
            distribution_loss_type = 'BCE'
        region_loss_type = 'Dice' if criterion[1].loss_type == 'dice' else 'IoU'
        pbar = tqdm(val_dataloader, desc=f'Validation. {distribution_loss_type}: 0.000 soft-{region_loss_type}: 0.000', leave=True)
    else:
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            distribution_loss_type = 'CE'
            pbar = tqdm(val_dataloader, desc=f'Validation. {distribution_loss_type}: 0.000', leave=True)
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            distribution_loss_type = 'BCE'
            pbar = tqdm(val_dataloader, desc=f'Validation. {distribution_loss_type}: 0.000', leave=True)
        else:
            region_loss_type = 'Dice' if criterion.loss_type == 'dice' else 'IoU'
            pbar = tqdm(val_dataloader, desc=f'Validation. soft-{region_loss_type}: 0.000', leave=True)

    for idx, batch in enumerate(pbar):
        images, targets = batch['image'], batch['target']
        images = images.to(device)
        targets = targets.to(device)

        with (torch.no_grad()):
            predictions = model(images)

            if isinstance(criterion, tuple):
                distribution_loss = criterion[0](predictions, targets)
                region_loss = criterion[1](
                    predicted_mask=F.softmax(predictions, dim=1),
                    target_mask=targets,
                    batch_first=True,
                    num_classes=num_classes,
                    ignore_last_index=True if ignore_class else False
                )
            else:
                if isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    distribution_loss = criterion(predictions, targets)
                else:
                    region_loss = criterion(
                        predicted_mask=F.softmax(predictions, dim=1),
                        target_mask=targets,
                        batch_first=True,
                        num_classes=num_classes,
                        ignore_last_index=True if ignore_class else False
                    )

            mean_iou_score = mean_iou(
                predicted_mask=torch.argmax(predictions, dim=1),
                target_mask=targets,
                average='micro',
                mode='confusion_matrix',
            )

            loss_val_info['score'] += mean_iou_score.item()

        if distribution_loss_type and region_loss_type:
            loss_val_info['distribution'] += distribution_loss.item()
            loss_val_info['region'] += region_loss.item()
            pbar.set_description(desc=f'Validation.'
                                      f' {distribution_loss_type}: {loss_val_info["distribution"] / (idx + 1):.3f}'
                                      f' soft-{region_loss_type}: {loss_val_info["region"] / (idx + 1):.3f}'
                                      f' mIoU score: {loss_val_info["score"] / (idx + 1):.3f}')
        elif distribution_loss_type:
            loss_val_info['distribution'] += distribution_loss.item()
            pbar.set_description(desc=f'Validation.'
                                      f' {distribution_loss_type}: {loss_val_info["distribution"] / (idx + 1):.3f}'
                                      f' mIoU score: {loss_val_info["score"] / (idx + 1):.3}')
        else:
            loss_val_info['region'] += region_loss.item()
            pbar.set_description(desc=f'Validation.'
                                      f' soft-{region_loss_type}: {loss_val_info["region"] / (idx + 1):.3f}'
                                      f' mIoU score: {loss_val_info["score"] / (idx + 1):.3f}')

    loss_val_info['distribution'] /= len(val_dataloader)
    loss_val_info['region'] /= len(val_dataloader)
    loss_val_info['score'] /= len(val_dataloader)

    return loss_val_info


def fit(
        model,
        optimizer,
        criterion,
        scheduler,
        train_dataloader,
        val_dataloader,
        epochs=100,
        accum_iter=1,
        device='cuda',
        num_classes=21,
        validation_epoch_step=1,
        amp=True,
        experiment_dir='experiments/',
        ignore_class=None
):
    """
    Args:
        model: trainable model
        optimizer: model optimizer
        criterion: optimization loss function.
                   Can be a single loss or a tuple of loss (distributed loss, region loss).
                   Available loss functions: Cross Entropy, Dice, IoU.
                   If you want to use BCE loss see details in your dataset structure and loss inputs.
        scheduler: learning rate scheduler
        train_dataloader: training dataloader
        val_dataloader: validation dataloader
        epochs: num of epochs
        accum_iter: iterations for update weights using gradient accumulation 
        device: working device.
                Can be either 'cpu' or 'cuda'
        num_classes: number of classes in the dataset
        validation_epoch_step: the number of epochs to repeat validation
        amp: automatic mixed precision.
             True or False
        experiment_dir: dir to save experiment results
        ignore_class: ignore class label for loss functions and metrics.
    Returns:
        saved model weights and optimizer state to experiment dir after every validation epoch
    """
    
    compute_dtype = torch.float32
    scaler = GradScaler()
    if amp:
        compute_dtype = torch.float16
    
    for epoch in range(epochs):
        model.train()

        loss_info = {'distribution': 0.0, 'region': 0.0, 'total': 0.0}
        region_loss_type = None
        distribution_loss_type = None
        if isinstance(criterion, tuple):
            if isinstance(criterion[0], torch.nn.CrossEntropyLoss):
                distribution_loss_type = 'CE'
            elif isinstance(criterion[0], torch.nn.BCEWithLogitsLoss):
                distribution_loss_type = 'BCE'
            region_loss_type = 'Dice' if criterion[1].loss_type == 'dice' else 'IoU'
            pbar = tqdm(train_dataloader, desc=f'Train epoch: {epoch + 1}. {distribution_loss_type}: 0.000 soft-{region_loss_type}: 0.000', leave=True)
        else:
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                distribution_loss_type = 'CE'
                pbar = tqdm(train_dataloader, desc=f'Train epoch: {epoch + 1}. {distribution_loss_type}: 0.000', leave=True)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                distribution_loss_type = 'BCE'
                pbar = tqdm(train_dataloader, desc=f'Train epoch: {epoch + 1}. {distribution_loss_type}: 0.000', leave=True)
            else:
                region_loss_type = 'Dice' if criterion.loss_type == 'dice' else 'IoU'
                pbar = tqdm(train_dataloader, desc=f'Train epoch: {epoch + 1}. soft-{region_loss_type}: 0.000', leave=True)

        for idx, batch in enumerate(pbar):
            images, targets = batch['image'], batch['target']
            images = images.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=compute_dtype):
                predictions = model(images)

                if isinstance(criterion, tuple):
                    distribution_loss = criterion[0](predictions, targets)
                    region_loss = criterion[1](
                        predicted_mask=F.softmax(predictions, dim=1),
                        target_mask=targets,
                        batch_first=True,
                        num_classes=num_classes,
                        ignore_last_index=True if ignore_class else False
                    )
                    loss = distribution_loss + region_loss
                else:
                    if isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                        distribution_loss = criterion(predictions, targets)
                        loss = distribution_loss
                    else:
                        region_loss = criterion(
                            predicted_mask=F.softmax(predictions, dim=1),
                            target_mask=targets,
                            batch_first=True,
                            num_classes=num_classes,
                            ignore_last_index=True if ignore_class else False
                        )
                        loss = region_loss
                
                scaler.scale(loss / accum_iter).backward()  # Normalize loss

            # Gradient accumulation
            if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if distribution_loss_type and region_loss_type:
                loss_info['distribution'] += distribution_loss.item()
                loss_info['region'] += region_loss.item()
                pbar.set_description(desc=f'Train epoch: {epoch + 1}'
                                          f' {distribution_loss_type}: {loss_info["distribution"] / (idx + 1):.3f}'
                                          f' soft-{region_loss_type}: {loss_info["region"] / (idx + 1):.3f}')
            elif distribution_loss_type:
                loss_info['distribution'] += distribution_loss.item()
                pbar.set_description(desc=f'Train epoch: {epoch + 1}'
                                          f' {distribution_loss_type}: {loss_info["distribution"] / (idx + 1):.3f}')
            else:
                loss_info['region'] += region_loss.item()
                pbar.set_description(desc=f'Train epoch: {epoch + 1}'
                                          f' soft-{region_loss_type}: {loss_info["region"] / (idx + 1):.3f}')
        scheduler.step()

        if (epoch + 1) % validation_epoch_step == 0:
            loss_val_info = evaluate(model, criterion, val_dataloader, device, num_classes, ignore_class=ignore_class)
            loss_val_info['total'] = loss_val_info['distribution'] + loss_val_info['region']
            
            loss_info['total'] = (loss_info['distribution'] + loss_info['region']) / len(train_dataloader)
            loss_info['distribution'] /= len(train_dataloader)
            loss_info['region'] /= len(train_dataloader)

            display = [
                    ['Type',                             'Train',                           'Validation'],
                    [f'{distribution_loss_type} loss:',  f'{loss_info["distribution"]:.3f}', f'{loss_val_info["distribution"]:.3f}'],
                    [f'soft-{region_loss_type} loss:',   f'{loss_info["region"]:.3f}',       f'{loss_val_info["region"]:.3f}'],
                    [f'total loss:',                     f'{loss_info["total"]:.3f}',        f'{loss_val_info["total"]:.3f}']
            ]

            print(f'Epoch {epoch + 1} / {epochs}')
            if distribution_loss_type and region_loss_type:
                print(AsciiTable(display).table)
            elif distribution_loss_type:
                del display[2]
                print(AsciiTable(display).table)
            else:
                del display[1]
                print(AsciiTable(display).table)
            print(f'Mean IoU score: {loss_val_info["score"]:.3f}\n')

            torch.save(model.state_dict(), os.path.join(experiment_dir, 'model_{epoch}.pth'.format(epoch=epoch + 1)))
            torch.save(optimizer.state_dict(), os.path.join(experiment_dir, 'optim_{epoch}.pth'.format(epoch=epoch + 1)))

