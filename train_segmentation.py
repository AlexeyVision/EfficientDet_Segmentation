import os, shutil
from hydra import compose, initialize
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader
from datasets.segmentation import SegmentationDatasetVOC, SegmentationDatasetCOCO

from core.fit import fit
from core.model.segmentation.efficientdet import EfficientDetSegmentation
from core.model.segmentation.unet import UNet 


experiments_path = './experiments'
config_path = './config'
train_config = 'train_segmentation.yaml'

def train():
    with initialize(version_base=None, config_path=config_path):
        config = compose(config_name=train_config)

    # Create new experiment directory
    experiment_dir = os.path.join(experiments_path, config.experiment_name)
    assert not os.path.exists(experiment_dir), (
        'Experiment {exp_name} already exists.'.format(exp_name=config.experiment_name)
    )
    os.makedirs(experiment_dir)

    # Copy train config
    src = 'config/train_segmentation.yaml'
    dest = os.path.join(experiment_dir, 'train_segmentation.yaml')
    shutil.copyfile(src, dest, follow_symlinks=True)

    # Copy model config
    src = f'config/segmentation/{config.train.model.name}.yaml'
    dest = os.path.join(experiment_dir, f'{config.train.model.name}.yaml')
    shutil.copyfile(src, dest, follow_symlinks=True)

    # Optimal in memory usage than torch.cuda.is.available()
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

    # *************MS COCO*************
    # train_dataset = SegmentationDatasetCOCO(
    #     root='data/datasets/MS_COCO',
    #     width=config.data.preprocessing.resize_width,
    #     height=config.data.preprocessing.resize_height,
    #     mask_width=config.data.preprocessing.resize_mask_width,
    #     mask_height=config.data.preprocessing.resize_mask_height,
    #     split='train'
    # )
    # val_dataset = SegmentationDatasetCOCO(
    #     root='data/datasets/MS_COCO',
    #     width=config.data.preprocessing.resize_width,
    #     height=config.data.preprocessing.resize_height,
    #     mask_width=config.data.preprocessing.resize_mask_width,
    #     mask_height=config.data.preprocessing.resize_mask_height,
    #     split='val'
    # )

    # train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)
    # *********************************

    train_dataset = SegmentationDatasetVOC(
        path='data/VOC2012',
        width=config.data.preprocessing.resize_width,
        height=config.data.preprocessing.resize_height,
        mask_width=config.data.preprocessing.resize_mask_width,
        mask_height=config.data.preprocessing.resize_mask_height,
        train=True,
        augmented=True
    )
    val_dataset = SegmentationDatasetVOC(
        path='data/VOC2012',
        width=config.data.preprocessing.resize_width,
        height=config.data.preprocessing.resize_height,
        mask_width=config.data.preprocessing.resize_mask_width,
        mask_height=config.data.preprocessing.resize_mask_height,
        train=False
    )

    # Dataloaders initialization
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)

    # Model initialization
    if config.train.model.name.startswith('efficientdet'):
        model = EfficientDetSegmentation(
            name=config.train.model.name,
            num_classes=config.train.model.num_classes,
            backbone_pretrained=config.train.model.backbone_pretrained,
            weights=config.train.model.weights
        )
    elif config.train.model.name.startswith('unet'):
        model = UNet(
            in_channels=config.train.model.in_channels,
            num_classes=config.train.model.num_classes,
            weights=config.train.model.weights
        )
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Optimizer and scheduler initialization
    optimizer = instantiate(config.train.optimizer, params=model.parameters())
    if config.train.model.optimizer_state:
        optimizer_state_dict = torch.load(config.train.model.optimizer_state)
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = instantiate(config.train.scheduler, optimizer=optimizer)

    # Loss function(s) initialization
    criterion = list()
    if config.train.loss.distribution:
        criterion.append(instantiate(config.train.loss.distribution))
    if config.train.loss.region:
        criterion.append(instantiate(config.train.loss.region))

    assert len(criterion) > 0, 'Loss function initialization error. Please, use available loss functions.'
    
    # Using tuple only if there are multiple loss functions
    if len(criterion) > 1:
        criterion = tuple(criterion)
    else:
        criterion = criterion[0]

    # Checking ignore index
    if not hasattr(config.train.loss.distribution, 'ignore_index'):
        ignore_class = None
    else:
        ignore_class = config.train.loss.distribution

    fit(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        accum_iter=config.train.loss.accumulation_iter,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config.train.num_epochs,
        scheduler=scheduler,
        device=device,
        num_classes=config.train.model.num_classes,
        validation_epoch_step=config.validation.epoch_step,
        amp=config.train.amp,
        experiment_dir=experiment_dir,
        ignore_class=ignore_class
    )


if __name__ == "__main__":
    train()


