import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as alb


class SegmentationDatasetVOC(Dataset):
    num_classes = 21
    color_mapping = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]
    ]

    def __init__(self, path, height, width, mask_height, mask_width, train=True, augmented=False):
        self.main_dir = path
        self.height = height
        self.width = width
        self.train = train
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.augmented = augmented

        if self.train:
            if self.augmented:
                with open(os.path.join(self.main_dir, 'ImageSets/Segmentation/train_aug.txt')) as f:
                    names = f.readlines()
            else:
                with open(os.path.join(self.main_dir, 'ImageSets/Segmentation/train.txt')) as f:
                    names = f.readlines()
            
            self.transforms = alb.Compose(
                [
                    alb.Resize(self.height, self.width),
                    alb.HorizontalFlip(p=0.5),
                    alb.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    alb.RandomBrightnessContrast(p=0.5),
                    alb.Affine(
                        scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate = (-45, 45),
                        shear = {"x": (-10, 10), "y": (-10, 10)}, 
                        p=0.6
                    ),
                    alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2])
                ]
            )
        else:
            with open(os.path.join(self.main_dir, 'ImageSets/Segmentation/val.txt')) as f:
                names = f.readlines()

            self.transforms = alb.Compose(
                [
                    alb.Resize(self.height, self.width),
                    alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2])
                ]
            )

        mask_dir = 'SegmentationClassAug/' if self.augmented else 'SegmentationClass/'
        names = list(map(lambda x: x.rstrip('\n'), names))
        self.image_names = [os.path.join(self.main_dir, 'JPEGImages/{name}.jpg').format(name=name) for name in names]
        self.mask_names = [os.path.join(self.main_dir, mask_dir + '{name}.png').format(name=name) for name in names]

        assert len(self.image_names) == len(self.mask_names)


    def __len__(self):
        return len(self.image_names)

    
    def __getitem__(self, item):
        image_name = self.image_names[item]
        mask_name = self.mask_names[item]
        image = np.array(Image.open(image_name).convert('RGB'))

        # Simple Cross Entropy target preprocessing
        target_mask = np.array(Image.open(mask_name))

        # Casting 255 to 21 for further calculation of metrics
        target_mask = np.where(target_mask == 255, self.num_classes, target_mask) 

        # mask = np.array(Image.open(mask_name).convert('RGB'))
        # height, width = mask.shape[:2]

        # Cross Entropy target preprocessing
        # fill_value = num_classes use for ignore class
        # target_mask = np.full((height, width), fill_value=self.num_classes, dtype=np.int64)
        # for class_idx, color in enumerate(self.color_mapping):
        #     target_mask = np.where(np.all(mask == color, axis=-1), class_idx, target_mask)

        # Binary Cross Entropy target preprocessing
        # target_mask = np.zeros((height, width, self.num_classes), dtype=np.float32)
        # for idx, color in enumerate(self.color_mapping):
        #     target_mask[:, :, idx] = np.all(mask == color, axis=-1).astype(float) 

        transformed = self.transforms(image=image, mask=target_mask)
        transformed_image = transformed['image']
        transformed_target_mask = transformed['mask']

        # Resize mask to output shape from network
        if self.height != self.mask_height or self.width != self.mask_width:
            transformed_target_mask = alb.Resize(self.mask_height, self.mask_width)(image=transformed_target_mask)['image']

        transformed_image = (torch.tensor(transformed_image, dtype=torch.float32).permute(2, 0, 1))
        transformed_target_mask = torch.tensor(transformed_target_mask, dtype=torch.int64)  # For Cross Entropy
        # transformed_target_mask = torch.tensor(transformed_target_mask, dtype=torch.float32).permute(2, 0, 1)  # For Binary Cross Entropy 
        
        return {'image': transformed_image, 'target': transformed_target_mask}