# EfficientDet for semantic segmentation
![image](https://github.com/AlexeyVision/EfficientDet_Segmentation/assets/171249650/32fdbaf0-cae7-40fa-a134-f187b10ee94c)

## Major modification 💡
**After the last BiFPN pathway, a top-down pathway has been added to fuse multi-scale features from level 2 to 7 (P2-P7). This structure, when viewed with the last path from BiFPN, is similar to the PANet architecture. Transposed convolutions at the output level P2 are used to bring the output resolution of the mask to the input resolution of the image.**
___
Note: Architecture for using features exclusively from the P2 level without a top-down pathway is also available.

## Datasets
* Backbone pre-trained [ImageNet](https://www.image-net.org/) weights were imported from [PyTorch EfficientNet framework](https://github.com/lukemelas/EfficientNet-PyTorch).
* [MS COCO](https://cocodataset.org/) segmentation dataset was used for Pascal VOC pre-training.
* [Augmented Pascal VOC 2012](https://www.dropbox.com/scl/fi/xccys1fus0utdioi7nj4d/SegmentationClassAug.zip?rlkey=0wl8iz6sc40b3qf6nidun4rez&e=1&dl=0) segmentation dataset was used for training with the same validation set as in the regular [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
___
**Used augmentations:**
>
    alb.Resize(512, 512),
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

## Inference
Segmentation:
> 
    python3 --data dog.jpg --model efficientdet_d4_seg --resolution 512 --num_classes 21 --labels config/PascalVOC2012.json --weights model.pth --output dog_mask.jpg
Classification:
> 
    python3 --data dog.jpg --model efficientnet_b4 --resolution 380 --num_classes 21 --labels config/ImageNet.json --weights model.pth

## Results
| Model   		      		                         | Dataset 	          | Input size <br> <sub> (pixel)   | Output size <br> <sub> (pixel)  | mIoU <br> <sub> 
| :---:   		      		                         | :---:   	          | :---:    	                      | :---: 		                      | :---: 
| Modified EfficientDet D4 <br> <sub> (Ours⭐)   | Pascal VOC 2012   | 512 x 512       	              | 512 x 512     	  	            | 71.90
| Modified EfficientDet D4 <br> <sub> (Paper)  	 | Pascal VOC 2012    | Not specify       	      	    | Not specify    	                | 81.74

## Training
> 
    python3 train_segmentation.py
    
All training parameters in `train_segmentation.yaml`:
| Parameter  		      		               | Description
| :---   		      		                   | :---  	     
|`experiment_name`                       | Folder name for experiment
|`train.amp`                             | Using Automatic Mixed Precision
|`train.num_epochs`                      | Total training epochs
|`train.model.name`                      | Imported model config (unet, efficientdet_d4_seg or custom)
|`train.model.num_classes`               | Number of classes
|`train.model.backbone_pretrained`       | Using backbone pre-trained weights
|`train.model.weights`                   | Using pre-trained weights
|`train.model.optimizer_state`           | Using checkpount of optimizer      
|`train.optimizer._target_`              | Initialization of the optimizitaion method
|`train.optimizer.lr`                    | Initial learning rate
|`train.optimizer.weight_decay`          | Initial weight decay
|`train.optimizer._target_`              | Initialization of the scheduler method
|`loss.accumulation_iter`                | Gradient accumulation
|`loss.distribution._target_`            | Distribution loss 
|`loss.distribution.ignore_index`        | Boundary ignore for Pascal VOC only
|`loss.region._target_`                  | Region loss
|`loss.region.loss_type`                 | soft-IoU or soft-Dice
|`validation.epoch_step`                 | Validation process every {epoch_step} epoch
|`data.preprocessing.resize_height`      | Input image height 
|`data.preprocessing.resize_width`       | Input iamge width
|`data.preprocessing.resize_mask_height` | Input mask height
|`data.preprocessing.resize_mask_width`  | Input mask width
|`data.preprocessing.batch_size`         | Initial batch size
    
Note: *mean IoU will be calculated on the validation dataloader after `validation.epoch_step`.*

![image](https://github.com/AlexeyVision/EfficientDet_Segmentation/assets/171249650/124d9877-0de5-48ef-ba55-3ec8c8cb2d04)
## Main Dependencies
**PyTorch** 
> Version: 2.2.2

**TorchVision** 
> Version: 0.17.2

**Hydra** 
> Version: 1.3.2

**Albumentations**
> Version: 1.4.4

**OpenCV**
> Version: 4.9.0.80

**NumPy**
> Version: 1.26.4

**terminaltables**
> Version: 3.1.10

## References
* [MobileNet v1 paper](https://arxiv.org/pdf/1704.04861)
* [MobileNet v2 paper](https://arxiv.org/pdf/1801.04381)
* [MobileNet v3 paper](https://arxiv.org/pdf/1905.02244)
___
* [EfficientNet paper](https://arxiv.org/pdf/1905.11946)
* [EfficientDet paper](https://arxiv.org/pdf/1911.09070)

## Contact
Developer: **Alexey Serzhantov**

Email: serzhantov0289@gmail.com  
