import argparse
import torch
import json
import numpy as np
import cv2
import albumentations as alb
from core.model.segmentation.unet import UNet
from core.model.segmentation.efficientdet import EfficientDetSegmentation


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, help='Path to image for segmentation')
parser.add_argument('--model', type=str, default=None,
                    help='Model name for segmentation (efficientdet_d[0-7]_seg, unet)')
parser.add_argument('--resolution', type=int, default=512, help='Input height and width of image')
parser.add_argument('--num_classes', type=int, default=3, help='Output model classes')
parser.add_argument('--labels', type=str, default=None, help='Path to image labels')
parser.add_argument('--in_channels', type=int, default=None, help='Input image channels. Actual for UNet only')
parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
parser.add_argument('--output_name', type=str, default=None, help='Output filename if you want to save predicted image mask')
args = parser.parse_args()

# Optimal in memory usage than torch.cuda.is.available()
device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')

# Model initialization
if args.model.startswith('efficientdet'):
    model = EfficientDetSegmentation(
        name=args.model,
        num_classes=args.num_classes,
        weights=args.weights,
    )
elif args.model.startswith('unet'):
    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        weights=args.weights
    )
model = model.to(device)

# Prepocessing
input_image = cv2.imread(args.data)
height, width, _ = input_image.shape
image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
transforms = alb.Compose(
    [
        alb.Resize(args.resolution, args.resolution),
        alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.2])
    ]
)
image = transforms(image=image)['image']
image = torch.tensor(image, dtype=torch.float32).to(device)
image = image.permute(2, 0, 1).to(device) # H x W x C -> C x H x W

model.eval()
with torch.no_grad():
    image = image.unsqueeze(dim=0)
    predictions = model(image)
    predicted_mask = torch.argmax(predictions, dim=1)

with open(args.labels) as file:
    labels = json.load(file)

# Postprocessing
predicted_mask = predicted_mask.to('cpu').squeeze(dim=0)
predicted_mask = np.array(predicted_mask, dtype=np.uint8)

predicted_color_mask = np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8)
for label, item in labels.items():
    predicted_color_mask[predicted_mask == int(label)] = item['color']

predicted_color_mask = cv2.resize(predicted_color_mask, (width, height))
output_image = np.hstack((input_image, predicted_color_mask))
if args.output_name:
    cv2.imwrite(filename=args.output_name, img=output_image)
else:
    cv2.imshow(winname='Predicted mask', mat=output_image)
    cv2.waitKey(0)
