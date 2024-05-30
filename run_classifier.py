import argparse
import json
import torch
import torchvision
from PIL import Image
from core.model.classification.efficientnet import EfficientNet


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, help='Path to image for classification')
parser.add_argument('--model', type=str, default=None,
                    help='Model name for classification (efficientnet_b0 - efficientnet_b7)')
parser.add_argument('--resolution', type=int, help='Input height and width of image')
parser.add_argument('--num_classes', type=int, default=None, help='Output model classes')
parser.add_argument('--labels', type=str, default=None, help='Path to image labels')
parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
args = parser.parse_args()

# Optimal in memory usage than torch.cuda.is.available()
device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')

default_resolution = {
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600
}

if args.resolution is None:
    resolution = default_resolution[args.model]
else:
    resolution = args.resolution

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((resolution, resolution)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
image = transforms(Image.open(args.data))
image = torch.unsqueeze(image, dim=0).to(device).float()

model = EfficientNet(
    name=args.model,
    num_classes=args.num_classes,
    weights=args.weights,
    pretrained=True,
    load_classifier=True
).to(device)

model.eval()
with torch.no_grad():
    outputs = model(image)

with open(args.labels) as file:
    labels = json.load(file)

# Set the color style for most confident class
green = '\033[1;32m'
default = '\033[1;0m'

k = 5 if len(labels) >= 5 else 1  # Set the number of most confident classes
for num, idx in enumerate(torch.topk(outputs, k=k).indices.squeeze(0).tolist()):
    prob = str(round(torch.softmax(outputs, dim=1)[0, idx].item() * 100, 2)) + '%'
    if num == 0:
        prob = green + prob + default
    print('{label:<75} {p}'.format(label=labels[str(idx)]['class'], p=prob))
