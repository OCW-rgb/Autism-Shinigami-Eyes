"""Predict one of nine image labels using the selected ResNet18 model.

    python AI_Hackathon.py predict /path/to/image.jpg

Images are oriented, converted to RGB, padded to 128x128, and normalized
in memory. No training data, pretrained download, or internet is required.

This is the port of the michael chen model (../micheal chen/AI_Hackathon.py)
into uno-q-live/uno-complete/. The checkpoint bundled here is lexfm0.pt
(the ResNet18 trained on the FANE 9-class expression set).
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

DEFAULT_MODEL = Path(__file__).resolve().parent / 'lexfm0.pt'


class ResidualBlock(nn.Module):
    """Basic ResNet block with names compatible with the official weights."""

    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample = None
        if stride != 1 or inputs != outputs:
            self.downsample = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + identity)


class ResNet18(nn.Module):
    """ResNet18 architecture for loading the selected local checkpoint."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def read_image(path, image_size=128):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image = ImageOps.pad(image, (image_size, image_size),
                             method=Image.Resampling.LANCZOS, color=(128, 128, 128))
        pixels = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def load_model(path=DEFAULT_MODEL, device='cpu'):
    checkpoint = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'resnet18' or checkpoint.get('normalization') != 'imagenet':
        raise ValueError('This runner requires the selected ResNet18/ImageNet-normalized checkpoint.')
    labels = checkpoint['labels']
    model = ResNet18(len(labels))
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), labels, checkpoint.get('image_size', 128)


def predict(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model, labels, image_size = load_model(args.model, device)
    with torch.inference_mode():
        for path in args.images:
            inputs = read_image(path.expanduser(), image_size).unsqueeze(0).to(device)
            probabilities = model(inputs).softmax(1)[0]
            index = probabilities.argmax().item()
            print(f'{path}: {labels[index]} (model probability {probabilities[index].item():.1%})')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    prediction = commands.add_parser('predict', help='Predict labels for image files.')
    prediction.add_argument('images', nargs='+', type=Path)
    prediction.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    prediction.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    try:
        predict(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
