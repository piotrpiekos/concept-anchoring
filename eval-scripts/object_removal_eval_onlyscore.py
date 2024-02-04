import argparse

import importlib
import pandas as pd
import os

import torchvision
# from torchvision.models import resnet50, ResNet50_Weights
# from eval_scripts.generate_images import generate_images
from torchvision.models import resnet50
import torch
import json

import numpy as np

classes_names = [
    'cassette player', 'chain saw', 'church', 'gas pump', 'tench', 'garbage truck', 'English springer',
    'golf ball', 'parachute', 'French horn'
]
classes_ids = torch.tensor([
    482, 491, 497, 571, 0, 569, 217, 574, 701, 566
])
OUTPUT_DIR = '../results/'


def evaluate_only(images_path, removed_class_name):
    model = resnet50(pretrained=True).eval()

    accuracies = dict()
    for im_id, class_id in enumerate(classes_ids):
        images = []
        for fname in filter(lambda x: x.startswith(str(im_id)), os.listdir(images_path)):
            im = torchvision.io.read_image(os.path.join(images_path, fname))
            images.append(im)

        images = torch.stack(images)
        with torch.no_grad():
            preds = model(images.float() / 255)
            local_pred = preds[:, classes_ids].argmax(dim=1)
            acc = torch.mean((local_pred == im_id).float())
            accuracies[classes_names[im_id]] = acc

    accuracy_on_other = np.mean([acc for name, acc in accuracies.items() if name != removed_class_name])
    return accuracies[removed_class_name], accuracy_on_other


def main():
    parser = argparse.ArgumentParser(
        prog='evaluateObjectRemoval',
        description='only evaluate already generated images ')
    parser.add_argument('--images_path', help='path to the s of model', type=str, required=True)
    parser.add_argument('--removed_class_name', help='name of class to remove', type=str, required=True)

    args = parser.parse_args()

    images_path = args.images_path
    removed_class_name = args.removed_class_name

    acc_on_removed, acc_on_other = evaluate_only(images_path, removed_class_name)
    filepath = os.path.join(OUTPUT_DIR, f'{removed_class_name}.json')
    results = {
        'acc_removed': acc_on_removed,
        'acc_other': acc_on_other
    }
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
