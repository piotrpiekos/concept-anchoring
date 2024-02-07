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

generate_images_module = importlib.import_module("eval-scripts.generate-images")
generate_images = generate_images_module.generate_images
batch_generate_images = generate_images_module.batch_generate_images

classes_names = [
    'cassette player', 'chain saw', 'church', 'gas pump', 'tench', 'garbage truck', 'english springer',
    'golf ball', 'parachute', 'french horn'
]
classes_ids = [
    482, 491, 497, 571, 0, 569, 217, 574, 701, 566
]

classes_codes = dict(zip(classes_names, classes_ids))

PROMPTS_PATH = 'data/unique-object-removal-prompts.csv'
SAVE_DIR = 'images/generations'
NUM_CLASSES = len(list(classes_codes.values()))
OUTPUT_DIR = 'results/'


def acc(pred_classes: torch.tensor, true_class: int):
    return float(torch.mean((pred_classes == true_class).float()))


def get_accuracy_per_class(model, images, true_class):
    print('images shape: ', images.shape)
    with torch.no_grad():
        preds = model(images)  # [num_images, num_class]

    all_classes = torch.tensor(classes_ids)

    absolute_pred = preds.argmax(dim=1)
    local_pred = preds[:, all_classes].argmax(dim=1)

    true_absolute_id = classes_codes[true_class]
    true_local_id = classes_ids.index(true_absolute_id)

    local_acc = acc(local_pred, true_local_id)
    absolute_acc = acc(absolute_pred, true_absolute_id)

    return local_acc, absolute_acc


def evaluate_object_removal(model_name, models_path, removed_class_name, num_samples):
    print('GENERATING IMAGES')
    model = resnet50(pretrained=True).eval()
    images_dict = batch_generate_images(model_name, models_path, PROMPTS_PATH, num_samples)
    print('GETTING PREDICTIONS')

    local_accuracies, absolute_accuracies = dict(), dict()
    for true_class, images in images_dict.items():
        local_acc, absolute_acc = get_accuracy_per_class(model, images, true_class)
        local_accuracies[true_class] = local_acc
        absolute_accuracies[true_class] = absolute_accuracies

    local_removed_acc = local_accuracies.pop(removed_class_name)
    # local_accuracies has only other classes now
    local_other_acc = np.mean(list(local_accuracies.values()))

    absolute_removed_acc = absolute_accuracies.pop(removed_class_name)
    # absolute_accuracies has only other classes now
    absolute_other_acc = np.mean(list(absolute_accuracies.values()))

    results = {
        'local': {
            'acc_removed': local_removed_acc,
            'acc_other': local_other_acc
        },
        'absolute': {
            'acc_removed': absolute_removed_acc,
            'acc_other': absolute_other_acc
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        prog='evaluateObjectRemoval',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--models_path', help='name of model', type=str, required=True)
    parser.add_argument('--num_samples', help='how many images used for evaluation', type=int, required=True)
    parser.add_argument('--removed_class_name', help='name of class to remove', type=str, required=True)

    args = parser.parse_args()

    model_name = args.model_name
    models_path = args.models_path
    num_samples = args.num_samples
    removed_class_name = args.removed_class_name

    results = evaluate_object_removal(model_name, models_path, removed_class_name, num_samples)
    filepath = os.path.join(OUTPUT_DIR, f'{removed_class_name}.json')
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
