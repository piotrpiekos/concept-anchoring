import argparse

import importlib
import pandas as pd
import os

import torchvision
#from torchvision.models import resnet50, ResNet50_Weights
#from eval_scripts.generate_images import generate_images
from torchvision.models import resnet50
import torch
import json

import numpy as np
import shutil

generate_images_module = importlib.import_module("eval-scripts.generate-images")
generate_images = generate_images_module.generate_images


classes_names = [
    'cassette player', 'chain saw', 'church', 'gas pump', 'tench', 'garbage truck', 'english springer',
    'golf ball', 'parachute', 'french horn'
]
classes_ids = torch.tensor([
    482, 491, 497, 571, 0, 569, 217, 574, 701, 566
])

classes_codes = dict(zip(classes_names, classes_ids))

PROMPTS_PATH = 'data/100_batch_25-object-removal-prompts.csv'
SAVE_DIR = 'images/generations'
BATCH_SIZE = 25
NUM_CLASSES = len(list(classes_codes.values()))
OUTPUT_DIR = 'results/'

def get_accuracy_of_class(eval_model: torch.nn.Module, gen_model_name: str, local_cls_id: int):
    dir = os.path.join(SAVE_DIR, gen_model_name)
    local_pred_total, absolute_pred_total = [], []
    images = []
    print('i calculate accuracy')
    for i, fname in enumerate(filter(lambda x: x.startswith(f'{local_cls_id}'), os.listdir(dir))):
        im = torchvision.io.read_image(os.path.join(SAVE_DIR, gen_model_name, fname))
        images.append(im)
        if (i+1)%BATCH_SIZE:
            images = torch.stack(images).float() / 255
            with torch.no_grad():
                pred_probs = eval_model(images)
                local_pred = pred_probs[:, classes_ids].argmax(dim=1)
                absolute_pred = pred_probs.argmax(dim=1)

                local_pred_total.append(local_pred)
                absolute_pred_total.append(absolute_pred)
            images = []

    absolute_cls_id = classes_ids[local_cls_id]
    print('i have arrays now, number of batches: ', len(absolute_pred_total))
    print('='*20)
    absolute_pred = torch.stack(absolute_pred_total)
    local_pred = torch.stack(local_pred_total)

    absolute_acc = torch.mean((absolute_pred == absolute_cls_id).float()).item()
    local_acc = torch.mean((local_pred == local_cls_id).float()).item()
    return absolute_acc, local_acc




def accuracies(model_name: str, removed_class_name: str):
    removed_class_local_id = classes_names.index(removed_class_name)
    model = resnet50(pretrained=True).eval()

    absolute_accuracy_removed, local_accuracy_removed = get_accuracy_of_class(model, model_name, removed_class_local_id)

    absolute_accuracies, local_accuracies = [], []
    for cls_id in filter(lambda x: x != removed_class_local_id, range(NUM_CLASSES)):
        absolute_accuracy, local_accuracy = get_accuracy_of_class(model, model_name, cls_id)

        absolute_accuracies.append(absolute_accuracy)
        local_accuracies.append(local_accuracy)

    absolute_accuracy_other, local_accuracy_other = np.mean(absolute_accuracies), np.mean(local_accuracies)

    result = {
        'local': {
            'acc_removed': local_accuracy_removed,
            'acc_other': local_accuracy_other
        },
        'absolute': {
            'acc_removed': absolute_accuracy_removed,
            'acc_other': absolute_accuracy_other
        }
    }

    return result

def main():
    parser = argparse.ArgumentParser(
        prog='evaluateObjectRemoval',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--models_path', help='name of model', type=str, required=True)
    parser.add_argument('--removed_class_name', help='name of class to remove', type=str, required=True)

    args = parser.parse_args()

    model_name = args.model_name
    models_path = args.models_path
    removed_class_name = args.removed_class_name

    model_save_dir = os.path.join(SAVE_DIR, model_name)
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)

    generate_images(model_name, models_path, PROMPTS_PATH, SAVE_DIR, num_samples=BATCH_SIZE)
    results = accuracies(model_name, removed_class_name)

    filepath = os.path.join(OUTPUT_DIR, f'{removed_class_name}.json')
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
