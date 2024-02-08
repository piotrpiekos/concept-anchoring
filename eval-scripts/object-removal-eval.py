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

generate_images_module = importlib.import_module("eval-scripts.generate-images")
generate_images = generate_images_module.generate_images


classes_names = [
    'cassette player', 'chain saw', 'church', 'gas pump', 'tench', 'garbage truck', 'English springer',
    'golf ball', 'parachute', 'French horn'
]
classes_ids = [
    482, 491, 497, 571, 0, 569, 217, 574, 701, 566
]

classes_codes = dict(zip(classes_names, classes_ids))

PROMPTS_PATH = 'data/object-removal-prompts.csv'
SAVE_DIR = 'images/generations'
NUM_SAMPLES = 5
NUM_CLASSES = len(list(classes_codes.values()))
OUTPUT_DIR = 'results/'


def get_predictions(model_name):
    images = []
    correct_preds = []
    for cls_id in range(NUM_CLASSES):
        for sample_num in range(NUM_SAMPLES):
            fname = f'{cls_id}_{sample_num}.png'
            im = torchvision.io.read_image(os.path.join(SAVE_DIR, model_name, fname))
            images.append(im)
            correct_preds.append(cls_id)
    images = torch.stack(images)
    correct_preds = torch.tensor(correct_preds)

    #model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    model = resnet50(pretrained=True).eval()

    with torch.no_grad():
        preds = model(images.float()/255) # [num_images, num_class]

    all_classes = torch.tensor(classes_ids)
    print('images shape: ', images.shape)
    print('preds shape: ', preds.shape)
    local_ids = preds[:, all_classes].argmax(dim=1)

    return local_ids, correct_preds


def evaluate_object_removal(model_name, models_path, removed_class_name):
    print('GENERATING IMAGES')
    generate_images(model_name, models_path, PROMPTS_PATH, SAVE_DIR, num_samples=NUM_SAMPLES)
    print('GETTING PREDICTIONS')
    preds, correct_preds = get_predictions(model_name)
    is_pred_correct = preds == correct_preds

    df = pd.read_csv(PROMPTS_PATH)
    removed_object_ids = {row['class']: row['case_number'] for _, row in df.iterrows()}
    removed_object_id = removed_object_ids[removed_class_name]

    print('removed class name: ', removed_class_name)
    print('removed object_id: ', removed_object_id)

    print('df.shape', df.shape)
    print('is_pred_correct', is_pred_correct.shape)
    df['pred'] = preds
    df['pred_right'] = is_pred_correct
    print(df.head())

    acc_on_removed = df[df['class'] == removed_class_name]['pred_right'].mean()
    acc_on_other = df[df['class'] != removed_class_name]['pred_right'].mean()

    return acc_on_removed, acc_on_other


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

    acc_on_removed, acc_on_other = evaluate_object_removal(model_name, models_path, removed_class_name)
    filepath = os.path.join(OUTPUT_DIR, f'{removed_class_name}.json')
    results = {
        'acc_removed': acc_on_removed,
        'acc_other': acc_on_other
    }
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
