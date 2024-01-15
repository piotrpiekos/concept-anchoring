import argparse

import importlib
import pandas as pd
import os

import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch
generate_images_module = importlib.import_module("eval-scripts.generate-images")
generate_images = generate_images_module.generate_images

PROMPTS_PATH = 'data/unique-object-removal-prompts.csv'
SAVE_DIR = 'images/generations'
NUM_SAMPLES = 500

classes_codes = {
 'cassette player': 482,
 'chain saw': 491,
 'church': 497,
 'gas pump': 571,
 'tench': 0,
 'garbage truck': 569,
 'English springer': 217,
 'golf ball': 574,
 'parachute': 701,
 'French horn': 566
}

def get_predictions():
    images = []
    for cls_id in range(10):
        for sample_num in range(NUM_SAMPLES):
            fname = f'f{cls_id}_{sample_num}'
            im = torchvision.io.read_image(os.path.join(SAVE_DIR, fname))
            images.append(im)
    images = torch.stack(images)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()

    preds = model(images)

    all_classes = torch.tensor(list(classes_codes.values()))
    preds = all_classes[preds[:, all_classes].argmax()]

    return preds



def evaluate_object_removal(model_name, removed_class_name):
    generate_images(model_name, PROMPTS_PATH, SAVE_DIR, num_samples=NUM_SAMPLES)
    df = pd.read_csv(PROMPTS_PATH)
    for i, image_class in df['class']:
        imagenet_class_code = classes_codes[image_class]
        if image_class == removed_class_name:
            # get accuracy on the unlearned class

        else:
            # other classes


parser = argparse.ArgumentParser(
    prog='evaluateObjectRemoval',
    description='Generate Images using Diffusers Code')
parser.add_argument('--model_name', help='name of model', type=str, required=True)
parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
args = parser.parse_args()
