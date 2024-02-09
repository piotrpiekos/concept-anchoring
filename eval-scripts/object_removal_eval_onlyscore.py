import argparse

import importlib
import os

import json

object_removal_eval = importlib.import_module("eval-scripts.object-removal-eval")
get_accuracies = object_removal_eval.accuracies

OUTPUT_DIR = 'results/'


def main():
    parser = argparse.ArgumentParser(
        prog='evaluateObjectRemoval',
        description='only evaluate already generated images ')
    parser.add_argument('--model_name', help='name of the model that generated images, used to find a path of images', type=str, required=True)
    parser.add_argument('--removed_class_name', help='name of class to remove', type=str, required=True)

    args = parser.parse_args()

    model_name = args.model_name
    removed_class_name = args.removed_class_name

    results = get_accuracies(model_name, removed_class_name)
    filepath = os.path.join(OUTPUT_DIR, f'{removed_class_name}.json')
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
