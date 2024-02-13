#!/bin/bash
#SBATCH --partition=batch
#SBATCH -J erasing_concepts
#SBATCH -o eval_metrics2.txt
#SBATCH -e eval_metrics2.err
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --account conf-iclr-2023.09.29-schmidhj

eval "$(conda shell.bash hook)"
echo "USER: ${USER}"

nvidia-smi
conda activate ldm

REMOVED_CLASS_NAME=$1
STRIPED_REMOVED_CLASS_NAME="${REMOVED_CLASS_NAME// /}"

MODEL_NAME=compvis-word_$STRIPED_REMOVED_CLASS_NAME
MODELS_DIR="/ibex/project/c2231/piekosp/models"

echo $REMOVED_CLASS_NAME
echo $STRIPED_REMOVED_CLASS_NAME
echo $MODEL_NAME

export PYTHONPATH=.

python eval-scripts/object-removal-eval.py --model_name $MODEL_NAME --models_path $MODELS_DIR --removed_class_name "$REMOVED_CLASS_NAME"
