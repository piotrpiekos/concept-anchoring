#!/bin/bash
#SBATCH --partition=batch
#SBATCH -J erasing_concepts
#SBATCH -o logs/eval_metrics.txt
#SBATCH -e logs/eval_metrics.err
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
echo "USER: ${USER}"

nvidia-smi
conda activate ldm

MODEL_NAME='compvis-word_chainsaw-negative_copingsaw;holesaw;keyholesaw-punlearn_0.5-method_full-sg_3-ng_1-iter_1000-lr_1e-05'
MODELS_PATH="/ibex/project/c2231/piekosp/models"
REMOVED_CLASS_NAME="chain saw"

export PYTHONPATH=.

python eval-scripts/object-removal-eval.py --model_name $MODEL_NAME --models_path $MODELS_PATH --removed_class_name "chain saw"
