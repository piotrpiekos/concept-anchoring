#!/bin/bash
#SBATCH --partition=batch
#SBATCH -J train_erasing
#SBATCH -o logs/train_chatgpt.txt
#SBATCH -e logs/train_chatgpt.err
#SBATCH --time=03:59:59
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
echo "USER: ${USER}"

nvidia-smi
conda activate ldm

echo prompt=$1
PROMPT=$1
echo var_prompt=$PROMPT
echo openai_key
echo $OPENAI_API_KEY

OPENAI_API_KEY="$OPENAI_API_KEY" python train-scripts/train-esd-hard-negatives_chatgpt.py --prompt "$PROMPT" --model_path "/ibex/project/c2231/piekosp" --p_unlearn 0.5 --train_method 'full' --devices '0,1'
