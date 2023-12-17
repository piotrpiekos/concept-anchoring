# Concept anchoring prevents unlearning leakage - CS 325 project

The code is an extension of the code base of [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing)

## Installation

1. go through the installation of https://github.com/rohitgandikota/erasing 
2. ``pip install openai``
3. get the openai token and put in the environmental variable OPENAI_API_KEY

## Training

For training use the script ``train-scripts/train-esd-hard-negatives_chatgpt.py``. It requires 2 gpus with 32Gbs memory. Example usage: 

```
python train-scripts/train-esd-hard-negatives_chatgpt.py --prompt 'your prompt' --train_method 'your choice of training' --devices '0,1'
```
The model will be saved at ``f'compvis-word_{prompt}-negative_{negative_prompt}-punlearn_{p_unlearn}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'``

## Generation 

let MODEL_PATH be the directory to the unlearned model (directory that starts with ``compvis`` from the step above).

--prompts_path specifies path to the csv file with prompts to generate, some examples are in the data directory

--save_path is the directory where generated image should be saved
```
python eval-scripts/generate-images.py --model_name=$MODEL_NAME --prompts_path 'data/art_prompts.csv' --save_path 'evaluation_folder' --num_samples 10
```