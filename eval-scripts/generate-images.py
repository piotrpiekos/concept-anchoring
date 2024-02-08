from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os

DEFAULT_GUIDANCE_SCALE = 0.7
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DDIM_STEPS = 100

BATCH_SIZE = 25

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from collections import defaultdict


class GenerativeModel:
    def __init__(self, model_name, models_path, guidance_scale=7.5, image_size=512, ddim_steps=100):
        if model_name == 'SD-v1-4':
            dir_ = "CompVis/stable-diffusion-v1-4"
        elif model_name == 'SD-V2':
            dir_ = "stabilityai/stable-diffusion-2-base"
        elif model_name == 'SD-V2-1':
            dir_ = "stabilityai/stable-diffusion-2-1-base"
        else:
            dir_ = "CompVis/stable-diffusion-v1-4"  # all the erasure models built on SDv1-4

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")
        if 'SD' not in model_name:
            try:
                model_path = f'{models_path}/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
                self.unet.load_state_dict(torch.load(model_path))
            except Exception as e:
                print(f'Model path is not valid, please check the file name and structure: {e}')
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                              num_train_timesteps=1000)

        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.ddim_steps = ddim_steps

    def to(self, device):

        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        self.device = device

    def generate(self, prompt, num_samples, seed):
        prompt = [str(prompt)] * num_samples

        generator = torch.manual_seed(seed)

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * num_samples, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        print('i am in the loop')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (num_samples, self.unet.in_channels, self.image_size // 8, self.image_size // 8),
            generator=generator,
        )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(self.ddim_steps)

        latents = latents * self.scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        self.scheduler.set_timesteps(self.ddim_steps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu()
        return image


def generate_images(model_name, models_path, prompts_path, save_path, device='cuda:0', guidance_scale=7.5,
                    image_size=512, ddim_steps=100, num_samples=10, from_case=0):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.




    if model_name == 'SD-v1-4':
        dir_ = "CompVis/stable-diffusion-v1-4"
    elif model_name == 'SD-V2':
        dir_ = "stabilityai/stable-diffusion-2-base"
    elif model_name == 'SD-V2-1':
        dir_ = "stabilityai/stable-diffusion-2-1-base"
    else:
        dir_ = "CompVis/stable-diffusion-v1-4"  # all the erasure models built on SDv1-4

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")
    if 'SD' not in model_name:
        try:
            model_path = f'{models_path}/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    '''

    model = GenerativeModel(model_name, models_path)
    model.to(device)

    df = pd.read_csv(prompts_path)

    folder_path = f'{save_path}/new_{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    case_imageid = defaultdict(int)

    for _, row in df.iterrows():
        case_number = row.case_number
        if case_number < from_case:
            continue

        """
        prompt = [str(row.prompt)] * num_samples
        seed = row.evaluation_seed

        height = image_size  # default height of Stable Diffusion
        width = image_size  # default width of Stable Diffusion

        num_inference_steps = ddim_steps  # Number of denoising steps

        guidance_scale = guidance_scale  # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        print('i am in the loop')
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        """
        image = model.generate(row.prompt, num_samples, row.evaluation_seed)
        image = image.permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        for im in pil_images:
            im_num = case_imageid[case_number]

            saving_path = f"{folder_path}/{case_number}_{im_num}.png"
            print('Saving. path: ', saving_path)
            im.save(saving_path)
            case_imageid[case_number] += 1


def batch_generate_images(model_name, models_path, prompts_path, num_samples, device='cuda:0'):
    """
    if model_name == 'SD-v1-4':
        dir_ = "CompVis/stable-diffusion-v1-4"
    elif model_name == 'SD-V2':
        dir_ = "stabilityai/stable-diffusion-2-base"
    elif model_name == 'SD-V2-1':
        dir_ = "stabilityai/stable-diffusion-2-1-base"
    else:
        dir_ = "CompVis/stable-diffusion-v1-4"  # all the erasure models built on SDv1-4

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet")
    if 'SD' not in model_name:
        try:
            model_path = f'{models_path}/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    """

    model = GenerativeModel(model_name, models_path)
    model.to(device)

    df = pd.read_csv(prompts_path)

    images_dict = dict()
    for _, row in df.iterrows():
        prompt = row['prompt']
        cls = row['class']
        seed = row['evaluation_seed']

        images_dict[cls] = batch_generate_prompt_images(model, prompt, num_samples, seed)

    return images_dict


def batch_generate_prompt_images(model: GenerativeModel, prompt, num_samples, evaluation_seed):
    """
    Returns a tensor of images (doesnt save it to a file). Uses batched generation for speedups
    """
    assert num_samples % BATCH_SIZE == 0

    images_list = []
    for batch_num in range(num_samples // BATCH_SIZE):
        images = model.generate(prompt, BATCH_SIZE, evaluation_seed)
        images_list.append(images)

    return torch.cat(images_list, dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--models_path', help='path to the models directory', type=str, default='models')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=100)
    args = parser.parse_args()

    model_name = args.model_name
    models_path = args.models_path
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case

    generate_images(model_name, models_path, prompts_path, save_path, device=device,
                    guidance_scale=guidance_scale, image_size=image_size, ddim_steps=ddim_steps,
                    num_samples=num_samples, from_case=from_case)
