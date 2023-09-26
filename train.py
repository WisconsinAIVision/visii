import argparse
import glob
import json
import os
import sys

import numpy as np
import open_clip
import PIL
import requests
import torch
import yaml
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from visii import StableDiffusionVisii


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subfolder', type=str, default=None)
    parser.add_argument('--config_file', type=str, default='configs/config_ip2p.yaml')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--init_expname', type=str, default=None)
    parser.add_argument('--embedding_learning_rate', type=float, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    print(args)

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.subfolder is not None:
        config['data']['subfolder'] = args.subfolder
    if args.init_expname is not None:
        config['exp']['init_expname'] = args.init_expname
    if args.image_folder is not None:
        config['data']['image_folder'] = args.image_folder
    if args.embedding_learning_rate is not None:
        config['hyperparams']['embedding_learning_rate'] = args.embedding_learning_rate

    model_id = config['model']['model_id']
    subfolders = os.listdir(config['data']['image_folder'])
    subfolders = [x for x in subfolders if config['data']['subfolder'] in x]
    print('Train with subfolders: ', subfolders)

    for folder in subfolders:
        current_folder = os.path.join(config['data']['image_folder'], folder)

        # by default, load first image pairs as training pairs (0_0.png and 0_1.png)
        before_path = os.path.join(current_folder, '0_0.png')
        cond_image = Image.open(before_path).resize((512, 512)).convert('RGB')
        target_image = Image.open(before_path.replace('_0.', '_1.')).resize((512, 512)).convert('RGB')

        exp_name = '{}_{}_{}'.format(config['exp']['init_expname'], folder, before_path.split('/')[-1])
        log_dir = os.path.join(args.log_dir, exp_name)
        os.makedirs(log_dir, exist_ok=True)
        
        if config['exp']['prompt_type'] == 'hard':
            prompt = config['exp']['init_prompt']
            print('Initialize with hard prompt: ', prompt)
        elif config['exp']['prompt_type'] == 'learn':
            config_path = "./configs/hard_prompts_made_easy.json"
            from pez import *
            print("Finding initial caption...")
            args1 = argparse.Namespace()
            args1.__dict__.update(read_json(config_path))
            args1.print_new_best = False

            # load CLIP model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(args1.clip_model, pretrained=args1.clip_pretrain, device=device)

            print(f"Running for {args1.iter} steps.")

            learned_prompt = optimize_prompt(model, preprocess, args1, device, target_images=[target_image])
            print(learned_prompt)
            prompt = learned_prompt
            del model, preprocess, args1, learned_prompt
        else:
            print("What is your prompt?")
            exit()

        with open(os.path.join(log_dir, "learned_prompt.txt"), "w") as text_file:
            text_file.write("{}".format(prompt))
        
        print('Save learned prompt to: ', os.path.join(log_dir, "learned_prompt.txt"))
        print('Init prompt: ', prompt)
        pipe = StableDiffusionVisii.from_pretrained(model_id,
            torch_dtype=torch.float32).to("cuda")

        pipe.train(
            prompt=prompt,
            prompt_embeds=None,
            target_images=[target_image],
            cond_images=[cond_image],
            exp_name=exp_name,
            embedding_learning_rate=config['hyperparams']['embedding_learning_rate'],
            text_embedding_optimization_steps=config['hyperparams']['optimization_steps'],
            clip_loss=config['hyperparams']['clip_loss'],
            lambda_clip=config['hyperparams']['lambda_clip'],
            lambda_mse=config['hyperparams']['lambda_mse'],
            eval_step = config['hyperparams']['eval_step'],
            log_dir=args.log_dir,
            )

        if config['exp']['eval']:
            checkpoints = [os.path.join(log_dir, 'prompt_embeds_{}.pt'.format(x)) for x in np.arange(0, config['hyperparams']['optimization_steps'], config['hyperparams']['eval_step'])]
            after_images = np.concatenate([np.array(cond_image), np.array(target_image)], axis=1)
            location = os.path.join(log_dir, 'eval_{}'.format(config['hyperparams']['eval_step']) + '.png')
            for checkpoint in checkpoints:
                opt_embs = torch.load(checkpoint)
                after_image = pipe.test(prompt_embeds=opt_embs,
                        image=cond_image,
                        image_guidance_scale=1.5,
                        guidance_scale=7.5
                        ).images[0]
                after_images = np.concatenate([after_images, after_image], axis=1)

            Image.fromarray(after_images).save(location)
            print('Optimization progress is saved at: ', location)
