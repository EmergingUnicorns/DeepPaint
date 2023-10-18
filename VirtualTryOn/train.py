
from default_config import get_config_default
import os
import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from data import PromptDataset
from utils import *

class VirtualTryOnTrain:
    def __init__(self) -> None:
        
        self.logger = get_logger(__name__)

        args = get_config_default()
        self.args = args
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank

        if args.instance_data_dir is None:
            raise ValueError("You must specify a train data directory.")

        if args.with_prior_preservation:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")

        print ("---- Initializing Accelerator --------")
        self.initialize_accelerator()
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
        print ("Done !!")

        print ("--------- Setting Seed --------")
        if args.seed is not None:
            set_seed(args.seed)
        print ("Done !!")

        print ("Prior Preservation : " + str(args.with_prior_preservations))
        if args.with_prior_preservation:
            print ("Class Data Path : " + str(args.class_data_dir))
            print ("--------------------------------------------------------")
            class_images_dir = Path(args.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))
            print ("Class Images in Dir : " + str(cur_class_images))
            print ("Num Class Images should be : " + str(args.num_class_images))
            print ("Generating Class Images ---> " + str(args.num_class_images - cur_class_images))
            print ("Class Prompts : " + str(self.args.class_prompt))
            print ("Sample Batch Size : " + str(self.args.sample_batch_size))
            print ("Resolution : " + str(self.args.resolution))
            if cur_class_images < args.num_class_images:
                self.generate_class_images(
                    cur_class_images = cur_class_images,
                    class_images_dir=class_images_dir
                )
            print ("--------------------------------------------------------")

            # Handle the repository creation
            if self.accelerator.is_main_process:
                if args.output_dir is not None:
                    os.makedirs(args.output_dir, exist_ok=True)

                if args.push_to_hub:
                    repo_id = create_repo(
                        repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
                    ).repo_id

            self.load_tokenizer()
            self.load_models()


    def load_tokenizer(self):
        if self.args.tokenizer_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.tokenizer_name)
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")

    def load_models(self):
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        self.vae.requires_grad_(False)
        if not self.args.train_text_encoder:
            self.text_encoder.requires_grad_(False)


    def generate_class_images(self, cur_class_images, class_images_dir):
        torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
        )
        pipeline.set_progress_bar_config(disable=True)
        num_new_images = self.args.num_class_images - cur_class_images
        self.logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=self.args.sample_batch_size, num_workers=1
        )
        sample_dataloader = self.accelerator.prepare(sample_dataloader)
        pipeline.to(self.accelerator.device)
        transform_to_pil = transforms.ToPILImage()
        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
        ):
            bsz = len(example["prompt"])
            fake_images = torch.rand((3, self.args.resolution, self.args.resolution))
            transform_to_pil = transforms.ToPILImage()
            fake_pil_images = transform_to_pil(fake_images)

            fake_mask = random_mask((self.args.resolution, self.args.resolution), ratio=1, mask_full_image=True)

            images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def initialize_accelerator(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(total_limit=self.args.checkpoints_total_limit)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
            project_config=project_config,
        )

    def train(self):
        pass

t = VirtualTryOnTrain()

        