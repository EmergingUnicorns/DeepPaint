
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
            self.init_opt_and_lr()
            self.init_lr_scheduler()

            if self.args.train_text_encoder:
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                    self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            else:
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                    self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

            weight_dtype = torch.float32
            if args.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            self.vae.to(self.accelerator.device, dtype=weight_dtype)
            if not self.args.train_text_encoder:
                self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)


            num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
            if self.overrode_max_train_steps:
                self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

            # We need to initialize the trackers we use, and also store our configuration.
            # The trackers initializes automatically on the main process.
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers("dreambooth", config=vars(self.args))


    def init_lr_scheduler(self):
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )



    def initialize_dreambooth_loaders(self):
        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            instance_prompt=self.args.instance_prompt,
            class_data_root=self.args.class_data_dir if self.args.with_prior_preservation else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            if self.args.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]
                pior_pil = [example["class_PIL_images"] for example in examples]

            masks = []
            masked_images = []
            for example in examples:
                pil_image = example["PIL_images"]
                mask = random_mask(pil_image.size, 1, False) # generate a random mask
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)  # prepare mask and masked image
                masks.append(mask)
                masked_images.append(masked_image)

            if self.args.with_prior_preservation:
                for pil_image in pior_pil:
                    mask = random_mask(pil_image.size, 1, False)
                    mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
                    masks.append(mask)
                    masked_images.append(masked_image)

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            masks = torch.stack(masks)
            masked_images = torch.stack(masked_images)
            batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
            return batch


        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=collate_fn
        )



    def init_opt_and_lr(self):
        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if self.args.train_text_encoder else self.unet.parameters()
        )
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

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

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()


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
        print ("\n" * 3)        
        print ("-------------------------- TRAINING ----------------------------------------")
        # Train!
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        global_step = 0
        first_epoch = 0

        


        pass

t = VirtualTryOnTrain()

        