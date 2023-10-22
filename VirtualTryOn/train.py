
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

from .data import PromptDataset, DreamBoothDataset
from .utils import *

class VirtualTryOnTrain:
    def __init__(self, args):
        print ("------ || Virtual Try ON Training || ----------------")
        print (args)
        self.logger = get_logger(__name__)

        if (args is None):
            print ("Please provide arguments!!")
            exit()

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

        self.initialize_accelerator()
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        print ("--------- Setting Seed --------")
        if args.seed is not None:
            set_seed(args.seed)
        print ("Done !!")

        print ("Prior Preservation Flag : " + str(args.with_prior_preservation))
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
            print ("Output Dir : " + str(self.args.output_dir))
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

            print ("Push to Hub : " + str(args.push_to_hub))
            if args.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
                ).repo_id

        self.load_tokenizer()
        self.load_models()
        self.init_opt_and_lr()
        self.initialize_dreambooth_loaders()
        self.init_lr_scheduler()
        self.init_noise_scheduler()

        if self.args.train_text_encoder:
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth", config=vars(self.args))

        print ("=========== Virtual Try ON is ready for Training. COOL!!!!! ========== \n \n")


    def init_noise_scheduler(self):
        print ("-------- Initializing Noise Scheduler --------- ")
        print ("DDPM Scheduler")
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        print ("------------------------------------------------ \n")

    def init_lr_scheduler(self):
        print ("----- Initializing Learning Rate Scheduler -----------")
        # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        print ("Lr warmup steps : " + str(self.args.lr_warmup_steps * self.args.gradient_accumulation_steps))
        print ("Number Training Steps : " + str(self.args.max_train_steps * self.args.gradient_accumulation_steps))
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        print ("----------------------------------------------------- \n \n")

    def initialize_dreambooth_loaders(self):
        print ("-------------- Initializing Dreambooth Loaders ---------------- ")
        self.train_dataset = DreamBoothDataset(
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
                pil_image.save("./check.jpg")

                # mask = clipseg_masks(pil_image) # generate a random mask
                mask = random_mask(pil_image, pil_image.size, 1, False) # generate a random mask
                mask.save("./check1.jpg")

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
            self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=collate_fn
        )
        print ("----------------------------------------------- \n \n")


    def init_opt_and_lr(self):
        print ("----------------- Initialize Optimizer and Learning Rate ------------------------")
        if self.args.scale_lr:
            print ("Scale Learning Rate : " + str(self.args.scale_lr))
            print ("Before Scaling : " + str(self.args.learning_rate))
            print ("Gradient Accumulation Steps : " + str(self.args.gradient_accumulation_steps))
            print ("Training Batch Size : " + str(self.args.train_batch_size))
            print ("Accelerator Num Proc : " + str(self.accelerator.num_processes))            
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )
            print ("After Scaling Lr : " + str(self.args.learning_rate))

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            print ("Using 8 bit ADAM Optimizer!")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            print ("Using Normal ADAM Optimizer!")
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if self.args.train_text_encoder else self.unet.parameters()
        )
        print ("Adam Beta 1 : " + str(self.args.adam_beta1))
        print ("Adam Beta 2 : " + str(self.args.adam_beta2))
        print ("Adam Weight Decay : " + str(self.args.adam_weight_decay))
        print ("Eps : " + str(self.args.adam_epsilon))
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        print ("-------------------------------------- \n \n")

    def load_tokenizer(self):
        print ("--------- Loading Tokenizer ---------")
        if self.args.tokenizer_name:
            print ("Tokenizer Name : " + str(self.args.tokenizer_name))
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.tokenizer_name)
        elif self.args.pretrained_model_name_or_path:
            print ("Tokenizer from Model : " + str(self.args.pretrained_model_name_or_path))
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        print ("-------------------------------------- \n \n")

    def load_models(self):
        print ("------------ Loading Models ------------------")
        print ("Loading Model : " + str(self.args.pretrained_model_name_or_path))
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        print ("===VAE is freezed===")
        self.vae.requires_grad_(False)

        if not self.args.train_text_encoder:
            print ("===Text Encoder is freezed===")
            self.text_encoder.requires_grad_(False)

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
        print ("----------------------------------------------- \n \n")


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
        print ("---- Initializing Accelerator --------")
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(total_limit=self.args.checkpoints_total_limit)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
            project_config=project_config,
        )
        print ("Logging Dir : " + str(logging_dir))
        print ("Checkpoints Total Limit : " + str(self.args.checkpoints_total_limit))
        print ("Mixed Precision : " + str(self.args.mixed_precision))
        print ("Gradient Accumulation Steps : " + str(self.args.gradient_accumulation_steps))
        print ("Done !!")
        print ("--------------------------------------- \n \n")


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

        progress_bar = tqdm(range(global_step, self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Convert masked images to latent space
                    masked_latents = self.vae.encode(
                        batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor

                    masks = batch["masks"]
                    # resize the mask to latents shape as we concatenate the mask to the latents
                    mask = torch.stack(
                        [
                            torch.nn.functional.interpolate(mask, size=(self.args.resolution // 8, self.args.resolution // 8))
                            for mask in masks
                        ]
                    )
                    mask = mask.reshape(-1, 1, self.args.resolution // 8, self.args.resolution // 8)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0] # Batch size
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    # concatenate the noised latents with the mask and the masked latents
                    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    if self.args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if self.args.train_text_encoder
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
    
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    # if global_step % self.args.checkpointing_steps == 0:
                    #     if self.accelerator.is_main_process:
                    #         save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    #         self.accelerator.save_state(save_path)
                    #         self.logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

            self.accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            )
            pipeline.save_pretrained(self.args.output_dir)

            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.rgs.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()

        pass


        