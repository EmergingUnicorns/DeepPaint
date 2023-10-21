## Main Flow of Virtual Try On using dreambooth

from diffusers import StableDiffusionInpaintPipeline
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from HumanParser import HumanParser
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from .utils import *


class VirtualTryOnInference:
    def __init__(self, model_path, 
                 device = "cuda", 
                 run_on = "original",
                 num_inference_steps = 50,
                 guidance_scale = 10,
                 seed = 10) -> None:
        
        print ("------ Initializing Virtual Try on Inference -------")
        print ("Model Path : " + str(model_path))
        print ("Device : " + str(device))
        self.device = device
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype = torch.float32,
            safety_checker = None
        )
        self.model = pipeline
        self.model = self.model.to(device)
        self.human_parser = HumanParser()
        self.run_on = run_on
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        print ("---------------------------------------")

    def infer(self, person_img_path = None, meta_prompt = None):
        if (person_img_path is None or meta_prompt is None):
            print ("Please provide all inputs for generating Inference!")
            return None

        img = read_img_rgb(person_img_path, resize = (512, 512))
        masked_img, cloth_mask = self.human_parser.infer(img)
        img = img.astype(np.uint8)
        masked_img = masked_img.astype(np.uint8)
        cloth_mask = cloth_mask.astype(np.uint8)

        cv2.imwrite("./masked_img1.jpg", masked_img)
        masked_img = img * (cloth_mask/255)
        masked_img = (masked_img/255) + (1 - (cloth_mask/255))
        masked_img = masked_img * 255.0
        masked_img = masked_img.astype(np.uint8)
        cv2.imwrite("./masked_img.jpg", masked_img)
        # exit()

        
        img, masked_img, cloth_mask = convert_numpy_to_PIL([img, masked_img, cloth_mask])
        print ("Running the model!")
        new_img = self.model(
            prompt = meta_prompt,
            image = img if self.run_on == "original" else masked_img,
            mask_image = cloth_mask,
            num_inference_steps = self.num_inference_steps,
            guidance_scale = self.guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(self.seed)
        ).images[0]


        final_image = np.array(new_img)

        # # Refinement Process
        # img, cloth_mask = convert_PIL_to_numpy([img, cloth_mask])
        # final_image = new_img * (cloth_mask/255.0) + img * ((np.array([255, 255, 255]) - cloth_mask)/255.0)
        return final_image






        