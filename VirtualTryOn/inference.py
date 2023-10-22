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
    
    def reset(self):
        self.oh = None
        self.ow = None
        self.rxmin = None 
        self.rymin = None
        self.rh = None
        self.rw = None

    def preprocess(self, img, masked_img, cloth_mask):
        h,w = img.shape[:2]
        offset = 10
        non_zero_points = np.argwhere(cloth_mask)
        min_x = np.min(non_zero_points[:, 1]) - offset
        max_x = np.max(non_zero_points[:, 1]) + offset
        min_y = np.min(non_zero_points[:, 0]) - offset
        max_y = np.max(non_zero_points[:, 0]) + offset
        min_x = 0 if min_x < 0 else min_x
        min_y = 0 if min_y < 0 else min_y
        max_x = w-1 if max_x > w-1 else max_x
        max_y = h-1 if max_y > h-1 else max_y
        self.rxmin = min_x 
        self.rymin = min_y
        img = img[min_y: max_y, min_x: max_x]
        masked_img = masked_img[min_y: max_y, min_x: max_x]
        cloth_mask = cloth_mask[min_y: max_y, min_x: max_x]
        new_height = max_y - min_y
        new_width = max_x - min_x
        self.rh = new_height
        self.rw = new_width

        masked_img[masked_img == 0] = 255.0
        masked_img = masked_img.astype(np.uint8)

        img = cv2.resize(img, (512, 512))
        masked_img = cv2.resize(masked_img, (512,512))
        cloth_mask = cv2.resize(cloth_mask, (512, 512))

        return img, masked_img, cloth_mask
    
    def postprocess(self, pred_image, img, pimg, pcloth_mask):
        print (pred_image.shape)
        print (pimg.shape)
        print (pcloth_mask.shape)
        pimg[pcloth_mask[:,:,0]==255] = pred_image[pcloth_mask[:,:,0]==255]
        pimg = cv2.resize(pimg, (self.rw, self.rh))
        img[self.rymin:self.rymin+self.rh , self.rxmin:self.rxmin+self.rw] = pimg
        cv2.imwrite("./pimg.jpg", img)
        img = cv2.resize(img, (self.ow, self.oh), interpolation = cv2.INTER_CUBIC)
        return img

    def _inference(self, pimg, pmasked_img, pcloth_mask, meta_prompt):
        pimg, pmasked_img, pcloth_mask = convert_numpy_to_PIL([pimg, pmasked_img, pcloth_mask])
        print ("Running the model!")
        new_img = self.model(
            prompt = meta_prompt,
            image = pimg if self.run_on == "original" else pmasked_img,
            mask_image = pcloth_mask,
            num_inference_steps = self.num_inference_steps,
            guidance_scale = self.guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(self.seed)
        ).images[0]


        final_image = np.array(new_img)
        return final_image

    def infer(self, person_img_path = None, meta_prompt = None):
        print ("Doing Inference ::: ")
        print ("Person Image Path : " + str(person_img_path))
        print ("Meta Prompt : " + str(meta_prompt))

        if (person_img_path is None or meta_prompt is None):
            print ("Please provide all inputs for generating Inference!")
            return None

        img = read_img_rgb(person_img_path)
        self.oh, self.ow = img.shape[:2]
        img = cv2.resize(img, (512, 512))
        masked_img, cloth_mask = self.human_parser.infer(img)
        img = img.astype(np.uint8)
        masked_img = masked_img.astype(np.uint8)
        cloth_mask = cloth_mask.astype(np.uint8)

        # cv2.imwrite("./Img.jpg", img)
        # cv2.imwrite("./Img1.jpg", masked_img)
        # cv2.imwrite("./Img2.jpg", cloth_mask)

        pimg, pmasked_img, pcloth_mask = self.preprocess(img, masked_img, cloth_mask)
        cv2.imwrite("./AImg.jpg", pimg)
        cv2.imwrite("./AImg1.jpg", pmasked_img)
        cv2.imwrite("./AImg2.jpg", pcloth_mask)

        
        final_image = self._inference(pimg, pmasked_img, pcloth_mask, meta_prompt)
        final_image = self.postprocess(final_image, img, pimg, pcloth_mask)

        # # Refinement Process
        # img, cloth_mask = convert_PIL_to_numpy([img, cloth_mask])
        # final_image = new_img * (cloth_mask/255.0) + img * ((np.array([255, 255, 255]) - cloth_mask)/255.0)
        return final_image






        