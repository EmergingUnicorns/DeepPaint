from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        
        print ("Instance Data Dir : " + str(instance_data_root))
        print ("Instance Prompt : " + str(instance_prompt))
        print ("Resolution : " + str(size))
        print ("Center Crop : " + str(center_crop))

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        print (self.num_instance_images)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

import os
import glob
from HumanParser import HumanParser
from .utils import *
import albumentations as A

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

class DataCreation:
    def __init__(self, instance_dir, save_dir, target_number = 20) -> None:
        print ("-------- DATA CREATION INITIALIZATION ----------")
        print ("Instance Dir : " + str(instance_dir))
        MakeDir(save_dir)

        self.all_imf = []
        self.all_imf += glob.glob(instance_dir + "*.png")
        self.all_imf += glob.glob(instance_dir + "*.jpeg")
        self.all_imf += glob.glob(instance_dir + "*.jpg")

        print ("Number of Images found : " + str(len(self.all_imf)))
        self.save_dir = save_dir
        self.human_parser = HumanParser()
        self.target_number = target_number
        print ("Number of Images to be generated : " + str(target_number)) 
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.6, 1.2), rotate = (-45, 45), translate_percent = 0.05)
        ])       
        print ("-------------------------------------- \n")

    def reset(self):
        self.c = 0
        self.cp = []

    def create_augmentation(self, img, save_idx):
        c = 0
        while (c < 10):
            transformed_image =self.transform(image=img)['image']
            transformed_image[transformed_image == 0] = 255
            cv2.imwrite(self.save_dir + "/cloth_da_" + str(save_idx) + "_" + str(c) + ".png", transformed_image)
            self.c += 1
            c += 1


    def create(self):
        print ("Creating Dataset!!")
        self.reset()
        for idx, imf in enumerate(self.all_imf):
            img = read_img_rgb(imf, resize = (512, 512))
            masked_img, cloth_mask = self.human_parser.infer(img)
            cloth_img = img * (cloth_mask/255.0)
            cloth_img = cloth_img + ([255,255,255] - cloth_mask)
            non_zero_points = np.argwhere(cloth_mask)
            min_x = np.min(non_zero_points[:, 1])
            max_x = np.max(non_zero_points[:, 1])
            min_y = np.min(non_zero_points[:, 0])
            max_y = np.max(non_zero_points[:, 0])
            bbox = [min_x, min_y, max_x, max_y]

            cropped_region = cloth_img[min_y:max_y, min_x:max_x]
            cropped_region = cv2.resize(cropped_region, (512, 512), interpolation = cv2.INTER_AREA)    
            cv2.imwrite(self.save_dir + "/cloth_" + str(idx) + ".png", cropped_region)
            self.cp.append(self.save_dir + "/cloth_" + str(idx) + ".png")
            self.c += 1
        
        if (self.c >=  self.target_number):
            print ("Dataset Created : " + str(self.c))
        else:
            print ("Need to generate augmented dataset : " + str(self.target_number - self.c))
            random.shuffle(self.cp)
            for idx in range(len(self.cp)):
                img = read_img_rgb(self.cp[idx])
                self.create_augmentation(img, idx)
                if (self.c >= self.target_number):
                    break

