from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# prompt = "5 people wearning a photo of sks t-shirt, people wearning t-shirt, out 5 people 3 are male and 2 are female"
# image = pipe(prompt, num_inference_steps=50, guidance_scale=8.5).images[0]

# image.save("dog-bucket.png")


# device = "cuda"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
# ).to(device)


# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open("testimg.jpg").convert("RGB")
# init_image = init_image.resize((768, 512))

# prompt = "1 person wearing sks t-shirt, person with t-shirt"

# images = pipe(prompt=prompt, image=init_image, strength=0.05, guidance_scale=10.5).images

# images[0].save("fantasy_landscape.png")

from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    image =  Image.open(url).convert("RGB")

    # right = 200
    # left = 200
    # top = 50
    # bottom = 50
    right = 0
    left = 0
    top = 0
    bottom = 0
    
    width, height = image.size
    
    new_width = width + right + left
    new_height = height + top + bottom
    
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))

    return result

img_url = "./A1.jpg"
mask_url = "./mask0.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))


# def download_image(url):
#     response = requests.get(url)
#     return Image.open(BytesIO(response.content)).convert("RGB")


# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = download_image(img_url).resize((512, 512))
# mask_image = download_image(mask_url).resize((512, 512))

# model_id = "/data/Kaggle/StableDiff/ResultsShirt3_2/"
# model_id = "/data/Kaggle/StableDiff/ResultsJacketDenim1/"
model_id = "/data/Kaggle/StableDiff/ResultsJacket3_2/"
# model_id = "/data/Kaggle/StableDiff/ResultsShirt3_2/"
# model_id = "/data/Kaggle/StableDiff/ResultsShirtStripe/"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker = None,
)
pipe = pipe.to("cuda")


# prompt = "Brown Solid colour ska jacket, Shell jacket, Stand-up collar, Shell: Polyester 100%, Lining: Polyamide 83%, Elastane 17%, Long sleeve, white background"
prompt = "Brown Solid colour ska jacket"
# prompt = "Light blue/Green Patterned Trees ska shirt, Lyocell 100% on white background"
# prompt = "Dark denim blue Solid colour ska jacket"
# prompt = "SEA GREEN TEXTURED CROCHET SKA JACKET"
# prompt = "Beige/Black/White Striped ska shirt"

image = pipe(prompt=prompt, image=init_image, mask_image=mask_image,num_inference_steps = 50, guidance_scale = 10, generator=torch.Generator(device="cuda").manual_seed(10)).images[0]

init_image = np.asarray(init_image)
mask_image = np.asarray(mask_image)
res_image = np.asarray(image)


# init_image[np.where(mask_image == 255)] = res_image[np.where(mask_image == 255)]
final_image = res_image * (mask_image/255.0) + init_image * ((np.array([255, 255, 255]) - mask_image)/255.0)

final_image1 = np.zeros((512, 512 * 2, 3), dtype = np.uint8)
final_image1[:512, :512] = init_image
final_image1[:512, 512:] = final_image

# final_image = init_image
final_image = Image.fromarray(final_image1.astype('uint8'))
final_image.save("./Final1.jpg")
# image = Image.composite(init_image, image, mask_image)


# image.resize((512, 512)).save("fantasy_landscape.png")
