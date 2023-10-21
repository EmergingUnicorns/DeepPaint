import cv2
from VirtualTryOn import VirtualTryOnInference, VirtualTryOnTrain, get_config_default, DataCreation


def Train(model_name, output_dir, instance_dir, instance_prompt):
    params = get_config_default()
    params.pretrained_model_name_or_path = model_name
    params.instance_data_dir = instance_dir
    params.output_dir = output_dir
    params.instance_prompt = instance_prompt
    params.resolution = 512
    params.train_batch_size = 1
    params.gradient_accumulation_steps = 1
    params.learning_rate = 2e-6
    params.lr_scheduler = "constant"
    params.lr_warmup_steps = 0
    params.use_8bit_adam = True
    params.max_train_steps = 300
    # params.train_text_encoder = True
    vt = VirtualTryOnTrain(params)
    vt.train()

def Inference(model_path, instance_prompt):
    for idx in range(0, 10):
        vt = VirtualTryOnInference(
            model_path=model_path,
            device = "cuda",
            run_on="original",
            num_inference_steps=50,
            guidance_scale=10,
            seed = idx
        )

        prompt = instance_prompt
        img_path = "./DebugImages/Images/img1.jpg"
        # img_path = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Shirt3/a.png"
        res_img = vt.infer(img_path, prompt)
        cv2.imwrite("./res_" + str(idx) + ".jpg", res_img)

model_name = "runwayml/stable-diffusion-inpainting"
# instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/CheckShirt/"
# instance_dir = "./DataCreation/"
output_dir = "/data/Kaggle/StableDiff/Shirt_Outputs/"
instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Shirt3/"
instance_prompt = "a photo of UBIAA shirt"


# Train(model_name, output_dir, instance_dir, instance_prompt)
Inference(
    model_path=output_dir,
    instance_prompt=instance_prompt
)


# d = DataCreation(
#     instance_dir= "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Shirt3/",
#     save_dir="./DataCreation/",
#     target_number=100
# )

# d.create()

# exit()


# instance_prompt = "UBIAA shirt"
