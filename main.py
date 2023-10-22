import cv2
from VirtualTryOn import VirtualTryOnInference, VirtualTryOnTrain, get_config_default, DataCreation
import argparse
import os

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def Train(output_dir, instance_dir, instance_prompt):
    model_name = "runwayml/stable-diffusion-inpainting"
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
    params.max_train_steps = 900
    # params.train_text_encoder = True
    vt = VirtualTryOnTrain(params)
    vt.train()

def Inference(img_path, model_path, instance_prompt, output_path, prefix):
    # for idx in range(0, 10):
    idx = 0
    vt = VirtualTryOnInference(
        model_path=model_path,
        device = "cuda",
        run_on="moriginal",
        num_inference_steps=50,
        guidance_scale=20,
        seed = idx
    )

    MakeDir(output_path)
    img_name = img_path.split("/")[-1].split(".")[-2]
    prompt = instance_prompt
    # img_path = "./DebugImages/Images/img2.jpg"
    # # img_path = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Shirt3/a.png"
    res_img = vt.infer(img_path, prompt)
    cv2.imwrite(output_path + "/" + prefix + img_name + ".jpg", res_img)


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="do_training", action="store_true")
    parser.add_argument("--infer", help="do_inference", action="store_true")
    parser.add_argument("--instance_dir", help="instance dir", default=None)
    parser.add_argument("--model_dir", help="model_dir", default=None)
    parser.add_argument("--prompt", help="instance_prompt", default=None)
    parser.add_argument("--img_path", help="image path", default=None)
    parser.add_argument("--infer_output", help="infer output", default=None)
    parser.add_argument("--infer_output_prefix", help="infer output", default="res")

    args = parser.parse_args()
    if args.train:
        Train(output_dir = args.model_dir, 
              instance_dir = args.instance_dir, 
              instance_prompt = args.prompt
            )

    if (args.infer):

        Inference(
            img_path = args.img_path,
            model_path=args.model_dir,
            instance_prompt=args.prompt,
            output_path=args.infer_output,
            prefix = args.infer_output_prefix
        )

    # # instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Jackets/"
    # instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/CheckShirt/"
    # # instance_dir = "./DataCreation/"
    # # instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Shirt2/"

    # output_dir = "/data/Kaggle/StableDiff/Shirt_Outputs1/"
    # instance_prompt = "UBIAA shirt"
    # # instance_prompt = "UBIAA jacket, high resolution"

    # # d = DataCreation(
    # #     instance_dir= instance_dir,
    # #     save_dir="./DataCreation/",
    # #     target_number=100
    # # )

    # # d.create()
