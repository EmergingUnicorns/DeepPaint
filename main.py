import cv2
from VirtualTryOn import VirtualTryOnInference, VirtualTryOnTrain, get_config_default


model_name = "runwayml/stable-diffusion-inpainting"
instance_dir = "/home/user/anmol/StableDiff/d1/diffusers/examples/research_projects/dreambooth_inpaint/Jacket3/"
output_dir = "/data/Kaggle/StableDiff/Jacket3_Outputs/"
instance_prompt = "SEA GREEN TEXTURED CROCHET SKA JACKET"

# params = get_config_default()
# params.pretrained_model_name_or_path = model_name
# params.instance_data_dir = instance_dir
# params.output_dir = output_dir
# params.instance_prompt = instance_prompt
# params.resolution = 512
# params.train_batch_size = 1
# params.gradient_accumulation_steps = 1
# params.learning_rate = 2e-6
# params.lr_scheduler = "constant"
# params.lr_warmup_steps = 0
# params.use_8bit_adam = True
# params.max_train_steps = 1000
# vt = VirtualTryOnTrain(params)
# vt.train()

vt = VirtualTryOnInference(
    model_path=output_dir,
    device = "cuda",
    run_on="original",
    num_inference_steps=50,
    guidance_scale=10,
    seed = 10
)

prompt = instance_prompt
img_path = "./DebugImages/Images/img2.jpg"
res_img = vt.infer(img_path, prompt)
cv2.imwrite("./res.jpg", res_img)
