import cv2
from VirtualTryOn import VirtualTryOnInference

vt = VirtualTryOnInference(
    model_path="/data/Kaggle/StableDiff/ResultsJacket3_2/",
    device = "cuda",
    run_on="original",
    num_inference_steps=50,
    guidance_scale=10,
    seed = 10
)

prompt = "Brown Solid colour ska jacket"
img_path = "./DebugImages/Images/img2.jpg"
res_img = vt.infer(img_path, prompt)
cv2.imwrite("./res.jpg", res_img)
