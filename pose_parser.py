from mmpose.apis import MMPoseInferencer

img_path = './inputs/1.jpeg'
inferencer = MMPoseInferencer('human')
result_generator = inferencer(img_path, show=True)
result = next(result_generator)