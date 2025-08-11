import torch
from transformers import pipeline
'''
see:
https://huggingface.co/docs/transformers/en/model_doc/depth_anything?usage=Pipeline
'''
pass

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", torch_dtype=torch.bfloat16, device=0)
pipe("http://images.cocodataset.org/val2017/000000039769.jpg")["depth"]


from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

import numpy as np

from PIL import Image

import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")

model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")




inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)