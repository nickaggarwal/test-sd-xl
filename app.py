import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from io import BytesIO
import base64
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
      base = "stabilityai/stable-diffusion-xl-base-1.0"
      repo = "SG161222/RealVisXL_V4.0_Lightning"
      ckpt = "RealVisXL_V4.0_Lightning.safetensors" # Use the correct ckpt for your step setting!

      # Load model.
      self.pipe = DiffusionPipeline.from_pretrained( repo, torch_dtype=torch.float16,variant="fp16",use_safetensors = True).to("cuda")
      # Ensure sampler uses "trailing" timesteps.
      self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    def infer(self, inputs):
      prompt = inputs["prompt"]

      image_output = self.pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
      buff = BytesIO()
      image_output.save(buff, format="JPEG")
      img_str = base64.b64encode(buff.getvalue()).decode()
      return { "generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None 
