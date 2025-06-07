import os
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

token = os.getenv("HF_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=token,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

gr.Interface(
    fn=generate,
    inputs=gr.Textbox(placeholder="Describe your image..."),
    outputs="image",
    title="AI Image Generator"
).launch()
