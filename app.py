import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------
# Streamlit UI
# ---------------------
st.title("Stable Diffusion Image Generator (Colab)")
st.write("Generate AI art using Dreamlike Diffusion model.")

# Default prompt
prompt = st.text_input("Enter your prompt:", "Mom loves kids and flowers")

generate = st.button("Generate Image")

if generate:
    st.write("Loading model... (only first time)")

    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = pipe.to("cuda")

    st.write(f"**[PROMPT]**: {prompt}")

    # Generate image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    st.image(image, caption="Generated Image", use_column_width=True)
