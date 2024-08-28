## Dependency Setup
Create the conda environment with below commands.
Our code is implemented based on torch + diffusers.
You should first check your cuda compiler version, and install the compatible version of pytorch.

We ran our experiment with torch 2.1.1 + cuda 12.1.

```Shell
conda create -n diffusehigh python=3.9.0 -y
conda activate diffusehigh
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 xformers --index-url https://download.pytorch.org/whl/cu121
```

Please install below pacakges in your environment.
We leave our settings below:

- diffusers==0.24.0
- accelerate
- transformers
- pywavelets
- pytorch-wavelets


## Run DiffuseHigh!
You can easily import the DiffuseHighSDXLPipeline from our provided code below.

example code for generating 4K image.
```Python
from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline
pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
prompt = "a photo of an astronaut riding a horse on mars."

image = model(
        prompt,
        negative_prompt=negative_prompt,
        target_height=[2048, 3072, 4096],
        target_width=[2048, 3072, 4096],
        enable_dwt=True,
        dwt_steps=5,
        enable_sharpening=True,
        sharpness_factor=1.0,
    ).images[0]

image.save("sample.png")
```