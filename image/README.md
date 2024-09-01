## DiffuseHigh image generation

# Environment Setting
### Python3.9   
### Packages:
* torch==2.0.1   
* diffuesrs==0.25.0

First create the anaconda environment.
```Shell
conda create -n diffusehigh python=3.9.0 -y
conda activate diffusehigh
```

Install the required packages.
```Shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

You should also install pytorch_wavelet on the current directory
```Shell
git clone https://github.com/fbcotter/pytorch_wavelets.git
pip install pytorch_wavelets/
```

# Inference
```Shell
python main.py --config configs/sdxl_4096x4096.yaml \
--log_dir log \
--prompt "a baby bunny sitting on a stack of pancakes"
```
