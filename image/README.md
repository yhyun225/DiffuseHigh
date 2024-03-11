## DiffuseHigh image generation

# Environment Setting
```Shell
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

You should also install pytorch_wavelet on the current directory
```Shell
git clone https://github.com/fbcotter/pytorch_wavelets.git
pip install .
```

# Inference
```Shell
python main.py --config configs/sdxl_4096x4096.yaml \
--log_dir DiffuseHigh \
--prompt "a baby bunny sitting on a stack of pancakes"
```
