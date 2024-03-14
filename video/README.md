## DiffuseHigh video generation

# Environment
### Python3.10   
### Packages:
* torch==2.2.1   
* diffuesrs==0.26.3

# Environment Setting
```Shell
pip install -r requirements.txt
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

You should also install pytorch_wavelet on the current directory
```Shell
git clone https://github.com/fbcotter/pytorch_wavelets.git
cd pytorch_wavelets/
pip install .
```

# Inference
```Shell
python main.py --config configs/config.yaml \
--prompt "A polar bear is walking on the glacier"
```
