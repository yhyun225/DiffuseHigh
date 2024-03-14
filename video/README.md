## DiffuseHigh video generation

# Environment
### Python3.10   
### Packages:
* torch==2.2.1   
* diffuesrs==0.26.3

# Environment Setting
First create the anaconda environment.
```Shell
conda create -n diffusehigh python=3.10.0 -y
conda activate diffusehigh
```

Install the required packages.
```Shell
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

You should also install pytorch_wavelet on the current directory
```Shell
git clone https://github.com/fbcotter/pytorch_wavelets.git
pip install pytorch_wavelets/
```

# Inference
```Shell
python main.py --config configs/config.yaml \
--prompt "A polar bear is walking on the glacier"
```
