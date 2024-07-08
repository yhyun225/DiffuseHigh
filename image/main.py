import os
import argparse

import torch
from omegaconf import OmegaConf

from utils.logger_utils import setup_logger
from utils.utils import set_seeds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Fix seed.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="log", help="Directory for the generated images to be saved.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/sdxl_4096x4096.yaml", help="Path of the configuration file.",
    )
    parser.add_argument(
        "--use_amp", type=bool, default=True, help="Use mixed precision.",
    )
    parser.add_argument(
        "--logger_level", type=str, default="info", choices=["debug", "info", "warn", "error", "fatal"],help="level of the logger.",
    )
    parser.add_argument(
        "--prompt", type=str, default="a baby bunny sitting on a stack of pancakes", help="Text prompt for image generation.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    set_seeds(args.seed)
    setup_logger(args)
    
    if "sd2.1" in config.version:
        from models.diffusion import DiffuseHigh
        pipeline = DiffuseHigh(config, use_amp=args.use_amp)
    elif "sdxl" in config.version:
        from models.diffusion_xl import DiffuseHighXL
        pipeline = DiffuseHighXL(config, use_amp=args.use_amp)

    os.makedirs(os.path.join(args.log_dir), exist_ok=True)

    image = pipeline.sample_HR(
        prompt=args.prompt
    )[0]

    image.save(os.path.join(args.log_dir, "sample.png"))

if __name__ == "__main__":
    main()
