import argparse
from models.DiffuseHighVideo import TextToVideoDiffusion
from omegaconf import OmegaConf

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_config.yaml")
    parser.add_argument("--prompt", type=str, default="A teddy bear is dancing in front of the building", help="Text prompt for video generation",)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    config = OmegaConf.load(args.config)
    prompt = args.prompt
    
    model = TextToVideoDiffusion(config)
    model.Sample_HR_Video_progress(prompt=prompt)[0]


if __name__ == "__main__":
    main()