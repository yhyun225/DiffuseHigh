import os
import cv2
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.utils import export_to_video

from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.image_processor import VaeImageProcessor

from tqdm import tqdm

from pytorch_wavelets import DWTForward, DWTInverse

logger = logging.getLogger()

def _append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs

class TextToVideoDiffusion(object):
    def __init__(self, config):
        
        self.version = config.version
        self.cfg = config.diffusion
        self.dwt_cfg = config.dwt
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_dtype = torch.float16
        
        self.configure(config)
        
    
    def configure(self, config):
        logger.info("Loading Stable Diffusion...")
        logger.info(f"__version__: {self.version}")
        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": "./cache",
        }
        
        pipeline = TextToVideoSDPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs
        ).to(self.device)
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()
        
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        self.pipeline = pipeline
        self.device = self.pipeline._execution_device
        logger.info("Loaded Stable Video Diffusion!")
        
        for p in self.pipeline.vae.parameters():
             p.requires_grad_(False)
        for p in self.pipeline.unet.parameters():
            p.requires_grad_(False)
            
        self.dwt = DWTForward(
            J=self.dwt_cfg.level, wave=self.dwt_cfg.wave, mode=self.dwt_cfg.mode
        ).to(self.device)

        self.idwt = DWTInverse(
            wave=self.dwt_cfg.wave, mode=self.dwt_cfg.mode
        ).to(self.device)
            
    def interpolate_frame_latent(
        self,
        latents,
        tar_height,
        tar_width,
        interpolation="bilinear"
    ):
            
        frames = torch.tensor([]).to(self.device)
        with torch.no_grad():
            frames = self.pipeline.decode_latents(latents)
           
        batch, channel, frame_nums, height, width = frames.shape
        frames = frames.permute(0, 2, 1, 3, 4).reshape(batch * frame_nums, channel, height, width)

    
        assert interpolation in ["bilinear", "bicubic"]
        HR_latents = torch.tensor([]).to(self.device, dtype=torch.float16)
        
        HR_frames = F.interpolate(
            frames, (tar_height, tar_width), mode=interpolation, align_corners=False
            ).to(self.device, dtype=torch.float16)
        
        with torch.no_grad():
            LL, _ = self.dwt(HR_frames.to(torch.float32))
            
            HR_latents = self.pipeline.vae.encode(HR_frames).latent_dist.mode()
            
        HR_latents = HR_latents.unsqueeze(0).permute(0, 2, 1, 3, 4).to(dtype=torch.float16) * self.pipeline.vae.config.scaling_factor

        return LL, HR_latents 
    
    def denoising_diffusion_process(
        self, 
        latents,
        timesteps,
        prompt_embeds,
        ):
        
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                    latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,    
                        return_dict=False,
                    )[0]

                    if self.cfg.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                    latents = self.pipeline.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample

                    latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

        return latents
    
    def denoising_diffusion_process_with_dwt(
        self, 
        latents,
        LL,
        timesteps,
        prompt_embeds,
        dwt_steps=5
        ):
        
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                    latent_model_input = torch.cat([latents] * 2) if self.cfg.do_classifier_free_guidance else latents
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,   
                        return_dict=False,
                    )[0]

                    if self.cfg.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                    if i < dwt_steps:
                        pred_clean_latents = self.pipeline.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).pred_original_sample
                        pred_clean_frames = self.pipeline.decode_latents(pred_clean_latents.unsqueeze(0).permute(0, 2, 1, 3, 4))
                        
                        bsz, channel, frames, width, height = pred_clean_frames.shape
                        pred_clean_frames = pred_clean_frames.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                        _, HH = self.dwt(pred_clean_frames)
                        coeffs = (LL, HH)
                        
                        pred_clean_frames = self.idwt(coeffs)
                        pred_clean_latents = self.pipeline.vae.encode(pred_clean_frames.to(dtype=torch.float16)).latent_dist.mode()
                        pred_clean_latents = pred_clean_latents.unsqueeze(0).permute(0, 2, 1, 3, 4).to(dtype=torch.float16) * self.pipeline.vae.config.scaling_factor

                        noise = torch.randn_like(pred_clean_latents)
                        prev_t = t - self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
                        latents = self.pipeline.scheduler.add_noise(pred_clean_latents, noise, prev_t)
                    
                    else:    
                        latents = self.pipeline.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample
                        latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

        return latents
    
    @torch.no_grad()
    def Sample_HR_Video_progress(
        self, 
        prompt,
        negative_prompt="",
        i=0
        ):
        target_height, target_width = self.cfg.target_height, self.cfg.target_width
        denoise_start, denoise_end = self.cfg.denoise_start, self.cfg.denoise_end
        seed = self.cfg.seed + i
        self.generator = torch.manual_seed(seed)
        
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            self.device,
            self.cfg.num_images_per_prompt,
            self.cfg.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if self.cfg.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        self.pipeline.scheduler.set_timesteps(self.cfg.num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        num_channels_latents = self.pipeline.unet.config.in_channels
        
        for i, tar_size in enumerate(zip(target_height, target_width)):
            tar_h, tar_w = tar_size
            start, end = denoise_start[i], denoise_end[i]
            partial_timesteps = timesteps[start:end]
            
            if i != 0:
                LL, latents = self.interpolate_frame_latent(latents, tar_h, tar_w, "bilinear") 
                noise = torch.randn_like(latents)
                t = timesteps[start]
                latents = self.pipeline.scheduler.add_noise(latents, noise, t)
                
            else:
                latents = None
                latents = self.pipeline.prepare_latents(
                    batch_size * self.cfg.num_images_per_prompt,
                    num_channels_latents,
                    self.cfg.num_frames,
                    tar_h,
                    tar_w,
                    prompt_embeds.dtype,
                    self.device,
                    generator=self.generator,
                    latents=latents,
                )
            
            self.extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(self.generator, self.cfg.eta)
            
            if i != 0:
                latents = self.denoising_diffusion_process_with_dwt(latents=latents, LL=LL, timesteps=partial_timesteps, prompt_embeds=prompt_embeds)
            
            else:
                latents = self.denoising_diffusion_process(latents=latents, timesteps=partial_timesteps, prompt_embeds=prompt_embeds)
                
            if self.cfg.save_video:
                with torch.no_grad():
                    video_tensor = self.pipeline.decode_latents(latents)
                video = tensor2vid(video_tensor, self.pipeline.image_processor, output_type="np")
                export_to_video(video[0], 
                                output_video_path=os.path.join(self.cfg.save_path, f'{prompt}_seed_{seed}_{tar_h}_startstep_{start}_dwt.mp4'), 
                                fps=7)
                return
        
        with torch.no_grad():    
            video_tensor = self.pipeline.decode_latents(latents)
        video = tensor2vid(video_tensor, self.pipeline.image_processor, "np")    
        
        return video