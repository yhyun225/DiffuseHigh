import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers import (
    DDIMScheduler, 
    StableDiffusionPipeline
)
from pytorch_wavelets import DWTForward, DWTInverse

logger = logging.getLogger()

class DiffuseHigh(object):
    def __init__(self, config, use_amp=True):
        self.version = config.version
        self.cfg = config.diffusion
        self.dwt_cfg = config.dwt

        self.weights_dtype = torch.float16 if use_amp else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.configure()
    
    def configure(self):
        logger.info("Loading Stable Diffusion...")
        logger.info(f"__version__: {self.version}")
        
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": "./cache",
        }

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs
        ).to(self.device)

        self.pipeline.enable_vae_tiling()
        self.vae_scale_factor = self.pipeline.vae_scale_factor

        logger.info("Loaded Stable Diffusion!")

        if self.cfg.freeu:
            if self.version == "sd1.5":
                self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.4)
            elif self.version == "sd2.1":
                self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
        
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir="./cache",
        )

        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        
        self.unet.eval()
        self.vae.eval()

        self.dwt = DWTForward(
            J=self.dwt_cfg.level, wave=self.dwt_cfg.wave, mode=self.dwt_cfg.mode
        ).to(self.device)

        self.idwt = DWTInverse(
            wave=self.dwt_cfg.wave, mode=self.dwt_cfg.mode
        ).to(self.device)
    
    @property
    def unet(self): return self.pipeline.unet

    @property
    def vae(self): return self.pipeline.vae

    @property
    def image_processor(self): return self.pipeline.image_processor

    @property
    def encode_prompt(self): return self.pipeline.encode_prompt
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(self.weights_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
        cross_attention_kwargs=None,
    ):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)
    
    def pred_original_samples(
        self,
        noise_pred,
        timestep,
        noisy_latents,
    ):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        
        return pred_original_sample
    
    def obtain_clean_images(
        self,
        noisy_latents,
        text_embeddings,
    ):
        latents = noisy_latents.to(self.device)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        timesteps = self.scheduler.timesteps

        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                )

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents
    
    def denoising_with_dwt(
        self,
        latents,
        prompt_embeds,
        timesteps,
        target_height,
        target_width,
        dwt_steps=5,
    ):
        low_res_images = self.decode_latents(latents)
        interp_images = F.interpolate(
            low_res_images, (target_height, target_width), mode="bilinear", align_corners=False,
        )

        with torch.no_grad():
            LL, _ = self.dwt(interp_images.to(torch.float32))

            latents = self.encode_images(interp_images)
            noise = torch.randn_like(latents)
            t = timesteps[0]

            latents = self.scheduler.add_noise(latents, noise, t)

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds
                )

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                
                # apply DWT and switch low-frequency components
                if i < dwt_steps:
                    pred_clean_latents = self.pred_original_samples(noise_pred, t, latents)
                    pred_clean_images = self.decode_latents(pred_clean_latents)

                    ll, HH = self.dwt(pred_clean_images.to(torch.float32))
                    coeffs = (LL, HH)

                    pred_clean_images = self.idwt(coeffs)
                    pred_clean_latents = self.encode_images(pred_clean_images.to(self.weights_dtype))
                    
                    noise = torch.randn_like(pred_clean_latents)
                    prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                    latents = self.scheduler.add_noise(pred_clean_latents, noise, prev_t)
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents


    def sample_HR(
        self,
        prompt,
        negative_prompt=None,
        verbose=False,
        height=None,
        width=None,
    ):
        with torch.no_grad():
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise NotImplementedError
        
            (
                prompt_embeds, 
                negative_prompt_embeds
            ) = self.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            b, c = batch_size, self.unet.config.in_channels
            h, w = height // self.vae_scale_factor, width // self.vae_scale_factor
            noisy_latents = torch.randn((b, c, h, w), dtype=self.weights_dtype)

            latents = self.obtain_clean_images(noisy_latents, prompt_embeds)

            self.scheduler.set_timesteps(self.cfg.diffusion_steps)
            hr_timesteps = self.scheduler.timesteps[-self.cfg.noising_steps: ]

            target_height = self.cfg.target_height
            target_width = self.cfg.target_width

            assert type(target_height) == type(target_width), "type of the target height and width should be the same"
            if type(target_height) == int:
                target_height = [target_height]
                target_width = [target_width]
            
            dwt_steps = self.dwt_cfg.steps
            if type(dwt_steps) == int:
                dwt_steps = [dwt_steps] * len(target_height)

            for h, w, d in zip(target_height, target_width, dwt_steps):
                latents = self.denoising_with_dwt(
                    latents,
                    prompt_embeds,
                    hr_timesteps,
                    target_height=h,
                    target_width=w,
                    dwt_steps=d,
                )

            images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            images = self.image_processor.postprocess(images, output_type="pil")
        
        return images