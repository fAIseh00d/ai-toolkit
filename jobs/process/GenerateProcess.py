import gc
import os
from collections import OrderedDict
from typing import ForwardRef, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_model_hash_to_meta, \
    add_base_model_info_to_meta
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from extensions.VitonHDDatagen import VitonHDTestDataset
from einops import rearrange
from diffusers import AutoencoderKL
import random


class GenerateConfig:

    def __init__(self, **kwargs):
        self.prompts: List[str]
        self.sampler = kwargs.get('sampler', 'ddpm')
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)
        self.size_list: Union[List[int], None] = kwargs.get('size_list', None)
        self.neg = kwargs.get('neg', '')
        self.seed = kwargs.get('seed', -1)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.prompt_2 = kwargs.get('prompt_2', None)
        self.neg_2 = kwargs.get('neg_2', None)
        self.prompts = kwargs.get('prompts', None)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.compile = kwargs.get('compile', False)
        self.ext = kwargs.get('ext', 'png')
        self.prompt_file = kwargs.get('prompt_file', False)
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.prompts_in_file = self.prompts
        self.inpaint_data_path = kwargs.get('inpaint_data_path', None)

        if self.prompts is None:
            raise ValueError("Prompts must be set")
        if isinstance(self.prompts, str) and self.inpaint_data_path is None:
            if os.path.exists(self.prompts):
                with open(self.prompts, 'r', encoding='utf-8') as f:
                    self.prompts_in_file = f.read().splitlines()
                    self.prompts_in_file = [p.strip() for p in self.prompts_in_file if len(p.strip()) > 0]
            else:
                raise ValueError("Prompts file does not exist, put in list if you want to use a list of prompts")

        self.random_prompts = kwargs.get('random_prompts', False)
        self.max_random_per_prompt = kwargs.get('max_random_per_prompt', 1)
        self.max_images = kwargs.get('max_images', 10000)

        if self.random_prompts:
            self.prompts = []
            for i in range(self.max_images):
                num_prompts = random.randint(1, self.max_random_per_prompt)
                prompt_list = [random.choice(self.prompts_in_file) for _ in range(num_prompts)]
                self.prompts.append(", ".join(prompt_list))
        else:
            self.prompts = self.prompts_in_file

        if kwargs.get('shuffle', False):
            # shuffle the prompts
            random.shuffle(self.prompts)


class GenerateProcess(BaseProcess):
    process_id: int
    config: OrderedDict
    progress_bar: ForwardRef('tqdm') = None
    sd: StableDiffusion

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.model_config = ModelConfig(**self.get_conf('model', required=True))
        self.device = self.get_conf('device', self.job.device)
        self.generate_config = GenerateConfig(**self.get_conf('generate', required=True))
        self.torch_dtype = get_torch_dtype(self.get_conf('dtype', 'float16'))

        self.progress_bar = None
        self.sd = StableDiffusion(
            device=self.device,
            model_config=self.model_config,
            dtype=self.model_config.dtype,
        )

        print(f"Using device {self.device}")

    def clean_prompt(self, prompt: str):
        # remove any non alpha numeric characters or ,'" from prompt
        return ''.join(e for e in prompt if e.isalnum() or e in ", '\"")

    def run(self):
        with torch.no_grad():
            super().run()
            print("Loading model...")
            self.sd.load_model()
            self.sd.pipeline.to(self.device, self.torch_dtype)

            print("Compiling model...")
            # self.sd.unet = torch.compile(self.sd.unet, mode="reduce-overhead", fullgraph=True)
            if self.generate_config.compile:
                self.sd.unet = torch.compile(self.sd.unet, mode="reduce-overhead")

            # build prompt image configs
            prompt_image_configs = []
            if self.generate_config.inpaint_data_path is None:
                print(f"Generating {len(self.generate_config.prompts)} images")
                for _ in range(self.generate_config.num_repeats):
                    for i in range(len(self.generate_config.prompts)):
                        prompt = self.generate_config.prompts[i]
                        width = self.generate_config.width
                        height = self.generate_config.height
                        # prompt = self.clean_prompt(prompt)

                        if self.generate_config.size_list is not None:
                            # randomly select a size
                            width, height = random.choice(self.generate_config.size_list)

                        prompt_image_configs.append(GenerateImageConfig(
                            prompt=prompt,
                            prompt_2=self.generate_config.prompt_2,
                            width=width,
                            height=height,
                            num_inference_steps=self.generate_config.sample_steps,
                            guidance_scale=self.generate_config.guidance_scale,
                            negative_prompt=self.generate_config.neg,
                            negative_prompt_2=self.generate_config.neg_2,
                            seed=self.generate_config.seed,
                            guidance_rescale=self.generate_config.guidance_rescale,
                            output_ext=self.generate_config.ext,
                            output_folder=self.output_folder,
                            add_prompt_file=self.generate_config.prompt_file
                        ))
            else:
                print("Expecting VitonHD Dataset...")
                dataset = VitonHDTestDataset(self.generate_config.inpaint_data_path,
                                            phase='test', order='unpaired',
                                            size=(self.generate_config.height, self.generate_config.width // 2),
                                            data_list="test_pairs.txt")
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

                print(f"Generating {len(dataset)} images")
                for data in dataloader:
                    prompt = self.generate_config.prompts[0]
                    latents = None
                    masked_latents = None
                    if self.model_config.is_flux_fill:
                        latents, masked_latents = prepare_fill_latents(
                            data["image"],
                            data["im_mask"],
                            data["inpaint_mask"],
                            self.sd.vae,
                            self.device,
                            self.torch_dtype
                        )
                    width = self.generate_config.width
                    height = self.generate_config.height
                    # prompt = self.clean_prompt(prompt)

                    if self.generate_config.size_list is not None:
                        # randomly select a size
                        width, height = random.choice(self.generate_config.size_list)

                    prompt_image_configs.append(GenerateImageConfig(
                        prompt=prompt,
                        prompt_2=self.generate_config.prompt_2,
                        width=width,
                        height=height,
                        num_inference_steps=self.generate_config.sample_steps,
                        guidance_scale=self.generate_config.guidance_scale,
                        negative_prompt=self.generate_config.neg,
                        negative_prompt_2=self.generate_config.neg_2,
                        seed=self.generate_config.seed,
                        guidance_rescale=self.generate_config.guidance_rescale,
                        output_ext=self.generate_config.ext,
                        output_folder=self.output_folder,
                        latents=latents,
                        masked_image_latents=masked_latents,
                        image = data['image'],
                        mask_image = data['inpaint_mask'],
                        add_prompt_file=self.generate_config.prompt_file
                    ))
            # generate images
            self.sd.generate_images(prompt_image_configs, sampler=self.generate_config.sampler)

            print("Done generating images")
            # cleanup
            del self.sd
            gc.collect()
            torch.cuda.empty_cache()

def prepare_fill_latents(
        image_input: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        vae: AutoencoderKL,
        device,
        dtype: str
    ):
    """
    Prepares image and mask for fill operation with proper rearrangement.
    Focuses only on image and mask processing.
    """
    image_input = image_input.to(device=device, dtype=dtype)
    image = image.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # Apply mask to image
    masked_image = image.clone()
    masked_image = masked_image * (1 - mask)

    # Encode to latents
    input_latents = vae.encode(image_input.to(vae.dtype)).latent_dist.sample()
    input_latents = (
        input_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    input_latents = input_latents.to(dtype)

    image_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # Process mask following the example's specific rearrangement
    mask = mask[:, 0, :, :] if mask.shape[1] > 1 else mask[:, 0, :, :]
    mask = mask.to(torch.bfloat16)
    
    # First rearrangement: 8x8 patches
    mask = rearrange(
        mask,
        "b (h ph) (w pw) -> b (ph pw) h w",
        ph=8,
        pw=8,
    )
    
    # Second rearrangement: 2x2 patches
    mask = rearrange(
        mask, 
        "b c (h ph) (w pw) -> b (h w) (c ph pw)", 
        ph=2, 
        pw=2
    )

    # Rearrange image latents similarly
    input_latents = rearrange(
        input_latents,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2
    )

    image_latents = rearrange(
        image_latents,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2
    )

    # Combine images and mask
    cond_latent = torch.cat([image_latents, mask], dim=-1)

    return input_latents, cond_latent