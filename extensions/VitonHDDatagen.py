# coding=utf-8
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor


debug_mode=False

def tensor_to_image(tensor, image_path):
    """
    Convert a torch tensor to an image file.

    Args:
    - tensor (torch.Tensor): the input tensor. Shape (C, H, W).
    - image_path (str): path where the image should be saved.

    Returns:
    - None
    """
    if debug_mode: 
        # Check the tensor dimensions. If it's a batch, take the first image
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Check for possible normalization and bring the tensor to 0-1 range if necessary
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Convert tensor to PIL Image
        to_pil = ToPILImage()
        img = to_pil(tensor)

        # Save the PIL Image
        dir_path = os.path.dirname(image_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img.save(image_path)


def sep_tags(txt: str):
    lines = txt.replace('\n\n', '\n').split('\n')
    data = {}
    tag = ''
    for l in lines:
        sep_idx = l.find(':')
        if sep_idx < 0:
            data[tag] = '\n'.join([data.get(tag, ''), l])
        if ' ' not in l[:sep_idx]:
            tag = l[:sep_idx]
            data[tag] = l[sep_idx+2:]
        else:
            data[tag] = '\n'.join([data.get(tag, ''), l])
    return data


PROMPT_CAMS = [
    "Canon EOS R5, 85mm lens at f/1.4, f/5.6, sharp focus, minimal distortion.",
    "Nikon Z7 II, 50mm lens at f/1.8, f/8, sharp focus, accurate color representation.",
    "Fujifilm GFX 100S, 110mm lens at f/2, f/5.6, exceptional detail, minimal distortion.",
    "Canon EOS 5D Mark IV, 100mm macro lens at f/2.8, f/8, sharp focus, true-to-life colors.",
    "Sony Alpha 7R III, 70-200mm lens at f/2.8, f/5.6, sharp focus, minimal distortion.",
    "Nikon D850, 105mm macro lens at f/2.8, f/8, exceptional detail, accurate color representation.",
    "Canon EOS R6, 24-70mm lens at f/2.8, f/5.6, sharp focus, minimal distortion.",
    "Sony A9 II, 135mm lens at f/1.8, f/5.6, sharp focus, true-to-life colors.",
    "Nikon Z6, 85mm lens at f/1.8, f/5.6, exceptional detail, minimal distortion.",
    "Canon EOS RP, 35mm macro lens at f/1.8, f/8, sharp focus, accurate color representation.",
    "Sony A7R IV, 50mm lens at f/5.6, sharp focus, minimal distortion."
]
PROMPT_PERSON_ORDER = ['Person', 'Pose', 'Outfit', 'Top', 'Bottom', 'Accessories']
PROMPT_START = """\
This side-by-side pair of photographic images highlights a clothing and its styling on a model;
[IMAGE1] Left side: A product photograph of a clothing item alone in a flat-lay fashion, displayed against a clean, solid-color background, only garment;
[IMAGE2] Right Side: The exact same clothing is worn by a model in a lifestyle setting."""
PROMPT_END = "A high-resolution image, professional photography, high contrast, editorial quality."
PROMPT_TOP = "Top: The person is wearing the same clothing as featured on the left image."


def gen_prompt(cloth_txt, human_txt, prompt_cam):
    # Fix for incorrect description
    prompt_cloth = cloth_txt.replace("The image depicts", "The left image depicts")
    human_data = sep_tags(human_txt)
    prompt_human = []
    for tag in PROMPT_PERSON_ORDER:
        if tag == 'Top':
            # Replace actual clothing with replacing sentence
            prompt_human.append(PROMPT_TOP)
            continue
        prompt_human.append(tag + ': ' + human_data[tag])
    prompt_human = '\n'.join(prompt_human)

    result = PROMPT_START + '\n\n'
    result += prompt_cloth + '\n\n'
    result += prompt_human + '\n\n'
    result += PROMPT_END + '\n'
    result += prompt_cam

    return result


class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        data_list: Optional[str] = None,
    ):
        super(VitonHDTestDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        # This code defines a transformation pipeline for image processing
        self.transform = transforms.Compose(
            [
                # Convert the input image to a PyTorch tensor
                transforms.ToTensor(),
                # Normalize the tensor values to a range of [-1, 1]
                # The first [0.5] is the mean, and the second [0.5] is the standard deviation
                # This normalization is applied to each color channel
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        filename = os.path.join(dataroot_path, data_list)

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

        with open(os.path.join(self.dataroot, self.phase, "clothes.json"), 'rt', encoding='utf-8') as fp:
            self.cloth_desc = json.load(fp)
        with open(os.path.join(self.dataroot, self.phase, "images.json"), 'rt', encoding='utf-8') as fp:
            self.human_desc = json.load(fp)

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).resize((self.width,self.height))
        cloth_pure = self.transform(cloth)
        cloth_mask = Image.open(os.path.join(self.dataroot, self.phase, "cloth-mask", c_name)).resize((self.width,self.height))
        cloth_mask = self.transform(cloth_mask)
        
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg','_mask.png'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        mask = 1-mask
        im_mask = image * mask
 
        pose_img = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        ).resize((self.width,self.height))
        pose_img = self.transform(pose_img)  # [-1,1]
 
        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["cloth_pure"] = cloth_pure
        result["cloth_mask"] = cloth_mask
        
        # Concatenate image and garment along width dimension
        inpaint_image = torch.cat([cloth_pure, im_mask], dim=2)  # dim=2 is width dimension
        result["im_mask"] = inpaint_image
        
        GT_image = torch.cat([cloth_pure, image], dim=2)  # dim=2 is width dimension
        result["image"] = GT_image
        
        # Create extended black mask for garment portion
        garment_mask = torch.zeros_like(1-mask)  # Create mask of same size as original
        extended_mask = torch.cat([garment_mask, 1-mask], dim=2)  # Concatenate masks
        result["inpaint_mask"] = extended_mask

        result["prompt"] = gen_prompt(self.cloth_desc[c_name],
                                      self.human_desc[im_name],
                                      PROMPT_CAMS[index % len(PROMPT_CAMS)])

        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)


if __name__ == "__main__":
    dataset = VitonHDTestDataset("/mnt/data/Development/Work/VTON/Datasets/TEST-HD/", phase="test", order="paired", data_list="test_pairs.txt")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for data in loader:
        print(type(data))
        pass