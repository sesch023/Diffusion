import copy
import os
import shutil
import sys
from abc import ABC, abstractmethod

import lightning.pytorch as pl
import lightning.pytorch.callbacks as cb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
import clip
from diffusers import LDMSuperResolutionPipeline
from PIL import Image
from super_image import DrlnModel, ImageLoader
from torch import Tensor, nn, optim, utils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

import wandb
from DiffusionModules.ClipTranslatorModules import (ClipTranslator,
                                                    ClipTranslatorTrainer)
from DiffusionModules.DataModules import CIFAR10DataModule
from DiffusionModules.ModelLoading import load_udm
from einops import rearrange
import random

class ClipTools(nn.Module):
    def __init__(self, clip_model="ViT-B/32", device=None):
        super().__init__()
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._clip_model, self._clip_preprocess = clip.load(clip_model, device=self._device)
        self._clip_model.eval()
    
    def get_clip_emb_size(self):
        return self._clip_model.visual.output_dim 
    
    def get_clip_emb_images(self, images):
        images = torch.stack([self._clip_preprocess(i) for i in images])
        return self._clip_model.encode_image(images.to(self._device)).float()

    def get_clip_emb_videos(self, videos):
        b, t = videos.shape[0], videos.shape[1]
        stacked_frames = rearrange(videos, 'b t c h w -> (b t) c h w')
        videos = torch.stack([self._clip_preprocess(i) for i in stacked_frames])
        embs = self._clip_model.encode_image(videos.to(self._device)).float()
        embs = rearrange(embs, '(b t) e -> b t e', b=b, t=t)
        return embs.mean(dim=1)
    
    def get_clip_emb_text(self, texts):
        # TODO: Truncate hack is stupid for long sentences
        return self._clip_model.encode_text(clip.tokenize(texts, truncate = True).to(self._device)).float()
        
    def forward(self, images, texts):
        return self.get_clip_emb_images(images), self.get_clip_emb_text(texts)

class BaseEmbeddingProvider(ABC, nn.Module):
    @abstractmethod
    def get_embedding(self, images, labels):
        pass

    def forward(self, images, labels):
        return self.get_embedding(images, labels)

class ClipRandomImageTextTranslatorEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, translator_model_path, clip_tools=None, ):
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        self.clip_translator_provider = ClipTranslatorEmbeddingProvider(translator_model_path, self.clip_tools)
        self.clip_image_provider = ClipEmbeddingProvider(self.clip_tools)
        self.clip_text_provider = ClipTextEmbeddingProvider(self.clip_tools)
        self.providers = [self.clip_translator_provider, self.clip_image_provider, self.clip_text_provider]

    def get_embedding(self, images, labels):
        return random.choice(self.providers).get_embedding(images, labels)

class ClipTranslatorEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, translator_model_path, clip_tools=None):
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        self.translator_model_path = translator_model_path
        self.model = ClipTranslatorTrainer.load_from_checkpoint(self.translator_model_path, map_location=self.clip_tools._device).model
        self.model.eval()
        self.model.to(self.clip_tools._device)

    def get_embedding(self, images, labels):
        cap_emb = self.clip_tools.get_clip_emb_text(labels)
        i_embs = self.model(cap_emb)
        return i_embs
    
class ClipEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, clip_tools=None):
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        
    def get_embedding(self, images, labels):
        # In the New version this only uses images
        i_embs = self.clip_tools.get_clip_emb_images(images)    
        return i_embs
    

class ClipTextEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, clip_tools=None):
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools

    def get_embedding(self, videos, labels):
        v_embs = self.clip_tools.get_clip_emb_text(labels)    
        return v_embs

    
class CF10EmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, classes=None):
        super().__init__()
        self.classes = classes if classes is not None else CIFAR10DataModule.classes
        self.num_classes = len(self.classes)
        
    def get_embedding(self, images, labels):
        labels = [self.classes.index(label) for label in labels]
        labels = torch.Tensor(labels).long()
        return nn.functional.one_hot(labels, self.num_classes).float()