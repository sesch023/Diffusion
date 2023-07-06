from Diffusion import *
from ClipTranslatorModules import ClipTranslator, ClipTranslatorTrainer
import os
import shutil
import torch
from torch import optim, nn, utils, Tensor
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import numpy as np
import wandb
import copy
from abc import ABC, abstractmethod
from super_image import DrlnModel, ImageLoader
from diffusers import LDMSuperResolutionPipeline


class DiffusionImageTransformer():
    def __init__(self, transform_images=None, reverse_transform_images=None, img_target_size=64):
        if transform_images is None:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
                transforms.Resize(img_target_size),
                transforms.CenterCrop(img_target_size),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transform_images = transform_images
            
        if reverse_transform_images is None:
            self.reverse_transform_image = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                transforms.Lambda(lambda t: t * 255.),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                transforms.ToPILImage()
            ])
        else:
            self.reverse_transform_image = reverse_transform_images
            
    def transform_images(self, images):
        return torch.stack([self.transform_image(image) for image in images])
    
    def reverse_transform_images(self, tensors):
        return [self.reverse_transform_image(tensor) for tensor in tensors]
    
    
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, images, labels):
        pass
    
class ClipTranslatorEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, translator_model_path, clip_tools=None, device=None):
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.clip_tools = ClipTools(device=self.dev) if clip_tools is None else clip_tools
        self.translator_model_path = translator_model_path
        self.model = ClipTranslatorTrainer.load_from_checkpoint(self.translator_model_path).model
        self.model = self.model.to(self.dev)
        self.model.eval()

    def get_embedding(self, images, labels):
        cap_emb = self.clip_tools.get_clip_emb_text(labels).to(self.dev) 
        i_embs = self.model(cap_emb)
        return i_embs
    

class ClipEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, clip_tools=None, device=None):
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.clip_tools = ClipTools(device=self.dev) if clip_tools is None else clip_tools
        
    def get_embedding(self, images, labels):
        # In the New version this only uses images
        i_embs = self.clip_tools.get_clip_emb_images(images)    
        return i_embs
    
    
class CF10EmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, classes):
        super().__init__()
        self.num_classes = len(classes)
        self.classes = classes
        
    def get_embedding(self, images, labels):
        labels = [self.classes.index(label) for label in labels]
        labels = torch.Tensor(labels).long()
        return nn.functional.one_hot(labels, self.num_classes).float()

class UpscalerMode(Enum):
    NONE="NONE",
    DRLN="DRLN",
    LDM="LDM"
    
class DiffusionTrainer(pl.LightningModule):
    def __init__(self, unet, device=None, loss=None, val_score=None, diffusion_tools=None, embedding_provider=None, alt_validation_emb_provider=None, ema_beta=0.9999, diffusion_image_transformer=None, cfg_train_ratio=0.1, cfg_scale=3, captions_preprocess=None, optimizer=None, sample_upscaler_mode=UpscalerMode.LDM, sample_scale_factor=4, checkpoint_every_val_epochs=10, no_up_samples_out=True, sample_images_out_base_path="samples/"):
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.unet = unet.to(self.dev)
        self.diffusion_tools = DiffusionTools(device=self.dev, steps=1000, noise_scheduler=LinearScheduler()) if diffusion_tools is None else diffusion_tools  
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider(device=self.dev) if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.diffusion_image_transformer = DiffusionImageTransformer() if diffusion_image_transformer is None else diffusion_image_transformer
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-4, weight_decay=0.0) if optimizer is None else optimizer
        self.sample_scale_factor = sample_scale_factor
        self.sample_upscaler_mode = sample_upscaler_mode
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        self.no_up_samples_out = no_up_samples_out
        
        self.upscaler = None
        if self.sample_upscaler_mode is not None and self.sample_upscaler_mode != UpscalerMode.NONE:
            if self.sample_upscaler_mode == UpscalerMode.DRLN:
                self.up_model_pipeline = DrlnModel.from_pretrained('eugenesiow/drln', scale=sample_scale_factor).to(self.dev)
                self.upscaler = lambda image: self.up_model_pipeline(image).detach().cpu()      
                self.save_images = lambda image, path: ImageLoader.save_image(image, path)
            elif self.sample_upscaler_mode == UpscalerMode.LDM:
                model_id = "CompVis/ldm-super-resolution-4x-openimages"
                self.up_model_pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
                self.up_model_pipeline = self.up_model_pipeline.to(self.dev)
                self.upscaler = lambda image: self.up_model_pipeline(image, num_inference_steps=100, eta=1).images[0]
                self.save_images = lambda image, path: image.save(path)
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
            
        if val_score is None:
            fid = FrechetInceptionDistance(feature=64).to(self.dev)
            
            def get_fid_score(samples, real):
                fid.update((((samples + 1)/2)*255).byte(), real=False)
                fid.update((((real + 1)/2)*255).byte(), real=True)
                return fid.compute()
            
            clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.dev).eval()
            
            self.val_score = lambda samples, real, captions: {
                "clip_score": clip_model((((samples + 1)/2)*255).int(), captions),
                "fid_score": get_fid_score(samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=['unet'])
        
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.dev)
        else:
            i_embs = None
        images = self.diffusion_image_transformer.transform_images(images).to(self.dev)      
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        images = self.diffusion_image_transformer.transform_images(images).to(self.dev)        
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.dev)
        normal_sampled_images = self.diffusion_tools.sample_images(self.unet, images.shape[0], i_embs, self.cfg_scale)
        ema_samples_images = self.diffusion_tools.sample_images(self.ema_unet, images.shape[0], i_embs, self.cfg_scale)
        
        self.save_sampled_images(normal_sampled_images, captions, batch_idx, "normal")
        self.save_sampled_images(ema_samples_images, captions, batch_idx, "ema")
        if self.upscaler is not None and self.no_up_samples_out:
            self.save_sampled_images(normal_sampled_images, captions, batch_idx, "normal_no_up", no_upscale=True)
            self.save_sampled_images(ema_samples_images, captions, batch_idx, "ema_no_up", no_upscale=True)
            
        try:
            val_score = self.val_score(ema_samples_images, images, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        avg_dict = dict()
        outs = self.validation_step_outputs
        
        for key in outs[0].keys():
            values = [outs[i][key] for i in range(len(outs)) if key in outs[i]]
            avg = sum(values) / len(values)
            avg_dict[key] = avg
       
        self.log_dict(avg_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        if self.val_epoch % self.checkpoint_every_val_epochs == 0:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            
        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None, no_upscale=False):
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.diffusion_image_transformer.reverse_transform_images(sampled_images.detach().cpu())
            
        if self.upscaler is not None and not no_upscale:
            sampled_images = [ImageLoader.load_image(image) for image in sampled_images]
            sampled_images = [self.upscaler(image.to(self.dev)) for image in sampled_images]
            
        for image_id in range(len(sampled_images)):
            self.save_images(sampled_images[image_id], path_folder + f"img_{image_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=200, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss"
            }
        }
    
    
class UpscalerDiffusionTrainer(pl.LightningModule):
    def __init__(self, unet, start_size=64, target_size=256, device=None, loss=None, val_score=None, diffusion_tools=None, embedding_provider=None, alt_validation_emb_provider=None, ema_beta=0.9999, diffusion_image_transformer=None, cfg_train_ratio=0.1, cfg_scale=3, captions_preprocess=None, optimizer=None, checkpoint_every_val_epochs=10, sample_images_out_base_path="samples_upscale/"):#
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.unet = unet.to(self.dev)
        self.diffusion_tools = DiffusionTools(device=self.dev, steps=1000, noise_scheduler=LinearScheduler()) if diffusion_tools is None else diffusion_tools  
        self.start_size = start_size
        self.target_size = target_size
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider(device=self.dev) if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.diffusion_image_transformer = DiffusionImageTransformer(img_target_size=target_size) if diffusion_image_transformer is None else diffusion_image_transformer
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-4, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        
        self.transform_low_res = transforms.Compose([
            transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            transforms.Resize(start_size),
            transforms.CenterCrop(start_size),
            transforms.Resize(target_size),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
            
        if val_score is None:
            fid = FrechetInceptionDistance(feature=64).to(self.dev)
            
            def get_fid_score(samples, real):
                fid.update((((samples + 1)/2)*255).byte(), real=False)
                fid.update((((real + 1)/2)*255).byte(), real=True)
                return fid.compute()
            
            self.val_score = lambda samples, real, captions: {
                "fid_score": get_fid_score(samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=['unet'])
        
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.dev)
        else:
            i_embs = None
        
        low_res = torch.stack([self.transform_low_res(image).to(self.dev) for image in images]).to(self.dev)  
        images = self.diffusion_image_transformer.transform_images(images).to(self.dev)   
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs, x_unnoised_appendex=low_res)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        low_res = torch.stack([self.transform_low_res(image).to(self.dev) for image in images])
        images = self.diffusion_image_transformer.transform_images(images).to(self.dev)   
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.dev)
        normal_sampled_images = self.diffusion_tools.sample_images(self.unet, images.shape[0], i_embs, self.cfg_scale, x_appendex=low_res)
        ema_samples_images = self.diffusion_tools.sample_images(self.ema_unet, images.shape[0], i_embs, self.cfg_scale, x_appendex=low_res)
        self.save_sampled_images(normal_sampled_images, captions, batch_idx, "normal")
        self.save_sampled_images(ema_samples_images, captions, batch_idx, "ema")
        self.save_sampled_images(low_res, captions, batch_idx, "low_res")
        try:
            val_score = self.val_score(ema_samples_images, images, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        avg_dict = dict()
        outs = self.validation_step_outputs
        
        for key in outs[0].keys():
            values = [outs[i][key] for i in range(len(outs)) if key in outs[i]]
            avg = sum(values) / len(values)
            avg_dict[key] = avg
       
        self.log_dict(avg_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        if self.val_epoch % self.checkpoint_every_val_epochs == 0:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            
        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None):
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.diffusion_image_transformer.reverse_transform_images(sampled_images.detach().cpu())
        sampled_images = [ImageLoader.load_image(image) for image in sampled_images]
            
        for image_id in range(len(sampled_images)):
            self.save_images(sampled_images[image_id], path_folder + f"img_{image_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=200, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss"
            }
        }