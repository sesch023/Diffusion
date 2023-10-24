import copy
import os
import shutil
import sys
from einops import rearrange
from enum import Enum

import lightning.pytorch as pl
import torch
import torchvision.transforms as transforms
from diffusers import LDMSuperResolutionPipeline
from super_image import DrlnModel, ImageLoader
from torch import nn, optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

import DiffusionModules.ClipTranslatorModules as tools
from DiffusionModules.FVD import FVDLoss
from DiffusionModules.DiffusionModels import ExponentialMovingAverage
from DiffusionModules.ModelLoading import load_udm
from DiffusionModules.EmbeddingTools import ClipTextEmbeddingProvider, ClipEmbeddingProvider
from Configs import ModelLoadConfig

sys.modules['ClipTranslatorModules'] = tools


class UpscalerMode(Enum):
    NONE="NONE",
    DRLN="DRLN",
    LDM="LDM",
    # Due to a bug with model loading and enum changes this had to be removed. 
    # Instead for now a string is used!
    # UDM="UDM"
    
class DiffusionTrainer(pl.LightningModule):
    def __init__(self, unet, diffusion_tools, transformable_data_module, loss=None, val_score=None, embedding_provider=None, alt_validation_emb_provider=None, ema_beta=0.9999, cfg_train_ratio=0.1, cfg_scale=3, captions_preprocess=None, optimizer=None, sample_upscaler_mode=UpscalerMode.LDM, sample_scale_factor=4, checkpoint_every_val_epochs=10, no_up_samples_out=True, sample_images_out_base_path="samples/", c_device="cpu"):
        super().__init__()
        self.unet = unet
        self.diffusion_tools = diffusion_tools
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
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
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        self.no_up_samples_out = no_up_samples_out
        self.c_device = c_device
        
        self.upscaler = None
        if self.sample_upscaler_mode is not None and self.sample_upscaler_mode != UpscalerMode.NONE:
            if self.sample_upscaler_mode == UpscalerMode.DRLN:
                self.up_model_pipeline = DrlnModel.from_pretrained('eugenesiow/drln', scale=sample_scale_factor)
                self.upscaler = lambda image, caption: self.up_model_pipeline(ImageLoader.load_image(image).to(self.device)).detach().to(self.c_device)     
                self.save_images = lambda image, path: ImageLoader.save_image(image, path)
            elif self.sample_upscaler_mode == UpscalerMode.LDM:
                model_id = "CompVis/ldm-super-resolution-4x-openimages"
                self.up_model_pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id).to(self.c_device)
                self.upscaler = lambda image, caption: self.up_model_pipeline(ImageLoader.load_image(image).to(self.device), num_inference_steps=100, eta=1).images[0].detach().cpu() 
                self.save_images = lambda image, path: image.save(path)
            # This should later Change to a Enum value, but changing a enum with pickled models does not work and results in load errors.
            elif self.sample_upscaler_mode == "UDM":
                model_path = ModelLoadConfig.upscaler_model_path
                self.up_model_pipeline = load_udm(model_path, self.c_device, self.transformable_data_module.img_in_target_size*sample_scale_factor)
                self.up_model_pipeline.eval()
                self.upscaler = lambda image, caption: self.up_model_pipeline([image], [caption], ema=True)[0]
                self.save_images = lambda image, path: image.save(path)
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
        
        if val_score is None:
            self.fid = FrechetInceptionDistance(feature=2048)
            
            def get_fid(samples, real):
                self.fid.reset()
                self.fid.update((((samples + 1)/2)*255).byte(), real=False)
                self.fid.update((((real + 1)/2)*255).byte(), real=True)
                fid = self.fid.compute()
                self.fid.reset()
                return fid
            
            self.clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").eval()
            
            self.val_score = lambda samples, real, captions: {
                "clip_score": self.clip_model((((samples + 1)/2)*255).int(), captions),
                "fid_score": get_fid(samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
        
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.device)
        else:
            i_embs = None
        images = self.transformable_data_module.transform_batch(images).to(self.device)  
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions    
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        images = self.transformable_data_module.transform_batch(images).to(self.device)    
        sampled_images = self.diffusion_tools.sample_data(self.unet, images.shape, i_embs, self.cfg_scale)
        self.save_sampled_images(sampled_images, captions, batch_idx, "normal")
        if self.upscaler is not None and self.no_up_samples_out:
            self.save_sampled_images(sampled_images, captions, batch_idx, "normal_no_up", no_upscale=True)

        if self.ema is not None:
            sampled_images = self.diffusion_tools.sample_data(self.ema_unet, images.shape, i_embs, self.cfg_scale)
            self.save_sampled_images(sampled_images, captions, batch_idx, "ema")
            if self.upscaler is not None and self.no_up_samples_out:
                self.save_sampled_images(sampled_images, captions, batch_idx, "ema_no_up", no_upscale=True)
        
        try:
            val_score = self.val_score(sampled_images, images, captions)
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
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_score"]
        
        path = f"{self.sample_images_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None, no_upscale=False):
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
            
        if self.upscaler is not None and not no_upscale:
            sampled_images = [self.upscaler(image, caption) for image, caption in zip(sampled_images, captions)]
            
        for image_id in range(len(sampled_images)):
            self.save_images(sampled_images[image_id], path_folder + f"img_{image_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }
    
    
class UpscalerDiffusionTrainer(pl.LightningModule):
    def __init__(self, unet, diffusion_tools, transformable_data_module, start_size=64, target_size=256, loss=None, val_score=None, embedding_provider=None, alt_validation_emb_provider=None, ema_beta=0.9999, cfg_train_ratio=0.1, cfg_scale=3, captions_preprocess=None, optimizer=None, checkpoint_every_val_epochs=10, sample_images_out_base_path="samples_upscale/"):#
        super().__init__()
        self.unet = unet 
        self.diffusion_tools = diffusion_tools  
        self.start_size = start_size
        self.target_size = target_size
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-5, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
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
            self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
            
            def get_fid(samples, real):
                self.fid.reset()
                self.fid.update((((samples + 1)/2)*255).byte(), real=False)
                self.fid.update((((real + 1)/2)*255).byte(), real=True)
                fid = self.fid.compute()
                self.fid.reset()
                return fid
            
            self.val_score = lambda samples, real, captions: {
                "fid_score": get_fid(samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
        

    def forward(self, images, captions, ema=True):
        model = self.ema_unet if ema and self.ema is not None else self.unet
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        low_res = torch.stack([self.transform_low_res(image) for image in images]).to(self.device)
        sampled_images = self.diffusion_tools.sample_data(model, low_res.shape, i_embs, self.cfg_scale, x_appendex=low_res)
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
        return sampled_images


    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.device)
        else:
            i_embs = None
        
        low_res = torch.stack([self.transform_low_res(image).to(self.device) for image in images]).to(self.device)  
        images = self.transformable_data_module.transform_batch(images).to(self.device)   
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs, x_unnoised_appendex=low_res)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        low_res = torch.stack([self.transform_low_res(image).to(self.device) for image in images])   
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        images = self.transformable_data_module.transform_batch(images).to(self.device)
        sampled_images = self.diffusion_tools.sample_data(self.unet, images.shape, i_embs, self.cfg_scale, x_appendex=low_res)
        self.save_sampled_images(sampled_images, captions, batch_idx, "normal") 

        if self.ema is not None:
            sampled_images = self.diffusion_tools.sample_data(self.ema_unet, images.shape, i_embs, self.cfg_scale, x_appendex=low_res)
            self.save_sampled_images(sampled_images, captions, batch_idx, "ema")

        self.save_sampled_images(low_res, captions, batch_idx, "low_res")
        
        try:
            val_score = self.val_score(sampled_images, images, captions)
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
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_score"]
        
        path = f"{self.sample_images_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None):
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
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
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }


class SpatioTemporalDiffusionTrainer(pl.LightningModule):
    def __init__(
        self, 
        unet, 
        diffusion_tools, 
        transformable_data_module, 
        loss=None, 
        val_score=None, 
        embedding_provider=None, 
        temporal_embedding_provider=None,
        alt_validation_emb_provider=None, 
        alt_validation_temporal_emb_provider=None,
        ema_beta=0.9999, 
        cfg_train_ratio=0.1, 
        cfg_scale=3, 
        captions_preprocess=None, 
        optimizer=None, 
        checkpoint_every_val_epochs=10, 
        sample_data_out_base_path="samples_spatio_temporal/",
        disable_temporal_caption_embs=True,
        temporal=True,
        after_load_fvd=False
    ):
        super().__init__()
        self.unet = unet
        self.diffusion_tools = diffusion_tools
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.temporal_embedding_provider = ClipTextEmbeddingProvider() if temporal_embedding_provider is None else temporal_embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.alt_validation_temporal_emb_provider = self.temporal_embedding_provider if alt_validation_temporal_emb_provider is None else alt_validation_temporal_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_data_out_base_path = sample_data_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-5, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.temporal = temporal
        self.writer_module = self.transformable_data_module.t_data if self.transformable_data_module.t_data is not None else self.transformable_data_module.v_data
        if self.writer_module is None:
            print("Warning: No transformable dataset module found for saving samples!")
        self.save_videos= lambda video, path: self.writer_module.write_video(video, path)
        self.save_images = lambda image, path: image.save(path)
        self.disable_temporal_caption_embs = disable_temporal_caption_embs
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
        
        if val_score is None:
            self.fid = FrechetInceptionDistance(feature=2048)
            self.clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").eval()
            if not after_load_fvd:
                self.load_fvd()

            self.val_score = lambda samples, real, captions: {
                "clip_score": self.get_clip_score(samples, captions),
                "fid_fvd_score": self.get_fvd_fid(samples, real),
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
    
    def load_fvd(self):
        self.fvd = FVDLoss(self.device)

    def get_fvd_fid(self, s_data, r_data):
        if s_data.ndim == 5:
            score = self.fvd(s_data, r_data)
            return score

        self.fid.reset()
        self.fid.update((((s_data + 1)/2)*255).byte(), real=False)
        self.fid.update((((r_data + 1)/2)*255).byte(), real=True)
        fid = self.fid.compute()
        self.fid.reset()
        return fid

    def get_clip_score(self, s_data, captions):
        scores = []
        if s_data.ndim == 4:
            s_data = rearrange(s_data, 'b c h w -> b 1 c h w')
        else:
            s_data = rearrange(s_data, 'b c t h w -> b t c h w')
            
        for b_id in range(s_data.shape[0]):
            for f_id in range(s_data.shape[1]):
                score = self.clip_model((((s_data[b_id][f_id] + 1)/2)*255).int(), captions[b_id])
                scores.append(score)

        return sum(scores) / len(scores)
    

    def training_step(self, batch, batch_idx):
        data, captions, *other = batch  
        fps = other[1] if len(other) > 1 else None
        length = other[0] if len(other) > 0 else None
        f_emb = None
        transformed_data = self.transformable_data_module.transform_batch(data).to(self.device) 
        temporal = self.temporal and transformed_data.ndim == 5

        if temporal:
            f_emb = torch.stack([self.diffusion_tools.get_pos_encoding(f) for f in fps]).to(self._device) 

        embedding_provider = self.embedding_provider
        if transformed_data.ndim == 5:
            embedding_provider = self.temporal_embedding_provider

        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions

        if torch.rand(1)[0] > self.cfg_train_ratio and not (temporal and self.disable_temporal_caption_embs):
            d_embs = embedding_provider.get_embedding(data, captions).to(self.device)
        else:
            d_embs = None
 
        loss = self.diffusion_tools.train_step(self.unet, self.loss, transformed_data, d_embs, f_emb=f_emb, temporal=temporal)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=transformed_data.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        data, captions, *other = batch  
        fps = other[1] if len(other) > 1 else None
        length = other[0] if len(other) > 0 else None
        f_emb = None
        transformed_data = self.transformable_data_module.transform_batch(data).to(self.device) 
        temporal = self.temporal and transformed_data.ndim == 5
        if temporal:
            f_emb = torch.stack([self.diffusion_tools.get_pos_encoding(f) for f in fps]).to(self._device) 

        notes = ("normal", "ema")
        embedding_provider = self.embedding_provider
        if transformed_data.ndim == 5:
            embedding_provider = self.temporal_embedding_provider
            notes = ("normal_temp", "ema_temp")

        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        d_embs = embedding_provider.get_embedding(data, captions).to(self.device)

        normal_sampled = self.diffusion_tools.sample_data(self.unet, transformed_data.shape, d_embs, self.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=temporal)
        ema_sampled = self.diffusion_tools.sample_data(self.ema_unet, transformed_data.shape, d_embs, self.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=temporal)
        
        self.save_sampled_data(normal_sampled, captions, batch_idx, notes[0])
        self.save_sampled_data(ema_sampled, captions, batch_idx, notes[1])
            
        try:
            val_score = self.val_score(ema_sampled, transformed_data, captions)
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
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_fvd_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_data_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_fvd_score"]
        
        path = f"{self.sample_data_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_data(self, sampled_data, captions, batch_idx, note=None):
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_data_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        temporal = sampled_data.ndim == 5
        
        sampled_data = self.transformable_data_module.reverse_transform_batch(sampled_data.detach().cpu())
            
        for data_id in range(len(sampled_data)):
            if temporal:
                self.save_videos(sampled_data[data_id], path_folder + f"vid_{data_id}.mp4")
            else:
                self.save_images(sampled_data[data_id], path_folder + f"img_{data_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_fvd_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }