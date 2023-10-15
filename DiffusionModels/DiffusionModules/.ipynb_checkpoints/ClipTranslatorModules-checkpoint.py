import torch
import torch.nn as nn
from torchinfo import summary
import math
import numpy as np
import clip
import os
import shutil
from torch import optim, nn, utils, Tensor
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import wandb
import copy
from abc import ABC, abstractmethod
from super_image import DrlnModel, ImageLoader


class ClipTranslator(nn.Module):
    def __init__(self, in_out_dim=512, mid_dim=1024, num_mid=30, dropout=0.1):
        super().__init__()
        self._in_out_dim = in_out_dim
        self._in_layer = nn.Sequential(
            nn.Linear(in_out_dim, in_out_dim),
            nn.LayerNorm(in_out_dim)
        )
        self._sequential_mids = nn.ModuleList([nn.Sequential(
            nn.Linear(in_out_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mid_dim, in_out_dim),
            nn.Dropout(p=dropout)
        ) for i in range(num_mid)])
        
        self._out_layer = nn.Sequential(
            nn.LayerNorm(in_out_dim),
            nn.Linear(in_out_dim, in_out_dim)
        )
        
    def forward(self, x):
        current_out = self._in_layer(x)
        for layer in self._sequential_mids:
            current_out = layer(current_out) + current_out
        return self._out_layer(current_out)
      
        
class ClipTranslatorTrainer(pl.LightningModule):
    def __init__(self, model, device=None, loss=None, optimizer=None, clip_tools=None, model_out="clip_translator/model.ckpt"):
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model.to(self.dev)
        
        from DiffusionModules.EmbeddingTools import ClipTools
        
        self.clip_tools = ClipTools(device=self.dev) if clip_tools is None else clip_tools
        self.loss = nn.MSELoss() if loss is None else loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0001) if optimizer is None else optimizer
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_identity_outputs = []
        self.best_loss = float("inf")
        self.model_out = model_out
        self.save_hyperparameters()
    
    def eval_items(self, images, captions, ret_emb=False):
        cap_emb = self.clip_tools.get_clip_emb_text(captions).to(self.dev) 
        img_emb = self.clip_tools.get_clip_emb_images(images).to(self.dev) 
        model_out = self.model(cap_emb)
        loss = self.loss(img_emb, model_out)
        return (loss, cap_emb, img_emb) if ret_emb else loss
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        loss, cap_emb, img_emb = self.eval_items(images, captions, True)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                identity_loss = self.loss(img_emb, cap_emb)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
            self.log("identity_loss", identity_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        loss = self.eval_items(images, captions)
        self.validation_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, captions = batch
        loss, cap_emb, img_emb = self.eval_items(images, captions, True)  
        with torch.no_grad():
            identity_loss = self.loss(img_emb, cap_emb)
            self.test_step_identity_outputs.append(identity_loss)
        self.test_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            print(f"Saving Checkpoint at: {self.model_out}")
            self.trainer.save_checkpoint(self.model_out)
        
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        avg_loss = sum(self.test_step_outputs) / len(self.test_step_outputs)
        avg_identity_loss = sum(self.test_step_identity_outputs) / len(self.test_step_identity_outputs)
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_id_loss", avg_identity_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()
        self.test_step_identity_outputs.clear()
    
    def configure_optimizers(self):
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5000, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss_step"
            }
        }
    