from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.ClipTranslatorModules import *
from DiffusionModules.DataModules import *
import os
import torch
import sys
from torch import optim, nn, utils, Tensor
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.multimodal import CLIPScore
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import numpy as np
import wandb
import copy
from abc import ABC, abstractmethod
from torchinfo import summary
import glob

 

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

resume_from_checkpoint = True

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
unet = UpscalerUNet(device=device).to(device)
wandb.init()
wandb_logger = WandbLogger()
summary(unet, [(1, 6, 768, 768), (1, 256), (1, 512)], verbose=1)
batch_size = 1
wandb.save("*.py*")

# url_train = "/home/shared-data/LAION-400M/laion400m-data/{00010..99999}.tar"
# url_test = "/home/shared-data/LAION-400M/laion400m-data/{00000..00009}.tar"

data = WebdatasetDataModule(
    ["/home/archive/CC12M_HIGH_RES/cc12m/{00000..01200}.tar"],
    ["/home/archive/CC12M_HIGH_RES/cc12m/{01201..01242}.tar"],
    batch_size=batch_size,
    img_in_target_size=768
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_images_out_base_path="samples_upscale_768/"
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None
model = UpscalerDiffusionTrainer(
    unet, 
    start_size=192,
    target_size=768,
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    # alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=30000, 
    logger=wandb_logger, 
    default_root_dir="model/", 
    #gradient_clip_val=1.0, 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)

trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")