from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
import os
import torch
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

gpus=[0]
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = BasicUNet(device=device)
wandb.init()
wandb_logger = WandbLogger()
batch_size = 8
wandb.save("*.py*")

# url_train = "/home/shared-data/LAION-400M/laion400m-data/{00010..99999}.tar"
# url_test = "/home/shared-data/LAION-400M/laion400m-data/{00000..00009}.tar"

data = WebdatasetDataModule(
    ["/home/archive/CocoWebdataset/mscoco/{00000..00055}.tar"],
    ["/home/archive/CocoWebdataset/mscoco/{00056..00059}.tar"],
    batch_size=batch_size
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_images_out_base_path="samples_coco/"
model = DiffusionTrainer(
    unet, 
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=100, 
    limit_val_batches=10, 
    num_sanity_val_steps=0, 
    max_epochs=10000, 
    logger=wandb_logger, 
    default_root_dir="model/", 
    # gradient_clip_val=1.0, 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)

trainer.fit(model, data)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")