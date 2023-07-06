from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.multimodal import CLIPScore
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import numpy as np
import wandb
import copy
from abc import ABC, abstractmethod
from DiffusionModules.DataModules import *


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
unet = BasicUNet(device=device).to(device)
wandb.init()
wandb_logger = WandbLogger()
batch_size = 16
wandb.save("*.py*")


def preprocess(sample):
    image, json = sample
    label = json["caption"]
    return image, label

# url_train = "/home/shared-data/LAION-400M/laion400m-data/{00010..99999}.tar"
# url_test = "/home/shared-data/LAION-400M/laion400m-data/{00000..00009}.tar"

url_train = "/home/archive/CocoWebdataset/mscoco/{00000..00055}.tar"
url_test = "/home/archive/CocoWebdataset/mscoco/{00056..00059}.tar"


        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_images_out_base_path="samples_cococos/"
model = DiffusionTrainer(
    unet, 
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=CosineScheduler()), 
    device=device, 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(device=device, clip_tools=clip_tools),
    #alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(device=device, clip_tools=clip_tools, translator_model_path=translator_model_path)
    alt_validation_emb_provider=ClipEmbeddingProvider(device=device, clip_tools=clip_tools)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=100, 
    limit_val_batches=5, 
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

trainer.fit(model, loader_train, loader_val)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt.pt")