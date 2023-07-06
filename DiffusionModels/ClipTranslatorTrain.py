from DiffusionModules.ClipTranslatorModules import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
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
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import webdataset as wds
from PIL import Image
import wandb
import copy
from abc import ABC, abstractmethod
from super_image import DrlnModel, ImageLoader
import braceexpand
from DiffusionModules.Diffusion import ClipTools

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[1]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
clip = ClipTools(device=device)
translator = ClipTranslator(in_out_dim=clip.get_clip_emb_size())
wandb_logger = WandbLogger()
batch_size = 256
wandb.save("*.py*")

data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdataset/mscoco/{00000..00059}.tar"],
    batch_size=batch_size
)  

model_out="clip_translator/model.ckpt"
model_out_final = "clip_translator/final.ckpt"
model = ClipTranslatorTrainer(translator, device=device, model_out=model_out)

train_batches = int((11e6 + 3e6) / batch_size) + 1
val_batches = int(320000 / batch_size) + 1

lr_monitor = cb.LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    limit_train_batches=train_batches//2, 
    limit_val_batches=val_batches,
    check_val_every_n_epoch=1, 
    num_sanity_val_steps=0, 
    max_epochs=20, 
    logger=wandb_logger, 
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5), lr_monitor],
    devices=gpus
)

trainer.fit(model, data)
torch.save(model.state_dict(), model_out_final)


