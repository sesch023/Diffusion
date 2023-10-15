from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.multimodal import CLIPScore
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import numpy as np
from torchinfo import summary
import wandb
import copy
from abc import ABC, abstractmethod
import glob


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

resume_from_checkpoint = False
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
unet = UNet(
    i_emb_size=512,
    t_emb_size=256,
    mid_emb_size=1024,
    out_channels=6,
    device=device,
    mha_head_channels=(64, 64, 64, 64),
    use_sample_conv=True,
    use_functional_att_norm_out=True,
    use_res_block_scale_shift_norm=False,
    attention_at_downsample_factor=[]
)
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 512)], verbose=1)
wandb.init()
wandb_logger = WandbLogger()
batch_size = 4
wandb.save("*.py*")

# url_train = "/home/shared-data/LAION-400M/laion400m-data/{00010..99999}.tar"
# url_test = "/home/shared-data/LAION-400M/laion400m-data/{00000..00009}.tar"

data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    batch_size=batch_size,
    num_workers=4
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "/home/jovyan/DiffusionModels/DiffusionModels/clip_translator/model.ckpt"
sample_images_out_base_path="samples_cos_diffusion_no_att/"
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None
model = DiffusionTrainer(
    unet, 
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=CosineScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    #alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(device=device, clip_tools=clip_tools, translator_model_path=translator_model_path)
    alt_validation_emb_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    sample_upscaler_mode="UDM",
    c_device=device,
    optimizer=optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=0.0)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=100, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=20000, 
    logger=wandb_logger, 
    default_root_dir="model/", 
    # gradient_clip_val=1.0, 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)

trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")