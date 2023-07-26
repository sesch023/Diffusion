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
from lightning.pytorch import Trainer
import webdataset as wds
from PIL import Image
import numpy as np
import wandb
import copy
from abc import ABC, abstractmethod
import glob

resume_from_checkpoint = True
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
device = "cpu"
unet = BasicUNet(device=device).to(device)
wandb.init()
wandb_logger = WandbLogger()
batch_size = 2
wandb.save("*.py*")


unet_in_channels = 3
unet_in_size = 64
unet = SpatioTemporalUNet(
    base_channels=256,
    base_channel_mults=(1, 2, 3, 4),
    res_blocks_per_resolution=3,
    use_res_block_scale_shift_norm=True,
    attention_at_downsample_factor=(2, 4, 8),
    mha_heads=(4, 4, 4),
    mha_head_channels=None,
    i_emb_size=512,
    t_emb_size=256,
    f_emb_size=256,
    mid_emb_size=1024,
    in_channels=unet_in_channels,
    out_channels=unet_in_channels*2,
    device=device
).to(device)

summary(unet, [(1, unet_in_channels, 32, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)

dataset = VideoDatasetDataModule(
    # "/home/archive/Webvid/webvid_2M/results_2M_train.csv", 
    # "/home/archive/Webvid/webvid_2M/data/videos", 
    "/home/archive/Webvid/webvid_2M/results_2M_val.csv", 
    "/home/archive/Webvid/webvid_2M/data_val/videos",
    "/home/archive/Webvid/webvid_2M/results_2M_val.csv", 
    "/home/archive/Webvid/webvid_2M/data_val/videos",
    batch_size=batch_size,
    num_workers=4,
    nth_frames=10,
    max_frames_per_part=16
)

captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_images_out_base_path="samples_video/"
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

model = VideoDiffusionTrainer(
    unet, 
    video_transformable_data_module=dataset,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    temporal_embedding_provider=ClipVideoEmbeddingProvider(clip_tools=clip_tools)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=30000, 
    logger=wandb_logger, 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)

trainer.fit(model, dataset, ckpt_path=resume_from_checkpoint)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")
