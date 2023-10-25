import os
import glob
import sys

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchinfo import summary
import lightning.pytorch.callbacks as cb
import wandb

from DiffusionModules.Diffusion import DiffusionTools, LinearScheduler
from DiffusionModules.DiffusionTrainer import SpatioTemporalDiffusionTrainer
from DiffusionModules.DiffusionModels import SpatioTemporalUNet
from DiffusionModules.DataModules import WebdatasetDataModule, VideoDatasetDataModule, CollateType
from DiffusionModules.EmbeddingTools import ClipTextEmbeddingProvider, ClipEmbeddingProvider, ClipTools


# Name Changed, Hack for Model loading
sys.modules['DiffusionModules.DiffusionTrainer'].ClipVideoEmbeddingProvider = ClipTextEmbeddingProvider

resume_from_checkpoint = True
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[2]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
wandb.init()
wandb_logger = WandbLogger()
batch_size = 2
wandb.save("*.py*")
num_workers = 2
skip_spatio = True
print(device)

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

print("Summary Temporal")
summary(unet, [(1, unet_in_channels, 16, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)
print("Summary Spatial")
summary(unet, [(1, unet_in_channels, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)

spatial_dataset = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00059}.tar"],
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_TUPLE,
    num_workers=num_workers
)  

captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_out_base_path="samples_spatio_temporal_test/"
old_checkpoint = glob.glob(f"{sample_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_out_base_path}/*.ckpt")
resume_from_checkpoint_path = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

if not skip_spatio:
    model = SpatioTemporalDiffusionTrainer(
        unet, 
        transformable_data_module=spatial_dataset,
        diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
        captions_preprocess=captions_preprocess,
        sample_data_out_base_path=sample_out_base_path,
        checkpoint_every_val_epochs=1,
        embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
        temporal_embedding_provider=ClipTextEmbeddingProvider(clip_tools=clip_tools),
        temporal=False
    )

    lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        limit_train_batches=200, 
        check_val_every_n_epoch=200, 
        limit_val_batches=2, 
        num_sanity_val_steps=0, 
        max_epochs=2000, 
        logger=wandb_logger, 
        gradient_clip_val=0.008, 
        gradient_clip_algorithm="norm", 
        callbacks=[lr_monitor],
        devices=gpus
    )

    trainer.fit(model, spatial_dataset, ckpt_path=resume_from_checkpoint_path)

temporal_dataset = VideoDatasetDataModule(
    #"/home/shared-data/webvid/results_10M_train.csv", 
    #"/home/shared-data/webvid/data/videos",
    "/home/shared-data/webvid/results_10M_val.csv", 
    "/home/shared-data/webvid/data_val/videos",
    "/home/shared-data/webvid/results_10M_val.csv", 
    "/home/shared-data/webvid/data_val/videos",
    "/home/shared-data/webvid/results_10M_val.csv", 
    "/home/shared-data/webvid/data_val/videos",
    batch_size=batch_size,
    num_workers=num_workers,
    nth_frames=1,
    max_frames_per_part=16,
    min_frames_per_part=4,
    first_part_only=True
)

model = SpatioTemporalDiffusionTrainer(
    unet, 
    transformable_data_module=temporal_dataset,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_data_out_base_path=sample_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    temporal_embedding_provider=ClipTextEmbeddingProvider(clip_tools=clip_tools),
    temporal=True,
    #disable_temporal_embs=False
)

temporal_lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
temporal_trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=2, 
    num_sanity_val_steps=0, 
    max_epochs=20000, 
    logger=wandb_logger, 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[temporal_lr_monitor],
    devices=gpus
)
temporal_trainer.fit(model, temporal_dataset, ckpt_path=resume_from_checkpoint_path)


