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
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig
# Name Changed, Hack for Model loading
sys.modules['DiffusionModules.DiffusionTrainer'].ClipVideoEmbeddingProvider = ClipTextEmbeddingProvider

"""
This is the main file for training a SpatioTemporalDiffusion Model with a linear scheduler.
This is the final version of the code used for the experiments in the thesis.

The results of this model were described in the chapter:
7.5. Spatiotemporal Decoder des Make-A-Video-Systems
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
num_workers = 2
skip_spatio = True
batch_size = 2
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
sample_out_base_path="samples_spatio_temporal_test/"
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = True

# Initialize the data module for the spatial model
spatial_dataset = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_TUPLE,
    num_workers=num_workers
)  

# Initialize the data module for the temporal model
temporal_dataset = VideoDatasetDataModule(
    DatasetLoadConfig.webvid_10m_train_csv,
    DatasetLoadConfig.webvid_10m_train_data,
    DatasetLoadConfig.webvid_10m_val_csv,
    DatasetLoadConfig.webvic_10m_val_data,
    DatasetLoadConfig.webvid_10m_val_csv,
    DatasetLoadConfig.webvic_10m_val_data,
    batch_size=batch_size,
    num_workers=num_workers,
    nth_frames=1,
    max_frames_per_part=16,
    min_frames_per_part=4,
    first_part_only=True
)

if not os.path.exists(sample_out_base_path):
    os.makedirs(sample_out_base_path)

# Initialize the context
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

# Initialize the logger
wandb.init()
wandb_logger = WandbLogger()
wandb.save("*.py*")

# Initialize the UNet model
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

# Print the model summary for the spatial and temporal model
print("Summary Temporal")
summary(unet, [(1, unet_in_channels, 16, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)
print("Summary Spatial")
summary(unet, [(1, unet_in_channels, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)

# Find the latest checkpoint in the sample_images_out_base_path
old_checkpoint = glob.glob(f"{sample_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_out_base_path}/*.ckpt")
resume_from_checkpoint_path = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

clip_tools = ClipTools(device=device)
if not skip_spatio:
    # Initialize the spatial SpatioTemporalDiffusionTrainer
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
    # Initialize the trainer
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

    # Start the training
    trainer.fit(model, spatial_dataset, ckpt_path=resume_from_checkpoint_path)

# Initialize the temporal SpatioTemporalDiffusionTrainer
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
    disable_temporal_embs=False
)

temporal_lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
# Initialize the trainer
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
# Start the training
temporal_trainer.fit(model, temporal_dataset, ckpt_path=resume_from_checkpoint_path)
# Save the model
torch.save(model.state_dict(), f"{sample_out_base_path}/model.ckpt")


