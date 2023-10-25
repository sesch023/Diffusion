import os
import glob

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
import wandb
from torchinfo import summary

from DiffusionModules.Diffusion import DiffusionTools, LinearScheduler
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider
from DiffusionModules.DiffusionModels import UNet
from DiffusionModules.DataModules import WebdatasetDataModule, CollateType
from DiffusionModules.LatentDiffusionTrainer import LatentDiffusionTrainer
from DiffusionModules.ModelLoading import load_vqgan
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
This is the main file for training a LatentDiffusion Model with a cosine scheduler.
A cosine latent diffusion was tested, but not reported in the thesis.
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_workers = 4
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
translator_model_path = ModelLoadConfig.clip_translator_path
sample_images_out_base_path="samples_cos_latent_diffusion/"
latent_shape = (3, 64, 64)
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = False

# Initialize the data module
data = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_TUPLE,
    num_workers=num_workers,
    img_in_target_size=256
)  

if not os.path.exists(sample_images_out_base_path):
    os.makedirs(sample_images_out_base_path)

# Initialize the context
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

# Initialize the logger
wandb.init()
wandb_logger = WandbLogger()
wandb.save("*.py*")

# Initialize the VQGAN model
vqgan_path = ModelLoadConfig.vqgan_path
vqgan = load_vqgan(vqgan_path, device=device)

print("Creating UNet")

# Initialize the UNet model
unet_in_channels = 3
unet_in_size = 64
unet = UNet(
    base_channels=256,
    base_channel_mults=(1, 2, 3, 4),
    res_blocks_per_resolution=3,
    use_res_block_scale_shift_norm=True,
    attention_at_downsample_factor=(2, 4, 8),
    mha_heads=(4, 4, 4),
    mha_head_channels=None,
    i_emb_size=512,
    t_emb_size=256,
    mid_emb_size=1024,
    in_channels=unet_in_channels,
    out_channels=unet_in_channels*2,
    device=device
)
# Print the model summary
summary(unet, [(1, *latent_shape), (1, 256), (1, 512)], verbose=1)

# Load the checkpoint if resuming from a checkpoint
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

clip_tools = ClipTools(device=device)
# Create the LatentDiffusionTrainer instance
model = LatentDiffusionTrainer(
    unet, 
    vqgan=vqgan,
    latent_shape=latent_shape,
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler(), clamp_x_start_in_sample=False), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    quantize_after_sample=False
)

# Create the trainer
lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=100, 
    check_val_every_n_epoch=200, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=30000, 
    logger=wandb_logger, 
    default_root_dir="model/", 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)
# Fit the model
trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
# Save the model
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")
