import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
import wandb
from torchinfo import summary

from DiffusionModules.Diffusion import DiffusionTools, LinearScheduler
from DiffusionModules.DiffusionModels import UNet
from DiffusionModules.DataModules import WebdatasetDataModule, CollateType
from DiffusionModules.LatentDiffusionTrainer import LatentDiffusionTrainer
from DiffusionModules.ModelLoading import load_vqgan
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
This is a latent diffusion model with 100 steps. It is a legacy model, that was not reported in the thesis.
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
batch_size = 8
num_workers = 4
latent_shape = (3, 64, 64) 
sample_images_out_base_path="samples_latent_100/"

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

clip_tools = ClipTools(device=device)

# Initialize the LatentDiffusionTrainer
model = LatentDiffusionTrainer(
    unet, 
    vqgan=vqgan,
    latent_shape=latent_shape,
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=100, noise_scheduler=LinearScheduler(), clamp_x_start_in_sample=True), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')

# Create the trainer
trainer = pl.Trainer(
    limit_train_batches=100, 
    check_val_every_n_epoch=200, 
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

# Train the model
trainer.fit(model, data)
# Save the model
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")
