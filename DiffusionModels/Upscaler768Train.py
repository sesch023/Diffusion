import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
from torchinfo import summary
import wandb
import glob

from DiffusionModules.Diffusion import DiffusionTools, LinearScheduler
from DiffusionModules.DiffusionTrainer import UpscalerDiffusionTrainer
from DiffusionModules.DiffusionModels import UpscalerUNet
from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider

"""
This is the main file for training a upscaling diffusion 
model with a linear scheduler
and a high output resolution of 768x768.
This model was not reported in the thesis. 
"""

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
batch_size = 2
sample_images_out_base_path="samples_upscale_768/"
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = True

# Initialize the high res data module
data = WebdatasetDataModule(
    ["/home/archive/CC12M_HIGH_RES/cc12m/{00000..01200}.tar"],
    ["/home/archive/CC12M_HIGH_RES/cc12m/{01201..01210}.tar"],
    ["/home/archive/CC12M_HIGH_RES/cc12m/{01211..01221}.tar"],
    batch_size=batch_size,
    img_in_target_size=768
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

# Initialize the UNet model
unet = UpscalerUNet(device=device).to(device)
summary(unet, [(1, 6, 768, 768), (1, 256), (1, 512)], verbose=1)

# Find the latest checkpoint in the sample_images_out_base_path
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

clip_tools = ClipTools(device=device)
# Create the DiffusionTrainer instance
model = UpscalerDiffusionTrainer(
    unet, 
    start_size=192,
    target_size=768,
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
# Initialize the trainer
trainer = pl.Trainer(
    limit_train_batches=200, 
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