import os
import glob

import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as cb
import wandb
from torchinfo import summary
from lightning.pytorch.loggers import WandbLogger

from DiffusionModules.Diffusion import LinearScheduler, DiffusionTools
from DiffusionModules.DiffusionTrainer import DiffusionTrainer, UpscalerMode
from DiffusionModules.DiffusionModels import BasicUNet
from DiffusionModules.DataModules import CIFAR10DataModule
from DiffusionModules.EmbeddingTools import CF10EmbeddingProvider

"""
This is the main file for training the CIFAR10 dataset with the Diffusion Model.
This is the final version of the code used for the experiments in the thesis.

The results of this model were described in the chapter:
7.1. Diffusion mit dem CIFAR-10 Datensatz
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
batch_size = 4
sample_images_out_base_path = "samples_cifar"
captions_preprocess = None
# Should the training resume from the latest checkpoint in the sample_images_out_base_path? 
resume_from_checkpoint = False

# Initialize the data module
cifar_data = CIFAR10DataModule(batch_size=batch_size)

if not os.path.exists(sample_images_out_base_path):
    os.makedirs(sample_images_out_base_path)

# Initialize the context
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

# Initialize the logger
wandb.init()
wandb_logger = WandbLogger()
wandb.save("*.py*")

# Intialize the unet model
unet = BasicUNet(i_emb_size=len(CIFAR10DataModule.classes), device=device).to(device)

# Print the model summary
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 10)], verbose=1)

# Find the latest checkpoint in the sample_images_out_base_path
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint_path = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

# Create the DiffusionTrainer instance
model = DiffusionTrainer(
    unet, 
    transformable_data_module=cifar_data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    embedding_provider=CF10EmbeddingProvider(),
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    sample_upscaler_mode=UpscalerMode.NONE,
    c_device=device
)

# Create the trainer and fit the model
lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=2, 
    num_sanity_val_steps=0, 
    max_epochs=20000, 
    logger=wandb_logger, 
    default_root_dir="model/", 
    gradient_clip_val=0.008, 
    gradient_clip_algorithm="norm", 
    callbacks=[lr_monitor],
    devices=gpus
)
# Fit the model
trainer.fit(model, cifar_data, ckpt_path=resume_from_checkpoint_path)
# Save the final model
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")