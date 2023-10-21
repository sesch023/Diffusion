from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import lightning.pytorch.callbacks as cb
import copy
from torchinfo import summary
import glob
torch.autograd.set_detect_anomaly(True)

resume_from_checkpoint = True
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
wandb.init()
wandb_logger = WandbLogger()
batch_size = 4

cifar_data = CIFAR10DataModule(batch_size=batch_size)
unet = BasicUNet(i_emb_size=len(CIFAR10DataModule.classes), device=device).to(device)
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 10)], verbose=1)
wandb.save("*.py*")


sample_images_out_base_path = "samples_cifar_final"
captions_preprocess = None
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint_path = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None
model = DiffusionTrainer(
    unet, 
    transformable_data_module=cifar_data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    embedding_provider=CF10EmbeddingProvider(),
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    sample_upscaler_mode="UDM",
    c_device=device
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=100, 
    limit_val_batches=2, 
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
trainer.fit(model, cifar_data, ckpt_path=resume_from_checkpoint_path)