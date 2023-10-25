import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
from torchinfo import summary
import wandb
import glob

from DiffusionModules.Diffusion import DiffusionTools, CosineScheduler
from DiffusionModules.DiffusionTrainer import DiffusionTrainer
from DiffusionModules.DiffusionModels import BasicUNet
from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

resume_from_checkpoint = True
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
unet = BasicUNet(device=device).to(device)
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 512)], verbose=1)
wandb.init()
wandb_logger = WandbLogger()
batch_size = 4
wandb.save("*.py*")

data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    batch_size=batch_size,
    num_workers=4
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

clip_tools = ClipTools(device=device)
translator_model_path = "/home/jovyan/DiffusionModels/DiffusionModels/clip_translator/model.ckpt"
sample_images_out_base_path="samples_cos_diffusion/"
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
    alt_validation_emb_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    sample_upscaler_mode="UDM",
    c_device=device
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