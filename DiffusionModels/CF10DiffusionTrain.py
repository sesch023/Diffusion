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
torch.autograd.set_detect_anomaly(True)

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wandb.init()
wandb_logger = WandbLogger()
batch_size = 16

gpus=[0]
cifar_data = CIFAR10DataModule(batch_size=batch_size)
unet = BasicUNet(i_emb_size=len(CIFAR10DataModule.classes), device=device).to(device)
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 10)], verbose=1)
sample_images_out_base_path = "samples_cifar2/"
wandb.save("*.py*")


sample_images_out_base_path = "samples_cifar"
captions_preprocess = None
model = DiffusionTrainer(
    unet, 
    transformable_data_module=cifar_data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=LinearScheduler()), 
    embedding_provider=CF10EmbeddingProvider(),
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=100, 
    check_val_every_n_epoch=10, 
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
trainer.fit(model, cifar_data)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")