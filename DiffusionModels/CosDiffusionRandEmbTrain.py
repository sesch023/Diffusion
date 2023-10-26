import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
from torchinfo import summary
import wandb
import glob

from DiffusionModules.Diffusion import DiffusionTools, CosineScheduler
from DiffusionModules.DiffusionTrainer import DiffusionTrainer, UpscalerMode
from DiffusionModules.DiffusionModels import BasicUNet
from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools, ClipRandomImageTextTranslatorEmbeddingProvider, ClipTranslatorEmbeddingProvider
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
This is the main file for training a Diffusion Model with random embedding types.
This was tested to see if the problems with translator and text embeddings could be solved that way.
The model was not reported in the thesis.
"""
batch_size = 4
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
translator_model_path = ModelLoadConfig.translator_model_path
sample_images_out_base_path= "samples_cos_diffusion_rand_emb/"
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = False

# Initialize the data module
data = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    num_workers=4
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
unet = BasicUNet(device=device).to(device)
# Print the model summary
summary(unet, [(1, 3, 64, 64), (1, 256), (1, 512)], verbose=1)
    
# Find the latest checkpoint in the sample_images_out_base_path
old_checkpoint = glob.glob(f"{sample_images_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{sample_images_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

clip_tools = ClipTools(device=device)
embedding_provider = ClipRandomImageTextTranslatorEmbeddingProvider(translator_model_path=translator_model_path, clip_tools=clip_tools)

# Create the DiffusionTrainer instance
model = DiffusionTrainer(
    unet, 
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, steps=1000, noise_scheduler=CosineScheduler()), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=embedding_provider,
    alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path),
    sample_upscaler_mode=UpscalerMode.NONE,
    c_device=device
)

# Create the trainer
lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=5, 
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
trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
# Save the final model
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")