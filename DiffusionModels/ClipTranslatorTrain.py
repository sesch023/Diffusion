import glob
import os

import clip
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch.callbacks as cb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb

from DiffusionModules.ClipTranslatorModules import ClipTranslatorTrainer, ClipTranslator
from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
This is the train file for training a ClipTranslator model.
This is the final version of the code used for the experiments in the thesis.

The results of this model were described in the chapter:
6.2.3. Das EmbeddingTools-Modul und der ClipTranslator
"""
gpus=[0]
device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
batch_size = 64
model_out_base = "clip_translator_final"
model_out= f"{model_out_base}/latest.ckpt"
model_out_final = f"{model_out_base}/final.ckpt"
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = True

# Initialize the data module
data = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size
)  

if not os.path.exists(model_out_base):
    os.makedirs(model_out_base)

# Initialize the context
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

# Initialize the logger
wandb.init()
wandb_logger = WandbLogger()
wandb.save("*.py*")

# Initialize the ClipTranslator model
clip = ClipTools(device=device)
translator = ClipTranslator(in_out_dim=clip.get_clip_emb_size())

# Find the latest checkpoint in the sample_images_out_base_path
old_checkpoint = glob.glob(model_out)
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

# Create the ClipTranslatorTrainer instance
model = ClipTranslatorTrainer(translator, device=device, model_out=model_out)

# About every element in the train dataset
train_batches = (int((11e6 + 3e6) / batch_size) + 1) 
# About every element in the validation dataset
val_batches = int(320000 / batch_size) + 1

# Initialize the trainer
lr_monitor = cb.LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(
    limit_train_batches=int(train_batches / 10), 
    limit_val_batches=int(val_batches / 10),
    limit_test_batches=1000,
    check_val_every_n_epoch=10, 
    num_sanity_val_steps=0, 
    max_epochs=8, 
    logger=wandb_logger, 
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3), lr_monitor],
    devices=gpus,
)

# Fit the model
trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
# Test the model
trainer.test(model, dataloaders=data.test_dataloader())
# Save the model
torch.save(model.state_dict(), model_out_final)

