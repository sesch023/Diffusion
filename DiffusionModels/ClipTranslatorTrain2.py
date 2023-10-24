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

wandb.init()
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

resume_from_checkpoint = True
gpus=[2]
device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
clip = ClipTools(device=device)
translator = ClipTranslator(in_out_dim=clip.get_clip_emb_size())
wandb_logger = WandbLogger()
batch_size = 64
wandb.save("*.py*")

data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01100}.tar", "/home/archive/CC3M/cc3m/{00000..00300}.tar"],
    ["/home/archive/CC12M/cc12m/{01101..01242}.tar", "/home/archive/CC3M/cc3m/{00301..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00059}.tar"],
    batch_size=batch_size
)  

model_out="clip_translator/model.ckpt"
model_out_final = "clip_translator/final.ckpt"
old_checkpoint = glob.glob(model_out)
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None
model = ClipTranslatorTrainer(translator, device=device, model_out=model_out)

train_batches = (int((11e6 + 3e6) / batch_size) + 1) 
val_batches = int(320000 / batch_size) + 1

print(train_batches)

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

trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
trainer.test(model, dataloaders=data.test_dataloader())
torch.save(model.state_dict(), model_out_final)


