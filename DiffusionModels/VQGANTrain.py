from DiffusionModules.LatentVQGANModel import *
from DiffusionModules.LatentVQGANModules import *
from DiffusionModules.VQGANLosses import *
from DiffusionModules.DataModules import *
import os
import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.multimodal import CLIPScore
import lightning.pytorch.callbacks as cb
import webdataset as wds
from PIL import Image
import numpy as np
import wandb
import copy
from abc import ABC, abstractmethod
import braceexpand
from torchinfo import summary


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
wandb.init()
wandb_logger = WandbLogger()
batch_size = 16
num_workers = 8
wandb.save("*.py*")

data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdataset/mscoco/{00000..00059}.tar"],
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_DICT,
    num_workers=num_workers,
    img_in_target_size=256
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

# Definition nicht nach VQGAN Paper sondern https://github.com/CompVis/stable-diffusion/

z_channels = 3
shared_args = dict(
    z_channels=z_channels,
    ch=128,
    resolution=256,
    num_res_blocks=2,
    attn_resolutions=[],
    ch_mult=(1,2,4),
    dropout=0.0,
    emb_size=None,
    out_emb_size=None
)

encoder = Encoder(
    in_channels=3,
    double_z=False,
    **shared_args
)

print("Encoder")
summary(encoder, (1, encoder.in_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

decoder = Decoder(
    out_channels=3,
    **shared_args
)

decoder_in_res = shared_args["resolution"] // (2 ** (len(shared_args["ch_mult"])-1))
print("Decoder")
summary(decoder, (1, z_channels, decoder_in_res, decoder_in_res), verbose=1)

discriminator = NLayerDiscriminator(
    input_nc=decoder.out_channels,
    n_layers=3,
    ndf=64
)

print("Discriminator")
summary(discriminator, (1, decoder.out_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

loss = VQLPIPSWithDiscriminator(
    discriminator=discriminator,
    disc_start=0,
    disc_weight=0.75,
    codebook_weight=1.0,
    disc_conditional=False
)

reconstructions_out_base_path = "reconstructions/"
model = VQModel(
    encoder=encoder,
    decoder=decoder,
    loss=loss,
    transformable_data_module=data,
    n_codebook_embeddings=8192,
    codebook_embedding_size=z_channels,
    z_channels=z_channels,
    image_key="image",
    monitor="val/rec_loss",
    remap=None,
    sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    reconstructions_out_base_path = reconstructions_out_base_path,
    checkpoint_every_val_epochs = 1,
    learning_rate=4.5e-6
)

trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=10, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=10000, 
    logger=wandb_logger, 
    #gradient_clip_val=0.008, 
    #gradient_clip_algorithm="norm", 
    devices=gpus
)

trainer.fit(model, data)
torch.save(model.state_dict(), f"{reconstructions_out_base_path}/model.ckpt")
