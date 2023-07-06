from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.LatentDiffusionTrainer import *
from DiffusionModules.LatentVQGANModel import *
import os
import torch
from torch import optim, nn, utils, Tensor
import torchvision
import torchvision.transforms as transforms
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


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
wandb.init()
wandb_logger = WandbLogger()
batch_size = 16
num_workers = 8
wandb.save("*.py*")

def load_vqgan():
    z_channels = 3
    shared_args = dict(
        z_channels=z_channels,
        ch=128,
        resolution=256,
        num_res_blocks=2,
        attn_resolutions=[],
        ch_mult=(1,2,4),
        dropout=0.0,
        emb_size=512,
        out_emb_size=1024
    )

    encoder = Encoder(
        in_channels=3,
        double_z=False,
        **shared_args
    )

    print("Encoder")
    summary(encoder, [(1, encoder.in_channels, shared_args["resolution"], shared_args["resolution"]), (1, 512)], verbose=1)

    decoder = Decoder(
        out_channels=3,
        **shared_args
    )

    decoder_in_res = shared_args["resolution"] // (2 ** (len(shared_args["ch_mult"])-1))
    print("Decoder")
    summary(decoder, [(1, z_channels, decoder_in_res, decoder_in_res), (1, 512)], verbose=1)

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

    clip_tools = ClipTools(device=device)
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)

    return VQModel.load_from_checkpoint(
        "vqgan.ckpt", 
        device=device, 
        encoder=encoder, 
        decoder=decoder, 
        loss=loss, 
        transformable_data_module=None,
        embedding_provider=emb_prov
    )


data = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdataset/mscoco/{00000..00059}.tar"],
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_TUPLE,
    num_workers=num_workers,
    img_in_target_size=256
)  
        
captions_preprocess = lambda captions: [cap[:77] for cap in captions]

vqgan = load_vqgan()

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

clip_tools = ClipTools(device=device)
translator_model_path = "clip_translator/model.ckpt"
sample_images_out_base_path="samples_laten_diffusion/"
model = LatentDiffusionTrainer(
    unet, 
    vqgan=vqgan,
    transformable_data_module=data,
    diffusion_tools=DiffusionTools(device=device, in_size=unet_in_size, steps=1000, noise_scheduler=LinearScheduler(), clamp_x_start_in_sample=False), 
    captions_preprocess=captions_preprocess,
    sample_images_out_base_path=sample_images_out_base_path,
    checkpoint_every_val_epochs=1,
    embedding_provider=ClipEmbeddingProvider(clip_tools=clip_tools),
    # alt_validation_emb_provider=ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
)

lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(
    limit_train_batches=100, 
    check_val_every_n_epoch=100, 
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

trainer.fit(model, data)
torch.save(model.state_dict(), f"{sample_images_out_base_path}/model.ckpt")
