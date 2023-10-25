import os
import glob

import wandb
from torchinfo import summary
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from DiffusionModules.LatentVQGANModel import VQModel
from DiffusionModules.LatentVQGANModules import Encoder, Decoder, NLayerDiscriminator
from DiffusionModules.VQGANLosses import VQLPIPSWithDiscriminator
from DiffusionModules.DataModules import WebdatasetDataModule, CollateType
from DiffusionModules.EmbeddingTools import ClipTools
from DiffusionModules.DiffusionTrainer import ClipEmbeddingProvider

"""
This is the main file for training a LatentVQGAN Model with Embeddings.
This is the final version of the code used for the experiments in the thesis.

The results of this model were described in the chapter:
7.4. VQGANs und latente Diffusion
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_workers = 4
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
reconstructions_out_base_path = "emb_reconstructions/"
# Should the training resume from the latest checkpoint in the sample_images_out_base_path?
resume_from_checkpoint = True

# Initialize the data module
data = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_DICT,
    num_workers=num_workers,
    img_in_target_size=256
)  

if not os.path.exists(reconstructions_out_base_path):
    os.makedirs(reconstructions_out_base_path)

# Initialize the context
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

# Initialize the logger
wandb.init()
wandb_logger = WandbLogger()
wandb.save("*.py*")

# Shared arguments for the encoder and decoder
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

# Initializes the encoder
encoder = Encoder(
    in_channels=3,
    double_z=False,
    **shared_args
).to(device)

# Print the model summary of the encoder
print("Encoder")
summary(encoder, [(1, encoder.in_channels, shared_args["resolution"], shared_args["resolution"]), (1, 512)], verbose=1)

# Initializes the decoder
decoder = Decoder(
    out_channels=3,
    **shared_args
).to(device)

# Print the model summary of the decoder
decoder_in_res = shared_args["resolution"] // (2 ** (len(shared_args["ch_mult"])-1))
print("Decoder")
summary(decoder, [(1, z_channels, decoder_in_res, decoder_in_res), (1, 512)], verbose=1)

# Initializes the discriminator
discriminator = NLayerDiscriminator(
    input_nc=decoder.out_channels,
    n_layers=3,
    ndf=64
).to(device)

# Print the model summary of the discriminator
print("Discriminator")
summary(discriminator, (1, decoder.out_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

# Initializes the loss
loss = VQLPIPSWithDiscriminator(
    discriminator=discriminator,
    disc_start=0,
    disc_weight=0.75,
    codebook_weight=1.0,
    disc_conditional=False
).to(device)

# Find the latest checkpoint in the sample_images_out_base_path and resume from it if resume_from_checkpoint is True
old_checkpoint = glob.glob(f"{reconstructions_out_base_path}/latest.ckpt")
old_checkpoint = old_checkpoint if len(old_checkpoint) > 0 else glob.glob(f"{reconstructions_out_base_path}/*.ckpt")
resume_from_checkpoint = None if not resume_from_checkpoint else old_checkpoint[0] if len(old_checkpoint) > 0 else None

clip_tools = ClipTools(device=device)
emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)

# Create the VQModel instance
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
    learning_rate=4.5e-6,
    caption_key="caption",
    embedding_provider=emb_prov
).to(device)

# Create the trainer
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=100, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=20000, 
    logger=wandb_logger, 
    devices=gpus
)

# Fit the model
trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
# Save the model
torch.save(model.state_dict(), f"{reconstructions_out_base_path}/model.ckpt")
