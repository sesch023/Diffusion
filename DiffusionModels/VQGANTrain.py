import os

import wandb
from torchinfo import summary
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from DiffusionModules.LatentVQGANModel import VQModel
from DiffusionModules.LatentVQGANModules import Encoder, Decoder, NLayerDiscriminator
from DiffusionModules.VQGANLosses import VQLPIPSWithDiscriminator
from DiffusionModules.DataModules import WebdatasetDataModule, CollateType
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
This is the main file for training a LatentVQGAN Model without Embeddings.
This was not used for the experiments in the thesis and is considered legacy for now.
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_workers = 4
captions_preprocess = lambda captions: [cap[:77] for cap in captions]
reconstructions_out_base_path = "reconstructions/"

# Initialize the data module
data = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    collate_type=CollateType.COLLATE_NONE_TUPLE,
    num_workers=num_workers,
    img_in_target_size=256,
    preprocess=None,
    legacy=False
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
    emb_size=None,
    out_emb_size=None
)

# Initialzes the encoder
encoder = Encoder(
    in_channels=3,
    double_z=False,
    **shared_args
)

# Prints the encoder summary
print("Encoder")
summary(encoder, (1, encoder.in_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

# Initializes the decoder
decoder = Decoder(
    out_channels=3,
    **shared_args
)

# Prints the decoder summary
decoder_in_res = shared_args["resolution"] // (2 ** (len(shared_args["ch_mult"])-1))
print("Decoder")
summary(decoder, (1, z_channels, decoder_in_res, decoder_in_res), verbose=1)

# Initializes the discriminator
discriminator = NLayerDiscriminator(
    input_nc=decoder.out_channels,
    n_layers=3,
    ndf=64
)

# Prints the discriminator summary
print("Discriminator")
summary(discriminator, (1, decoder.out_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

# Initializes the loss
loss = VQLPIPSWithDiscriminator(
    discriminator=discriminator,
    disc_start=0,
    disc_weight=0.75,
    codebook_weight=1.0,
    disc_conditional=False
)

# Initializes the model
model = VQModel(
    encoder=encoder,
    decoder=decoder,
    loss=loss,
    transformable_data_module=data,
    n_codebook_embeddings=8192,
    codebook_embedding_size=z_channels,
    z_channels=z_channels,
    image_key=0,
    monitor="val/rec_loss",
    remap=None,
    sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    reconstructions_out_base_path = reconstructions_out_base_path,
    checkpoint_every_val_epochs = 1,
    learning_rate=4.5e-6
)

# Initializes the trainer
trainer = pl.Trainer(
    limit_train_batches=200, 
    check_val_every_n_epoch=200, 
    limit_val_batches=5, 
    num_sanity_val_steps=0, 
    max_epochs=10000, 
    logger=wandb_logger, 
    devices=gpus
)

# Starts the training
trainer.fit(model, data)
# Outputs the final model
torch.save(model.state_dict(), f"{reconstructions_out_base_path}/model.ckpt")
