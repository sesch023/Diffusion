import os
import shutil

import torch
import lightning.pytorch as pl
from DiffusionModules.LatentVQGANModules import Encoder, Decoder, VectorQuantizer
from DiffusionModules.VQGANLosses import VQLPIPSWithDiscriminator

"""
This Module was adapted from:
 https://github.com/CompVis/taming-transformers
"""

"""
Default Encoder and Decoder Arguments for VQGANs
"""
DEFAULT_ENCODER_ARGS = dict(
    ch=128,
    in_channels=3,
    resolution=256,
    num_res_blocks=2,
    attn_resolutions=[],
    ch_mult=(1,1,2,2,4), 
    dropout=0.0, 
    double_z=False,
    emb_size=None,
    out_emb_size=None
)

DEFAULT_DECODER_ARGS = dict(
    ch=128,
    out_channels=3,
    resolution=256,
    num_res_blocks=2,
    attn_resolutions=[],
    ch_mult=(1,1,2,2,4), 
    dropout=0.0, 
    emb_size=None,
    out_emb_size=None
)

class VQModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        loss,
        transformable_data_module,
        n_codebook_embeddings,
        codebook_embedding_size,
        z_channels=256,
        image_key="data",
        monitor=None,
        remap=None,
        sane_index_shape=False,  
        reconstructions_out_base_path = "reconstructions/",
        checkpoint_every_val_epochs = 1,
        learning_rate=4.5e-6,
        caption_key="caption",
        embedding_provider=None
    ):
        """
        VQGAN model for training and creating reconstructions.

        :param encoder: Encoder for the model.
        :param decoder: Decoder for the model.
        :param loss: Loss function for the model. This should be a VQLPIPSWithDiscriminator, defaults to VQLPIPSWithDiscriminator(disc_start=10000)
        :param transformable_data_module: Data module for the model.
        :param n_codebook_embeddings: Number of codebook embeddings.
        :param codebook_embedding_size: Size of the codebook embeddings. 
        :param z_channels: 
        :param image_key: Key for the image in the batch, defaults to "data"
        :param monitor: Which value to monitor, defaults to None
        :param reconstructions_out_base_path: Base path for the reconstructions, defaults to "reconstructions/"
        :param checkpoint_every_val_epochs: Create a checkpoint every x validation epochs if the validation loss is lower than the previous checkpoint, defaults to 1
        :param learning_rate: Learning rate for the model, defaults to 4.5e-6
        :param caption_key: Key for the caption in the batch, defaults to "caption"
        :param embedding_provider: Embedding provider for the model, defaults to None
        """    
        super().__init__()
        self.image_key = image_key
        # Create the encoder and decoder if they are not provided
        self.encoder = encoder if encoder is not None else Encoder(z_channels=z_channels, **DEFAULT_ENCODER_ARGS)
        self.decoder = decoder if decoder is not None else Decoder(z_channels=z_channels, **DEFAULT_DECODER_ARGS)
        self.transformable_data_module = transformable_data_module
        # Create the loss if it is not provided
        self.loss = loss if loss is not None else VQLPIPSWithDiscriminator(disc_start=10000)
        self.learning_rate = learning_rate
        # Create the vector quantizer
        self.quantize = VectorQuantizer(
            n_codebook_embeddings, 
            codebook_embedding_size, 
            beta=0.25,
            remap=remap, 
            sane_index_shape=sane_index_shape
        )
        self.quant_conv = torch.nn.Conv2d(z_channels, codebook_embedding_size, 1)
        self.post_quant_conv = torch.nn.Conv2d(codebook_embedding_size, z_channels, 1)
        self.image_key = image_key
        self.reconstructions_out_base_path = reconstructions_out_base_path
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.val_epoch = 0
        self.validation_step_outputs = []
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")

        self.caption_key = caption_key
        self.embedding_provider = embedding_provider

        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['loss', 'decoder', 'encoder', 'embedding_provider'])

    def encode(self, x, emb=None):
        """
        Encodes the input and quantizes it.

        :param x: Input to encode.
        :param emb: Embedding, defaults to None
        :return: Tuple of the encoded input, the embedding loss and the embedding info.
        """        
        h = self.encoder(x, emb=emb)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, emb=None, clamp=False):
        """
        Decodes the input.

        :param quant: Input to decode.
        :param emb: Embedding, defaults to None
        :param clamp: Whether to clamp the output to [-1, 1], defaults to False
        :return: Decoded input.
        """        
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, emb=emb)
        dec = dec.clamp(-1, 1) if clamp else dec
        return dec

    def decode_code(self, code_b, emb=None, clamp=False):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, emb=emb, clamp=clamp)
        return dec

    def forward(self, x, emb=None, clamp=False):
        """
        Forward pass of the model.

        :param x: Input to encode and decode.
        :param emb: Embedding, defaults to None
        :param clamp: Whether to clamp the output to [-1, 1], defaults to False
        :return: Tuple of the decoded input and the embedding loss.
        """        
        quant, diff, _ = self.encode(x, emb=emb)
        dec = self.decode(quant, emb=emb, clamp=clamp)
        return dec, diff

    def get_input(self, batch, k):
        """
        Gets the input from the batch given the image key.

        :param batch: Batch of data.
        :param k: Key to get the input from.
        :return: Tuple of the input, the captions and the embeddings.
        """        
        images = batch[k]
        x = self.transformable_data_module.transform_batch(images)
        captions = None if self.caption_key is None else batch[self.caption_key]
        embs = None if self.embedding_provider is None else self.embedding_provider.get_embedding(images, captions)
        return x.float(), captions, embs

    def training_step(self, batch, batch_idx):
        """
        A training step for the model.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        :return: Dictionary of loss values to log.
        """        
        batch_size = len(batch[self.image_key])
        x, _, embs = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, qloss = self(x, embs)
        opt_ae, opt_disc = self.optimizers() # pylint: disable=E0633

        # Autoencoder and loss with manual backward
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/rec_loss_prog", log_dict_ae["train/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        opt_ae.zero_grad()
        self.manual_backward(aeloss)

        opt_ae.step()

        # Discriminator and loss with manual backward
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        opt_disc.zero_grad()
        self.manual_backward(discloss)

        opt_disc.step()

        return {"loss": aeloss, "disc_loss": discloss}


    def validation_step(self, batch, batch_idx):
        """
        A validation step for the model.

        :param batch: Batch of data.
        :param batch_idx: Batch index.
        :return: Dictionary of loss values to log.
        """        
        batch_size = len(batch[self.image_key])
        # Get input
        x, captions, embs = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, qloss = self(x, embs)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # self.log_dict(log_dict_ae, batch_size=batch_size)
        self.log_dict(log_dict_disc, batch_size=batch_size)

        self.save_reconstructions(xrec, batch_idx, captions=captions, note="reconstructions")
        self.save_reconstructions(x, batch_idx, captions=captions, note="inputs")
      
        self.validation_step_outputs.append({"rec_loss": rec_loss, "disc_loss": discloss})

        return self.log_dict


    def on_validation_epoch_end(self):
        """
        Saves the model checkpoint if the validation loss is lower than the previous checkpoint and
        if checkpoint_every_val_epochs is reached. Also saves the latest checkpoint but overwrites the previous one.
        """        
        outs = self.validation_step_outputs
        
        values = [outs[i]["rec_loss"] for i in range(len(outs)) if "rec_loss" in outs[i]]
        avg_loss = sum(values) / len(values)
       
        self.log_dict({"val_rec_loss":avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_loss < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.reconstructions_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_loss
        
        path = f"{self.reconstructions_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 

    def save_reconstructions(self, reconstructions, batch_idx, captions=None, note=None):
        """
        Saves the reconstructions to the disk.

        :param reconstructions: Reconstructed images.
        :param batch_idx: Batch index.
        :param captions: Captions for the images, defaults to None
        :param note: Note to add to the folder name, defaults to None
        """        
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.reconstructions_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        reconstructions = reconstructions.clamp(-1, 1)
        reconstructions = self.transformable_data_module.reverse_transform_batch(reconstructions.detach().cpu())
            
        for image_id in range(len(reconstructions)):
            reconstructions[image_id].save(path_folder + f"img_{image_id}.png")
        
        if captions is not None:
            path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
            with open(path_cap, "w") as f:
                for cap in captions:
                    f.write(cap)
                    f.write("\n")


    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        :return: Tuple of optimizers.
        """        
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return (opt_ae, opt_disc)

    def get_last_layer(self):
        """
        Gets the last layer weights of the model.

        :return: Weights of the last layer.
        """        
        return self.decoder.conv_out.weight
