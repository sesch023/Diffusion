import torch 
import torch.nn as nn
from einops import rearrange
import functools
import numpy as np

from DiffusionModules.Modules import *

"""
This Module was adapted from:
https://github.com/CompVis/taming-transformers
"""

class Encoder(nn.Module):
    def __init__(self, 
            ch,
            in_channels,
            resolution, 
            z_channels,
            num_res_blocks,
            attn_resolutions,
            ch_mult=(1,2,4,8), 
            dropout=0.0, 
            double_z=False,
            emb_size=None,
            out_emb_size=1024
        ):
        """
        Encoder for the VQGAN model.

        :param ch: Number of base channels.
        :param in_channels: Number of input channels.
        :param resolution: Resolution of the input.
        :param z_channels: Number of channels in the latent space.
        :param num_res_blocks: Number of residual blocks per resolution.
        :param attn_resolutions: Resolutions to use attention on.
        :param ch_mult: Channel multipliers for each resolution, defaults to (1,2,4,8)
        :param dropout: Dropout probability, defaults to 0.0
        :param double_z: Whether to double the number of channels in the latent space, defaults to False
        :param emb_size: Size of the embeddings, defaults to None
        :param out_emb_size: Output size of the linear embedding layers, defaults to 1024
        """        
        super().__init__()
        self.ch = ch
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size if out_emb_size is not None and emb_size is not None else emb_size
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # This is inspired by our diffusion model
        if emb_size is not None:
            self.emb_seq = nn.Sequential(
                nn.Linear(self.emb_size, self.out_emb_size),
                nn.SiLU(),
                nn.Linear(self.out_emb_size, self.out_emb_size)
            )

        # Input block
        self.conv_in = nn.Conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Construct the down blocks
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = []
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks-1):   
                if curr_res in attn_resolutions:
                    block.append(SelfAttentionResBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        emb_size=self.out_emb_size,
                        dropout=dropout,
                        torch_mha=True
                    ))
                else:
                    block.append(ResBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        emb_size=self.out_emb_size,
                        dropout=dropout
                    ))
                block_in = block_out
            if i_level != self.num_resolutions-1:
                block.append(ResBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    emb_size=self.out_emb_size,
                    dropout=dropout,
                    sample_mode=ResBlockSampleMode.DOWNSAMPLE2X
                ))
                curr_res = curr_res // 2
            else: 
                block.append(ResBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    emb_size=self.out_emb_size,
                    dropout=dropout
                ))
            self.down.append(MultiParamSequential(*block))

        # Output block
        self.out = MultiParamSequential(
            SelfAttentionResBlock(
                in_channels=block_in,
                out_channels=block_in,
                emb_size=self.out_emb_size,
                dropout=dropout,
                torch_mha=True
            ),
            ResBlock(
                in_channels=block_in,
                out_channels=block_in,
                emb_size=self.out_emb_size,
                dropout=dropout
            )
        )
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # Output conv
        self.conv_out = nn.Conv2d(
            block_in,
            2*z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.out_non_res = nn.Sequential(
            self.norm_out,
            nn.SiLU(),
            self.conv_out
        )


    def forward(self, x, emb=None):
        """
        Forward pass of the encoder.

        :param x: Input tensor.
        :param emb: Embedding tensor, defaults to None
        :return: Output tensor.
        """        
        if self.emb_size is not None:
            emb = self.emb_seq(emb) if emb is not None else torch.zeros_like(emb)

        h = self.conv_in(x)
        for block in self.down:
            h = block(h, emb)

        h = self.out(h, emb)
        return self.out_non_res(h)


class Decoder(nn.Module):
    def __init__(self, 
            ch, 
            out_channels, 
            attn_resolutions,
            num_res_blocks,
            resolution, 
            z_channels,
            ch_mult=(1,2,4,8), 
            dropout=0.0, 
            give_pre_end=False,
            emb_size=None,
            out_emb_size=1024
        ):
        """
        Decoder for the VQGAN model.

        :param ch: Number of base channels at the top.
        :param out_channels: Number of output channels.
        :param attn_resolutions: Resolutions to use attention on.
        :param num_res_blocks: Number of residual blocks per resolution.
        :param resolution: Resolution of the input.
        :param z_channels: Number of channels in the latent space. This is the input of the decoder.
        :param ch_mult: Channel multipliers for each resolution, defaults to (1,2,4,8)
        :param dropout: Dropout probability, defaults to 0.0
        :param give_pre_end: No final conv and norm, defaults to False
        :param emb_size: Size of the embeddings, defaults to None
        :param out_emb_size: Output size of the linear embedding layers, defaults to 1024
        """        
        super().__init__()
        self.ch = ch
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size if out_emb_size is not None and emb_size is not None else emb_size
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.out_channels = out_channels
            
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        # This is inspired by our diffusion model
        if emb_size is not None:
            self.emb_seq = nn.Sequential(
                nn.Linear(self.emb_size, self.out_emb_size),
                nn.SiLU(),
                nn.Linear(self.out_emb_size, self.out_emb_size)
            )

        # Input block
        self.in_conv = nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.in_seq = MultiParamSequential(
            SelfAttentionResBlock(
                in_channels=block_in,
                out_channels=block_in,
                emb_size=self.out_emb_size,
                dropout=dropout,
                torch_mha=True
            ),
            ResBlock(
                in_channels=block_in,
                out_channels=block_in,
                emb_size=self.out_emb_size,
                dropout=dropout
            )
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks-1):
                if curr_res in attn_resolutions:
                    block.append(SelfAttentionResBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        emb_size=self.out_emb_size,
                        dropout=dropout,
                        torch_mha=True
                    ))
                else:
                    block.append(ResBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        emb_size=self.out_emb_size,
                        dropout=dropout
                    ))
                block_in = block_out
            if i_level > 0:
                block.append(ResBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    emb_size=self.out_emb_size,
                    dropout=dropout,
                    sample_mode=ResBlockSampleMode.UPSAMPLE2X
                ))
                curr_res = curr_res * 2
            else: 
                block.append(ResBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    emb_size=self.out_emb_size,
                    dropout=dropout
                ))
            self.up.append(MultiParamSequential(*block))

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # Output conv
        self.conv_out = nn.Conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.out_non_res = nn.Sequential(
            self.norm_out,
            nn.SiLU(),
            self.conv_out
        )

    def forward(self, z, emb=None):
        """
        Forward pass of the decoder.

        :param z: Input tensor of the codebook embeddings.
        :param emb: Embedding tensor, defaults to None
        :return: Output tensor.
        """        
        if self.emb_size is not None:
            emb = self.emb_seq(emb) if emb is not None else torch.zeros_like(emb)

        # Input block
        h = self.in_conv(z)
        h = self.in_seq(h, emb)

        # Upsampling blocks
        for block in self.up:
            h = block(h, emb)

        # Do not apply final conv and norm
        if self.give_pre_end:
            return h

        h = self.out_non_res(h)
        return h

def weights_init(m):
    """
    Initializes the weights of convolutional and batch norm layers.

    :param m: Module to initialize.
    """    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """
        Construct a PatchGAN discriminator.

        Quellen:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py

        :param input_nc: Number of channels in input images, defaults to 3
        :param ndf: Number of filters in the first conv layer, defaults to 64
        :param n_layers: Number of conv layers in the discriminator, defaults to 3
        """          
        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class ResidualEmbeddingConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=64, 
        number_hidden_layers=3,
        emb_size=None,
        out_emb_size=1024,
        patch_based=True,
        # Only relevant if patch_based=False
        input_resolution=256
    ):
        """
        Residual embedding conditional discriminator for the VQGAN model.
        This is a conditional discriminator that takes the embedding of the image as an additional input.
        It was inspired by our resblock architecture. Since the NLayersDiscriminator lead to good results,
        we did not use this discriminator.

        :param in_channels: Number of input channels.
        :param base_channels: Number of base channels, defaults to 64
        :param number_down_layers: Number of down layers, defaults to 3
        :param emb_size: Size of the embeddings, defaults to None
        :param out_emb_size: Output size of the linear embedding layers, defaults to 1024
        :param patch_based: Whether to output patch based values or a single value, defaults to True
        :param input_resolution: Resolution of the input, defaults to 256
        """        
        super().__init__()
        self.emb_size = emb_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.number_down_layers = number_down_layers
        self.patch_based = patch_based
        self.input_resolution = input_resolution
        self.out_emb_size = out_emb_size if out_emb_size is not None and emb_size is not None else emb_size
        # This is inspired by our diffusion model
        if emb_size is not None:
            self.emb_seq = nn.Sequential(
                nn.Linear(self.emb_size, self.out_emb_size),
                nn.SiLU(),
                nn.Linear(self.out_emb_size, self.out_emb_size)
            )

        self.seq_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        blocks = []
        channel_mult = 1
        channel_mult_prev = 1
        curr_res = self.input_resolution
        for i in range(self.number_down_layers):
            ds_mode = ResBlockSampleMode.DOWNSAMPLE2X if (i < self.number_down_layers-1 and curr_res >= 4) else ResBlockSampleMode.NO_DOWNSAMPLE
            channel_mult_prev = channel_mult
            channel_mult = min(2**(i+1), 8)
            blocks.append(
                ResBlock(
                    in_channels=self.base_channels*channel_mult_prev,
                    out_channels=self.base_channels*channel_mult,
                    emb_size=self.out_emb_size,
                    dropout=0.0,
                    sample_mode=ds_mode
                )
            )
            curr_res = curr_res // 2 if ds_mode == ResBlockSampleMode.DOWNSAMPLE2X else curr_res


        self.seq_hidden = MultiParamSequential(*blocks)
        out_blocks = []
        out_blocks.append(
            nn.Conv2d(self.base_channels*channel_mult, 1, kernel_size=3, stride=1, padding=1)
        )

        if not self.patch_based:
            out_blocks.append(nn.Flatten())
            out_blocks.append(nn.Linear(curr_res*curr_res, 1))

        self.seq_out = nn.Sequential(*out_blocks)


    def forward(self, x, emb=None):
        """
        Forward pass of the discriminator.

        :param x: Input tensor.
        :param emb: Embedding tensor, defaults to None
        :return: A value for each patch or a single value.
        """        
        if self.emb_size is not None:
            emb = self.emb_seq(emb) if emb is not None else torch.zeros_like(emb)
        
        h = self.seq_in(x)
        h = self.seq_hidden(h, emb)
        h = self.seq_out(h)

        return h


class VectorQuantizer(nn.Module):
    def __init__(
        self, 
        n_codebook_embeddings, 
        codebook_embedding_size, 
        beta, 
        remap=None, 
        unknown_index="random",
        sane_index_shape=False, 
        legacy=False
    ):
        """
        Vector quantizer for the VQGAN model. It quantizes the input to the nearest codebook embedding.
        This was taken from:
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py#L6

        Original description:
        Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
        avoids costly matrix multiplications and allows for post-hoc remapping of indices.

        # NOTE: due to a bug the beta term was applied to the wrong term. for
        # backwards compatibility we use the buggy version by default, but you can
        # specify legacy=False to fix it.

        :param n_codebook_embeddings: Number of codebook embeddings.
        :param codebook_embedding_size: Size of the codebook embeddings. This equals the z_channels of the encoder and decoder.
        :param beta: Beta parameter for the loss. This is the weight of the commitment loss.
        :param legacy: Whether to use the legacy version of the loss, defaults to False
        """        
        super().__init__()
        self.n_codebook_embeddings = n_codebook_embeddings
        self.codebook_embedding_size = codebook_embedding_size
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_codebook_embeddings, self.codebook_embedding_size)
        self.embedding.weight.data.uniform_(-1.0 / self.n_codebook_embeddings, 1.0 / self.n_codebook_embeddings)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_codebook_embeddings} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_codebook_embeddings

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        """
        Remaps the indices to the used indices.
        Since remap was not used, this was never tested.

        :param inds: Indices to remap.
        :return: Remapped indices.
        """        
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """
        Unmaps the indices to the original indices.
        Since remap was not used, this was never tested.

        :param inds: Indices to unmap.
        :return: Unmapped indices.
        """        
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        """
        Forward pass of the vector quantizer. Also calculates the quantization loss.

        :param z: Non quantized input tensor.
        :return: _description_
        """        
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.codebook_embedding_size)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        Gets the codebook entry for the given indices.

        :param indices: Indices to get the codebook entry for.
        :param shape: Shape of the output if not None.
        :return: Codebook entry.
        """        
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q