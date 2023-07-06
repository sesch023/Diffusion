from enum import Enum
import math
import torch
import torch.nn as nn
import numpy as np


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/nn.py#L68
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MultiParamSequential(nn.Sequential):
    def forward(self, x, *kwargs):
        for module in self._modules.values():
            x = module(x, *kwargs)
        return x

    
class ResBlockSampleMode(Enum):
    IDENTITY="IDENTITY",
    UPSAMPLE2X="UPSAMPLE2X",
    DOWNSAMPLE2X="DOWNSAMPLE2X"

    
class ResBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        sample_mode=ResBlockSampleMode.IDENTITY,
        emb_size=1024, 
        dropout=0.1,
        skip_con=True,
        use_scale_shift_norm=True):

        super().__init__()
        self._emb_size = emb_size
        self._use_scale_shift_norm  = use_scale_shift_norm 
        if self._emb_size is not None:
            self._emb_seq = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_size, 2 * out_channels if use_scale_shift_norm else out_channels)
            )
        
        self._skip_con = skip_con
        if skip_con:
            self._skip_conv = nn.Conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=3)    
        
        self._sample_mode = sample_mode
        if sample_mode == ResBlockSampleMode.UPSAMPLE2X:
            self._sample = nn.Upsample(scale_factor=2, mode="nearest")
            self._sample_skip = nn.Upsample(scale_factor=2, mode="nearest")
        elif sample_mode == ResBlockSampleMode.DOWNSAMPLE2X:
            self._sample = nn.AvgPool2d(2, stride=2)
            self._sample_skip = nn.AvgPool2d(2, stride=2)
        else:
            self._sample = self._sample_skip = nn.Identity()         
        
        self._pre_concat_input = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU()
        )
        self._pre_concat_conv = nn.Conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=3)
        
        self._aft_concat_norm = nn.GroupNorm(32, out_channels)
        self._aft_concat_input = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, stride=1, padding=1, kernel_size=3))
        )
    
    def forward(self, x, emb=None):
        mid_out = self._pre_concat_input(x)
        mid_out = self._sample(mid_out)
        mid_out = self._pre_concat_conv(mid_out)
        
        if self._emb_size is not None:
            emb_out = self._emb_seq(emb)
            while len(emb_out.shape) < len(mid_out.shape):
                emb_out = emb_out[..., None]
            
            if self._use_scale_shift_norm:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                out = self._aft_concat_norm(mid_out) * (1 + scale) + shift
                out = self._aft_concat_input(out)
            else:
                mid_out += emb_out
                out = self._aft_concat_input(self._aft_concat_norm(mid_out))
        else:
            out = self._aft_concat_input(self._aft_concat_norm(mid_out))
        
        if self._skip_con:
            return self._skip_conv(self._sample_skip(x)) + out
        else:
            return out

        
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/unet.py#L361
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    
class SelfAttention(nn.Module):
    """
    https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/unet.py#L361
    """
    def __init__(self, 
        channels,
        mha_heads=4,
        mha_head_channels=None,
        torch_mha=False):

        super().__init__()
        self.channels = channels
        self.torch_mha = torch_mha
        self.scale = 1/math.sqrt(2)
        
        if mha_head_channels is None:
            self.mha_heads = mha_heads
        else:
            self.mha_heads = channels // mha_head_channels
        
        self.norm = nn.GroupNorm(32, channels)
        if self.torch_mha:
            self.att =  nn.MultiheadAttention(channels, self.mha_heads, batch_first=True)
        else:     
            self.qkv = nn.Conv1d(self.channels, self.channels*3, 1)
            self.att = QKVAttention(self.mha_heads)
        self.proj_out = zero_module(nn.Conv1d(self.channels, self.channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        if self.torch_mha:
            x_n = self.norm(x).swapaxes(1, 2)
            h, _ = self.att(x_n, x_n, x_n)
            h = h.swapaxes(2, 1)
        else:
            qkv = self.qkv(self.norm(x))
            h = self.att(qkv)
            h = self.proj_out(h)
        x = nn.functional.group_norm(self.scale*x + h, 32)
        return x.reshape(b, c, *spatial)

    
class SelfAttentionResBlock(ResBlock):
    def __init__(self, 
        in_channels, 
        out_channels, 
        mha_heads=4, 
        mha_head_channels=None, 
        emb_size=1024, 
        dropout=0.1, 
        skip_con=True,
        use_scale_shift_norm=True):

        super().__init__(in_channels, out_channels, ResBlockSampleMode.IDENTITY, emb_size, dropout, skip_con, use_scale_shift_norm)
        self.self_at = SelfAttention(out_channels, mha_heads, mha_head_channels)
        
    def forward(self, x, emb=None):
        x = super().forward(x, emb)
        return self.self_at(x)
    
    
class VLBDiffusionLoss():
    # https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/losses.py#L12
    @staticmethod
    def kl_divergence(mean1, logvar1, mean2, logvar2):
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]
        
        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
   
    @staticmethod
    def approx_standard_normal_cdf(x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    @staticmethod
    def discretized_gaussian_log_likelihood(x, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = VLBDiffusionLoss.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = VLBDiffusionLoss.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return log_probs