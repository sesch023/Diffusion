from enum import Enum
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/nn.py#L68
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AdaptivePseudo3DConv(nn.Module):
    # https://github.com/lucidrains/make-a-video-pytorch/blob/main/make_a_video_pytorch/make_a_video.py
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        temp_kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=True,
        is_temporal=True,
        base_conv2d_for_weight_init=None
    ):
        super().__init__()
        self._conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self._is_temporal = is_temporal
        if is_temporal:
            self._conv1d = nn.Conv1d(out_channels, out_channels, temp_kernel_size, stride, padding, bias=bias)
            nn.init.dirac_(self._conv1d.weight)
            nn.init.zeros_(self._conv1d.bias)

        if base_conv2d_for_weight_init is not None:
            self.seed_with_conv2d(base_conv2d_for_weight_init)
  
    def seed_with_conv2d(self, base_conv2d):
        self._conv2d.weight.copy_(base_conv2d.weight)
        self._conv2d.bias.copy_(base_conv2d.bias)


    def forward(self, x, temporal=True):
        b, c, *_, h, w = x.shape
        is_vid_data = x.ndim == 5
        temporal = temporal and is_vid_data
        if is_vid_data:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self._conv2d(x)
        if is_vid_data:
            x = rearrange(x, '(b t) c h w -> b c t h w', b=b)

        if not temporal or not self._is_temporal:
            return x

        x = rearrange(x, 'b c t h w -> (b h w) c t')
        x = self._conv1d(x)
        x = rearrange(x, '(b h w) c t -> b c t h w', h=h, w=w)
        return x


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


class AdaptiveSpatioTemporalSelfAttention(nn.Module):
    # Inspired by: https://github.com/lucidrains/make-a-video-pytorch/blob/main/make_a_video_pytorch/make_a_video.py#L275
    def __init__(self, 
        channels,
        mha_heads=4,
        mha_head_channels=None,
        torch_mha=True,
        is_temporal=True,
        base_spatial_attention_for_weight_init=None
    ):
        super().__init__()
        self.channels = channels
        self.torch_mha = torch_mha
        self.scale = 1/math.sqrt(2)
        self.is_temporal = is_temporal
        
        if mha_head_channels is None:
            self.mha_heads = mha_heads
        else:
            self.mha_heads = channels // mha_head_channels
        
        self.norm_in = nn.GroupNorm(32, channels)
        self.norm_out_spat = nn.GroupNorm(32, channels)
        self.norm_out_temp = nn.GroupNorm(32, channels)
        if self.torch_mha:
            self.spatial_att =  nn.MultiheadAttention(channels, self.mha_heads, batch_first=True)
            if is_temporal:
                self.temp_att =  nn.MultiheadAttention(channels, self.mha_heads, batch_first=True)
        else:     
            self.qkv = nn.Conv1d(self.channels, self.channels*3, 1)
            self.spatial_att = QKVAttention(self.mha_heads)
            if is_temporal:
                self.temp_att = QKVAttention(self.mha_heads)
            self.proj_out = zero_module(nn.Conv1d(self.channels, self.channels, 1))

        if base_spatial_attention_for_weight_init is not None:
            self.seed_with_spatial_attention(base_spatial_attention_for_weight_init)


    def seed_with_spatial_attention(self, base_spatial_attention):
        self.spatial_att.weight.copy_(base_spatial_attention.weight)
        self.spatial_att.bias.copy_(base_spatial_attention.bias)


    def forward(self, x, temporal=True):
        b, c, *_, h, w = x.shape
        is_vid_data = x.ndim == 5
        temporal = temporal and is_vid_data

        if is_vid_data:
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.norm_in(x)
        if self.torch_mha:
            h, _ = self.spatial_att(x, x, x)
        else:
            h = h.swapaxes(-2, -1)
            qkv = self.qkv(x)
            h = self.att(qkv)
            h = self.proj_out(h)
            h = h.swapaxes(-1, -2)
        x = self.norm_out_spat(self.scale*x + h)

        if is_vid_data:
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b = b, h = h, w = w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)

        if not temporal or not self._is_temporal:
            return x

        x = rearrange(x, 'b c t h w -> (b h w) t c')

        if self.torch_mha:
            h, _ = self.temp_att(x, x, x)
        else:
            h = h.swapaxes(-2, -1)
            qkv = self.qkv(x)
            h = self.att(qkv)
            h = self.proj_out(h)
            h = h.swapaxes(-1, -2)

        x = self.norm_out_temp(self.scale*x + h)
        x = rearrange(x, '(b h w) t c -> b c t h w', h = h, w = w)
        return x


class MultiParamSequential(nn.Sequential):
    def forward(self, x, *kwargs):
        for module in self._modules.values():
            x = module(x, *kwargs)
        return x

    
class ResBlockSampleMode(Enum):
    IDENTITY="IDENTITY",
    UPSAMPLE2X="UPSAMPLE2X",
    DOWNSAMPLE2X="DOWNSAMPLE2X"

    
class Upsample2X(nn.Module):
    def __init__(self, in_channels, out_channels = None, use_conv=False):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels or in_channels
        self._use_conv = use_conv
        if use_conv:
            self._conv = nn.Conv2d(self._in_channels, self._out_channels, padding=1, kernel_size=3)
        self._sample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w') if x.ndim == 5 else x
        if self._use_conv:
            x = self._conv(x)
        x = self._sample(x)
        return rearrange(x, '(b t) c h w -> b c t h w', b=b) if x.ndim == 5 else x


class Downsample2X(nn.Module):
    def __init__(self, in_channels, out_channels = None, use_conv=False):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels or in_channels
        self._use_conv = use_conv
        if use_conv:
            self._sample = nn.Conv2d(self._in_channels, self._out_channels, stride=2, padding=1, kernel_size=3)
        else:
            self._sample = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w') if x.ndim == 5 else x
        x = self._sample(x)
        return rearrange(x, '(b t) c h w -> b c t h w', b=b) if x.ndim == 5 else x


class SelfAttention(nn.Module):
    """
    https://github.com/epfml/text_to_image_generation/blob/main/guided_diffusion/unet.py#L361
    """
    def __init__(self, 
        channels,
        mha_heads=4,
        mha_head_channels=None,
        torch_mha=True):

        super().__init__()
        self.channels = channels
        self.torch_mha = torch_mha
        self.scale = 1/math.sqrt(2)
        
        if mha_head_channels is None:
            self.mha_heads = mha_heads
        else:
            self.mha_heads = channels // mha_head_channels
        
        self.norm_in = nn.GroupNorm(32, channels)
        self.norm_out = nn.GroupNorm(32, channels)
        if self.torch_mha:
            self.att =  nn.MultiheadAttention(channels, self.mha_heads, batch_first=True)
        else:     
            self.qkv = nn.Conv1d(self.channels, self.channels*3, 1)
            self.att = QKVAttention(self.mha_heads)
            self.proj_out = zero_module(nn.Conv1d(self.channels, self.channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = self.norm_in(x)
        if self.torch_mha:
            x_n = x.swapaxes(1, 2)
            h, _ = self.att(x_n, x_n, x_n)
            h = h.swapaxes(2, 1)
        else:
            qkv = self.qkv(x)
            h = self.att(qkv)
            h = self.proj_out(h)
        x = self.norm_out(self.scale*x + h)
        return x.reshape(b, c, *spatial)


class ResBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        sample_mode=ResBlockSampleMode.IDENTITY,
        emb_size=1024, 
        dropout=0.1,
        skip_con=True,
        use_scale_shift_norm=True,
        use_conv=True
    ):
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
            self._skip_conv = self.get_convolution(in_channels, out_channels)
        
        self._sample_mode = sample_mode
        if sample_mode == ResBlockSampleMode.UPSAMPLE2X:
            self._sample = Upsample2X(in_channels, in_channels, use_conv=use_conv)
            self._sample_skip = Upsample2X(in_channels, in_channels, use_conv=use_conv)
        elif sample_mode == ResBlockSampleMode.DOWNSAMPLE2X:
            self._sample = Downsample2X(in_channels, in_channels, use_conv=use_conv)
            self._sample_skip = Downsample2X(in_channels, in_channels, use_conv=use_conv)
        else:
            self._sample = self._sample_skip = nn.Identity()         
        
        self._pre_concat_input = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU()
        )
        self._pre_concat_conv = self.get_convolution(in_channels, out_channels)
        
        self._aft_concat_norm = nn.GroupNorm(32, out_channels)
        self._aft_concat_input = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self._aft_concat_conv = zero_module(self.get_convolution(out_channels, out_channels))


    def get_convolution(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=3)


    def forward_conv(self, conv, x):
        return conv(x)


    def forward(self, x, emb=None):
        mid_out = self._pre_concat_input(x)
        mid_out = self._sample(mid_out)
        mid_out = self.forward_conv(self._pre_concat_conv, mid_out)
        
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
        
        out = self.forward_conv(self._aft_concat_conv, out)

        if self._skip_con:
            return self.forward_conv(self._skip_conv, self._sample_skip(x)) + out
        else:
            return out


class AdaptiveSpatioTemporalResBlock(ResBlock):
    def __init__(self,
        in_channels,
        out_channels,
        sample_mode=ResBlockSampleMode.IDENTITY,
        emb_size=1024, 
        dropout=0.1,
        skip_con=True,
        use_scale_shift_norm=True,
        use_conv=True
    ):
        super().__init__(in_channels, out_channels, sample_mode, emb_size, dropout, skip_con, use_scale_shift_norm, use_conv)
        self.forward_temporal = True

    def get_convolution(self, in_channels, out_channels):
        return AdaptivePseudo3DConv(in_channels, out_channels, kernel_size=3, temp_kernel_size=3, padding=1, bias=True, is_temporal=True)

    def forward_conv(self, conv, x):
        return conv(x, temporal=self.forward_temporal)

    def forward(self, x, emb=None, temporal=True):
        self.forward_temporal = temporal
        return super().forward(x, emb)

    
class SelfAttentionResBlock(ResBlock):
    def __init__(self, 
        in_channels, 
        out_channels, 
        mha_heads=4, 
        mha_head_channels=None, 
        emb_size=1024, 
        dropout=0.1, 
        skip_con=True,
        use_scale_shift_norm=True
    ):
        super().__init__(in_channels, out_channels, ResBlockSampleMode.IDENTITY, emb_size, dropout, skip_con, use_scale_shift_norm)
        self.self_at = SelfAttention(out_channels, mha_heads, mha_head_channels)
        
    def forward(self, x, emb=None):
        x = super().forward(x, emb)
        return self.self_at(x)
    
    
class AdaptiveSpatioTemporalSelfAttentionResBlock(AdaptiveSpatioTemporalResBlock):
    def __init__(self, 
        in_channels, 
        out_channels, 
        mha_heads=4, 
        mha_head_channels=None, 
        emb_size=1024, 
        dropout=0.1, 
        skip_con=True,
        use_scale_shift_norm=True
    ):
        super().__init__(in_channels, out_channels, ResBlockSampleMode.IDENTITY, emb_size, dropout, skip_con, use_scale_shift_norm)
        self.self_at = AdaptiveSpatioTemporalSelfAttention(out_channels, mha_heads, mha_head_channels, is_temporal=True)

    def forward(self, x, emb=None, temporal=True):
        x = super().forward(x, emb, temporal=temporal)
        return self.self_at(x, temporal=temporal)


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