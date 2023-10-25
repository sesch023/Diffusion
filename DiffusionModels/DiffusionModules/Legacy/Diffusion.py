import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torchinfo import summary
import math
import clip
from abc import ABC, abstractmethod
from DiffusionModules.Legacy.DiffusionModules import *
from tqdm import tqdm
from enum import Enum
from torchviz import make_dot
import wandb

# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
class ExponentialMovingAverage():
    def __init__(self, beta, step_start_ema=2000):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.step_start_ema = step_start_ema

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        

class UpscalerUNet(nn.Module):
    def __init__(self, i_emb_size=512, t_emb_size=256, mid_emb_size=1024, out_channels=6, device=None):
        super().__init__()
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._i_emb_size = i_emb_size
        self._t_emb_size = t_emb_size
        self._mid_emb_size = mid_emb_size
        
        if t_emb_size is not None:
            self._time_emb_seq = nn.Sequential(
                nn.Linear(t_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )
        
        if i_emb_size is not None:
            self._img_emb_seq = nn.Sequential(
                nn.Linear(i_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )
        
        self._in_conv = nn.Conv2d(6, 256, stride=1, padding=1, kernel_size=3)
        self._down_block_1 = MultiParamSequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self._down_128 = ResBlock(256, 256, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_2 = MultiParamSequential(
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self._down_64 = ResBlock(256, 256, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_3 = MultiParamSequential(
            ResBlock(256, 512),
            ResBlock(512, 512)
        )
        self._down_32 = ResBlock(512, 512, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_4 = MultiParamSequential(
            SelfAttentionResBlock(512, 512, mha_heads=4),
            SelfAttentionResBlock(512, 512, mha_heads=4)
        )
        self._down_16 = ResBlock(512, 512, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_5 = MultiParamSequential(
            SelfAttentionResBlock(512, 1024, mha_heads=4),
            SelfAttentionResBlock(1024, 1024, mha_heads=4)
        )
        self._down_8 = ResBlock(1024, 1024, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_6 = MultiParamSequential(
            SelfAttentionResBlock(1024, 1024, mha_heads=4),
            SelfAttentionResBlock(1024, 1024,mha_heads=4)
        )
    
        self._bot_down_mid = MultiParamSequential(
            SelfAttentionResBlock(1024 , 1024, mha_heads=4),
            SelfAttentionResBlock(1024, 1024, mha_heads=4)
        )
        self._bot_mid = MultiParamSequential(
            SelfAttentionResBlock(1024, 1024, mha_heads=4),
            ResBlock(1024, 1024)
        )
        self._bot_up_mid = MultiParamSequential(
            SelfAttentionResBlock(2*1024, 1024, mha_heads=4),
            SelfAttentionResBlock(1024, 1024, mha_heads=4)
        )
        self._up_16 = ResBlock(1024, 1024, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_1 = MultiParamSequential(
            SelfAttentionResBlock(2*1024, 1024, mha_heads=4),
            SelfAttentionResBlock(1024, 1024, mha_heads=4)
        )
        self._up_32 = ResBlock(1024, 512, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_2 = MultiParamSequential(
            SelfAttentionResBlock(2*512, 512, mha_heads=4),
            SelfAttentionResBlock(512, 512, mha_heads=4)
        )
        self._up_64 = ResBlock(512, 512, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_3 = MultiParamSequential(
            ResBlock(2*512, 512),
            ResBlock(512, 512)
        )
        self._up_128 = ResBlock(512, 256, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_4 = MultiParamSequential(
            ResBlock(2*256, 256),
            ResBlock(256, 256)
        )
        self._up_256 = ResBlock(256, 256, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_5 = MultiParamSequential(
            ResBlock(2*256, 256),
            ResBlock(256, 256)
        )
        
        self._out_conv = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, out_channels, stride=1, padding=1, kernel_size=3))
        )
        
    
    
    def forward(self, x, t_emb=None, i_emb=None):
        if self._i_emb_size is None and self._t_emb_size is None:
            emb = None
        else:
            zeros = torch.zeros(self._mid_emb_size).to(self._device)
            i_emb = zeros if i_emb is None else self._img_emb_seq(i_emb)
            t_emb = zeros if t_emb is None else self._time_emb_seq(t_emb)       
            
            emb = i_emb + t_emb
        
        xd1 = self._in_conv(x)
        xd1 = self._down_block_1(xd1, emb)
        xd2 = self._down_128(xd1, emb)
        xd2 = self._down_block_2(xd2, emb)
        xd3 = self._down_64(xd2, emb)
        xd3 = self._down_block_3(xd3, emb)
        xd4 = self._down_32(xd3, emb)
        xd4 = self._down_block_4(xd4, emb)
        xd5 = self._down_16(xd4, emb)
        xd5 = self._down_block_5(xd5, emb)
        xd6 = self._down_8(xd5, emb)
        xd6 = self._bot_down_mid(xd6, emb)
        
        x = self._bot_mid(xd6, emb)
        
        x = torch.cat((xd6, x), dim=1)
        x = self._bot_up_mid(x, emb)
        x = self._up_16(x, emb)
        x = torch.cat((xd5, x), dim=1)
        x = self._up_block_1(x, emb)
        x = self._up_32(x, emb)
        x = torch.cat((xd4, x), dim=1)
        x = self._up_block_2(x, emb)
        x = self._up_64(x, emb)
        x = torch.cat((xd3, x), dim=1)
        x = self._up_block_3(x, emb)
        x = self._up_128(x, emb)
        x = torch.cat((xd2, x), dim=1)
        x = self._up_block_4(x, emb)
        x = self._up_256(x, emb)
        x = torch.cat((xd1, x), dim=1)
        x = self._up_block_5(x, emb)
        
        x = self._out_conv(x)
        
        return x
        
class BasicUNet(nn.Module):
    def __init__(self, i_emb_size=512, t_emb_size=256, mid_emb_size=1024, out_channels=6, device=None):
        super().__init__()
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._i_emb_size = i_emb_size
        self._t_emb_size = t_emb_size
        self._mid_emb_size = mid_emb_size
        
        if t_emb_size is not None:
            self._time_emb_seq = nn.Sequential(
                nn.Linear(t_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )
        
        if i_emb_size is not None:
            self._img_emb_seq = nn.Sequential(
                nn.Linear(i_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )
        
        self._in_conv = nn.Conv2d(3, 256, stride=1, padding=1, kernel_size=3)
        self._down_block_1 = MultiParamSequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self._down_32 = ResBlock(256, 256, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_2 = MultiParamSequential(
            SelfAttentionResBlock(256, 512, mha_head_channels=64),
            SelfAttentionResBlock(512, 512, mha_head_channels=64),
            SelfAttentionResBlock(512, 512, mha_head_channels=64)
        )
        self._down_16 = ResBlock(512, 512, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._down_block_3 = MultiParamSequential(
            SelfAttentionResBlock(512, 768, mha_head_channels=64),
            SelfAttentionResBlock(768, 768, mha_head_channels=64),
            SelfAttentionResBlock(768, 768, mha_head_channels=64)
        )
        self._down_8 = ResBlock(768, 768, sample_mode=ResBlockSampleMode.DOWNSAMPLE2X)
        self._bot_down = MultiParamSequential(
            SelfAttentionResBlock(768 , 1024, mha_head_channels=64),
            SelfAttentionResBlock(1024, 1024, mha_head_channels=64),
            SelfAttentionResBlock(1024, 1024, mha_head_channels=64)
        )
        self._bot_mid = MultiParamSequential(
            SelfAttentionResBlock(1024, 1024, mha_head_channels=64),
            ResBlock(1024, 1024)
        )
        self._bot_up = MultiParamSequential(
            SelfAttentionResBlock(2*1024, 1024, mha_head_channels=64),
            SelfAttentionResBlock(1024, 1024, mha_head_channels=64),
            SelfAttentionResBlock(1024, 1024, mha_head_channels=64)
        )
        self._up_16 = ResBlock(1024, 768, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_1 = MultiParamSequential(
            SelfAttentionResBlock(2*768, 768, mha_head_channels=64),
            SelfAttentionResBlock(768, 768, mha_head_channels=64),
            SelfAttentionResBlock(768, 768, mha_head_channels=64)
        )
        self._up_32 = ResBlock(768, 512, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_2 = MultiParamSequential(
            SelfAttentionResBlock(2*512, 512, mha_head_channels=64),
            SelfAttentionResBlock(512, 512, mha_head_channels=64),
            SelfAttentionResBlock(512, 512, mha_head_channels=64)
        )
        self._up_64 = ResBlock(512, 256, sample_mode=ResBlockSampleMode.UPSAMPLE2X)
        self._up_block_3 = MultiParamSequential(
            ResBlock(2*256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self._out_conv = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, out_channels, stride=1, padding=1, kernel_size=3))
        )
        
    
    
    def forward(self, x, t_emb=None, i_emb=None):
        if self._i_emb_size is None and self._t_emb_size is None:
            emb = None
        else:
            zeros = torch.zeros(self._mid_emb_size).to(self._device)
            i_emb = zeros if i_emb is None else self._img_emb_seq(i_emb)
            t_emb = zeros if t_emb is None else self._time_emb_seq(t_emb)       
            
            emb = i_emb + t_emb
        
        xd1 = self._in_conv(x)
        xd1 = self._down_block_1(xd1, emb)
        xd2 = self._down_32(xd1, emb)
        xd2 = self._down_block_2(xd2, emb)
        xd3 = self._down_16(xd2, emb)
        xd3 = self._down_block_3(xd3, emb)
        xd4 = self._down_8(xd3, emb)
        xd4 = self._bot_down(xd4, emb)
        
        x = self._bot_mid(xd4, emb)
        
        x = torch.cat((xd4, x), dim=1)
        x = self._bot_up(x, emb)
        x = self._up_16(x, emb)
        x = torch.cat((xd3, x), dim=1)
        x = self._up_block_1(x, emb)
        x = self._up_32(x, emb)
        x = torch.cat((xd2, x), dim=1)
        x = self._up_block_2(x, emb)
        x = self._up_64(x, emb)
        x = torch.cat((xd1, x), dim=1)
        x = self._up_block_3(x, emb)
        
        x = self._out_conv(x)
        
        return x
    
class ClipTools():
    def __init__(self, clip_model="ViT-B/32", device=None):
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._clip_model, self._clip_preprocess = clip.load(clip_model, device=device)
        self._clip_model.eval()
    
    def get_clip_emb_size(self):
        return self._clip_model.visual.output_dim 
    
    def get_clip_emb_images(self, images):
        images = torch.stack([self._clip_preprocess(i) for i in images])
        return self._clip_model.encode_image(images.to(self._device)).float()
    
    def get_clip_emb_text(self, texts):
        return self._clip_model.encode_text(clip.tokenize(texts, truncate = True).to(self._device)).float()
        
    
class NoiseScheduler(ABC):
    @abstractmethod
    def get_noise_schedule(self, timesteps):
        pass
    
class CosineScheduler(NoiseScheduler):
    def __init__(self, offset=0.008):
        super().__init__()
        self._offset = offset
        
    def get_noise_schedule(self, timesteps):
        # https://huggingface.co/blog/annotated-diffusion
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + self._offset) / (1 + self._offset) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
class LinearScheduler(NoiseScheduler):
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self._beta_start = beta_start
        self._beta_end = beta_end
        
    def get_noise_schedule(self, timesteps):
        return torch.linspace(self._beta_start, self._beta_end, timesteps)
    
class VarianceMode(Enum):
    NOT_LEARNED = "NOT_LEARNED",
    LEARNED = "LEARNED",
    LEARNED_SCALE = "LEARNED_SCALE"
    
class DiffusionTools():
    # TODO: Über Steps nachdenken
    def __init__(self, t_enc_size=256, steps=1000, noise_scheduler=None, img_size=64, img_channels=3, variance_mode=VarianceMode.LEARNED_SCALE, variance_lambda=0.001, clamp_x_start_in_sample=True, device=None):
        assert t_enc_size % 2 == 0, "Size of Timestep embedding needs to be even"
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._t_enc_size = t_enc_size
        self._img_size = img_size
        self._img_channels = img_channels
        self._variance_mode = variance_mode
        self._variance_lambda = variance_lambda
        self._steps = steps
        self._noise_scheduler = CosineScheduler() if not isinstance(noise_scheduler, NoiseScheduler) else noise_scheduler
        self._schedule = self._noise_scheduler.get_noise_schedule(steps).to(device)
        self._alphas = 1.0 - self._schedule
        self._clamp_x_start_in_sample = clamp_x_start_in_sample
        self._alphas_cum = torch.cumprod(self._alphas, dim=0)
        self._alphas_cum_prev = torch.cat((torch.Tensor([1.0]).to(self._device), self._alphas_cum[:-1]))
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cum)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cum)
        self._posterior_variance  = self._schedule * (1. - self._alphas_cum_prev) / (1. - self._alphas_cum)
        self._posterior_log_variance_clipped = torch.log(torch.cat((self._posterior_variance[1:2], self._posterior_variance[1:])))
        self._posterior_mean_coef_1 = self._schedule * torch.sqrt(self._alphas_cum_prev) / (1.0 - self._alphas_cum)
        self._posterior_mean_coef_2 = (1 - self._alphas_cum_prev) * torch.sqrt(self._alphas) / (1.0 - self._alphas_cum) 
        self._sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self._alphas_cum)
        self._sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self._alphas_cum - 1)
        
        self._step = 0
    
    def get_timestep_encoding(self, t):
        enc_size = self._t_enc_size
        wk = (t/10000**((2*torch.arange(0, enc_size//2, step=1).to(self._device))/enc_size))
        pe = torch.cat((torch.sin(wk), torch.cos(wk)), dim=0)
        
        return pe
    
    def get_timestep_encoding_size(self):
        return self._t_enc_size
    
    def noise_images(self, xs, ts):
        noise = torch.randn_like(xs)
        sqrt_alpha_cumprod = self._sqrt_alphas_cumprod[ts][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = self._sqrt_one_minus_alphas_cumprod[ts][:, None, None, None]
                                          
        return sqrt_alpha_cumprod * xs + sqrt_one_minus_alphas_cumprod * noise, noise
    
    def train_step(self, unet, loss, x_start, i_embs, x_unnoised_appendex=None):
        ts = self.sample_timesteps(x_start.shape[0]).to(self._device)
        x_t, target = self.noise_images(x_start, ts)
        x_t_app = torch.cat((x_t, x_unnoised_appendex), dim=1) if x_unnoised_appendex is not None else x_t
        tse = torch.stack([self.get_timestep_encoding(t) for t in ts]).to(self._device)
        out = unet(x_t_app, tse, i_embs)
        loss_vlb = 0
        self._step += 1
        
        if self._variance_mode == VarianceMode.LEARNED or self._variance_mode == VarianceMode.LEARNED_SCALE:
            predicted, model_var = out.chunk(2, dim=1)  
            predicted_frozen = predicted.detach()
            
            mean_coef_1 = self._posterior_mean_coef_1[ts][:, None, None, None]
            mean_coef_2 = self._posterior_mean_coef_2[ts][:, None, None, None]
            posterior_log_variance = self._posterior_log_variance_clipped[ts][:, None, None, None]
            posterior_mean = mean_coef_1 * x_start + mean_coef_2 * x_t          
            
            if self._variance_mode == VarianceMode.LEARNED:
                model_log_variance = model_var
            else:
                log_schedule = torch.log(self._schedule)[ts][:, None, None, None]
                # Output is [-1, 1] -> Normalize to [0, 1]
                
                # Fix for single exploding var values leading to nan and inf
                # model_var = torch.clamp(model_var, -2.0, 2.0)
                model_var = (model_var + 1) / 2
                model_log_variance = model_var * log_schedule + (1 - model_var) * posterior_log_variance
            
            recip_alphas_cum = self._sqrt_recip_alphas_cumprod[ts][:, None, None, None]
            recipm1_alphas_cum = self._sqrt_recipm1_alphas_cumprod[ts][:, None, None, None]
            pred_x_start = recip_alphas_cum * x_t - recipm1_alphas_cum * predicted_frozen
            model_mean = mean_coef_1 * pred_x_start + mean_coef_2 * x_t           
            
            k1 = VLBDiffusionLoss.kl_divergence(
                posterior_mean, 
                posterior_log_variance,
                model_mean, 
                model_log_variance
            )
            k1_mean = k1.mean(dim=list(range(1, len(k1.shape)))) / np.log(2.0)
            
            gll = -VLBDiffusionLoss.discretized_gaussian_log_likelihood(
                x_start, 
                predicted_frozen, 
                0.5 * model_log_variance
            )
            gll_mean = gll.mean(dim=list(range(1, len(gll.shape)))) / np.log(2.0)
            
            loss_vlb = torch.where((ts == 0), gll_mean, k1_mean).mean()
              
            if self._step % 100 == 0:
                try:
                    wandb.log(
                        {
                         "step": self._step,
                         "predicted": predicted, 
                         "model_mean'": model_mean, 
                         "model_log_var": model_log_variance, 
                         "model_var": model_var,
                         "posterior_mean": posterior_mean, 
                         "posterior_log_variance": posterior_log_variance,
                         "posterior_variance": torch.exp(posterior_log_variance),
                         "self._posterior_variance": self._posterior_variance,
                         "gll": gll,
                         "k1": k1,
                         "loss_vlb": loss_vlb
                        }
                    )
                except:
                    pass
            
            out = predicted
        
        loss = loss(target, out) + self._variance_lambda * loss_vlb * self._steps
            
        return loss
    
    def sample_timesteps(self, num):
        return torch.randint(low=0, high=self._steps, size=(num,))
    
    def sample_images(self, model, num_samples, i_embs=None, cfg_scale=3, x_appendex=None):
        # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((num_samples, self._img_channels, self._img_size, self._img_size)).to(self._device)
            for i in (pbar := tqdm(list(reversed(range(0, self._steps))), position=1)):
                pbar.set_description(f"Sampling Images: ")
                ts = (torch.ones(num_samples)*i).long().to(self._device)
                tse = torch.stack([self.get_timestep_encoding(step) for step in ts]).to(self._device)
                x_t_app= torch.cat((x_t, x_appendex), dim=1) if x_appendex is not None else x_t
                pred = model(x_t_app, tse, i_embs)
                
                if cfg_scale > 0:
                    uncondtional_pred = model(x_t_app, tse, None)
                    pred = torch.lerp(uncondtional_pred, pred, cfg_scale)
                                   
                alphas = self._alphas[ts][:, None, None, None]
                alphas_cum = self._alphas_cum[ts][:, None, None, None]
                noise_schedule = self._schedule[ts][:, None, None, None]
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                    
                if self._variance_mode == VarianceMode.LEARNED or self._variance_mode == VarianceMode.LEARNED_SCALE:
                    predicted, model_var = pred.chunk(2, dim=1)
                    """
                    if self._variance_mode == VarianceMode.LEARNED:
                        x_var = model_sigma * noise
                    else:
                        # Output is [-1, 1] -> Normalize to [0, 1]
                        model_sigma = (model_sigma + 1) / 2
                        posterior_variance = self._posterior_log_variance_clipped[ts][:, None, None, None].expand(model_sigma.shape)
                        x_var = model_sigma * torch.log(noise_schedule) + (1 - model_sigma) * posterior_variance
                        x_var = torch.exp(x_var) * noise
                    """
                    mean_coef_1 = self._posterior_mean_coef_1[ts][:, None, None, None]
                    mean_coef_2 = self._posterior_mean_coef_2[ts][:, None, None, None]
                    posterior_log_variance = self._posterior_log_variance_clipped[ts][:, None, None, None]        

                    if self._variance_mode == VarianceMode.LEARNED:
                        model_log_variance = model_var
                        model_variance = torch.exp(model_log_variance)
                    else:
                        log_schedule = torch.log(self._schedule)[ts][:, None, None, None]
                        # Output is [-1, 1] -> Normalize to [0, 1]
                        # model_var = torch.clamp(model_var, -2.0, 2.0)
                        model_var = (model_var + 1) / 2
                        model_log_variance = model_var * log_schedule + (1 - model_var) * posterior_log_variance
                        
                    recip_alphas_cum = self._sqrt_recip_alphas_cumprod[ts][:, None, None, None]
                    recipm1_alphas_cum = self._sqrt_recipm1_alphas_cumprod[ts][:, None, None, None]
                    pred_x_start = recip_alphas_cum * x_t - recipm1_alphas_cum * predicted
                    if self._clamp_x_start_in_sample:
                        pred_x_start = pred_x_start.clamp(-1, 1)
                    model_mean = mean_coef_1 * pred_x_start + mean_coef_2 * x_t
                    x_var = torch.exp(0.5 * model_log_variance) * noise
                else:
                    x_var = torch.sqrt(noise_schedule) * noise                                            
                    model_mean = (1 / torch.sqrt(alphas)) * (x - (noise_schedule / (torch.sqrt(1 - alphas_cum))) * pred)
                    
                x_t = model_mean + x_var
                
        model.train()
        
        print(torch.min(x_t), torch.max(x_t), torch.mean(x_t))
        x_t = x_t.clamp(-1, 1)
        return x_t
    
    

    

    
    
