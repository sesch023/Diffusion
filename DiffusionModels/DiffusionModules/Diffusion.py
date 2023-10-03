import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torchinfo import summary
import math
import clip
from abc import ABC, abstractmethod
from DiffusionModules.Modules import *
from tqdm import tqdm
from enum import Enum
from torchviz import make_dot
import wandb
from einops import rearrange
from DiffusionModules.EmbeddingTools import ClipTools


def equalize_shape_of_first(t1, t2):
    while len(t1.shape) < len(t2.shape):
        t1 = t1[..., None]
    return t1

    
class NoiseScheduler(ABC):
    @abstractmethod
    def get_noise_schedule(self, timesteps):
        pass
    
class CosineScheduler(NoiseScheduler):
    # Tau inspired as factor by https://arxiv.org/pdf/2301.10972.pdf
    def __init__(self, offset=0.008, tau=1):
        super().__init__()
        self._offset = offset
        self._tau = np.clip(tau, 0, 1)
        
    def get_noise_schedule(self, timesteps):
        # https://huggingface.co/blog/annotated-diffusion
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        betas = torch.cos(((x / timesteps) + self._offset) / (1 + self._offset) * torch.pi * 0.5) ** 2
        # Not needed as: (ft / fo) / (ft-1 / fo) = ft / ft-1
        # betas = betas / betas[0]
        betas = self._tau*(1 - (betas[1:] / betas[:-1]))
        return torch.clip(betas, max=0.9999)

    
class LinearScheduler(NoiseScheduler):
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self._beta_start = beta_start
        self._beta_end = beta_end
        
    def get_noise_schedule(self, timesteps):
        return torch.linspace(self._beta_start, self._beta_end, timesteps)

class SigmoidScheduler(NoiseScheduler):
    def __init__(self, beta_start = 0.0001, beta_end=0.02):
        super().__init__()
        self._beta_start = beta_start
        self._beta_end = beta_end
    
    def get_noise_schedule(self, timesteps):
        # https://huggingface.co/blog/annotated-diffusion
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (self._beta_end - self._beta_start) + self._beta_start
        
    
class VarianceMode(Enum):
    NOT_LEARNED = "NOT_LEARNED",
    LEARNED = "LEARNED",
    LEARNED_SCALE = "LEARNED_SCALE"
    
DEBUG = False

class DiffusionTools():
    # TODO: Über Steps nachdenken
    def __init__(
        self, 
        t_enc_size=256, 
        steps=1000, 
        noise_scheduler=None, 
        variance_mode=VarianceMode.LEARNED_SCALE, 
        variance_lambda=0.001, 
        clamp_x_start_in_sample=True, 
        device=None
    ):
        assert t_enc_size % 2 == 0, "Size of Timestep embedding needs to be even"
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._t_enc_size = t_enc_size
        self._variance_mode = variance_mode
        self._variance_lambda = variance_lambda
        self._steps = steps
        self._noise_scheduler = CosineScheduler() if not isinstance(noise_scheduler, NoiseScheduler) else noise_scheduler
        self._schedule = self._noise_scheduler.get_noise_schedule(steps).to(self._device)
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
    
    def get_pos_encoding(self, t, m=10000):
        enc_size = self._t_enc_size
        wk = (t/m**((2*torch.arange(0, enc_size//2, step=1).to(self._device))/enc_size))
        pe = torch.cat((torch.sin(wk), torch.cos(wk)), dim=0)
        
        return pe
    
    def get_pos_encoding_size(self):
        return self._t_enc_size
    
    def noise_data(self, xs, ts):
        noise = torch.randn_like(xs)
        sqrt_alpha_cumprod = equalize_shape_of_first(self._sqrt_alphas_cumprod[ts], xs)
        sqrt_one_minus_alphas_cumprod = equalize_shape_of_first(self._sqrt_one_minus_alphas_cumprod[ts], xs)
                                          
        return sqrt_alpha_cumprod * xs + sqrt_one_minus_alphas_cumprod * noise, noise
    
    def train_step(self, unet, loss, x_start, data_embs, x_unnoised_appendex=None, **unet_kwargs):
        ts = self.sample_timesteps(x_start.shape[0]).to(self._device)
        x_t, target = self.noise_data(x_start, ts)
        x_t_app = torch.cat((x_t, x_unnoised_appendex), dim=1) if x_unnoised_appendex is not None else x_t
        tse = torch.stack([self.get_pos_encoding(t) for t in ts]).to(self._device)
        out = unet(x_t_app, tse, data_embs, **unet_kwargs)
        loss_vlb = 0
        self._step += 1
        
        if self._variance_mode == VarianceMode.LEARNED or self._variance_mode == VarianceMode.LEARNED_SCALE:
            predicted, model_var = out.chunk(2, dim=1)  
            predicted_frozen = predicted.detach()
            
            mean_coef_1 = equalize_shape_of_first(self._posterior_mean_coef_1[ts], x_t)
            mean_coef_2 = equalize_shape_of_first(self._posterior_mean_coef_2[ts], x_t)
            posterior_log_variance = equalize_shape_of_first(self._posterior_log_variance_clipped[ts], x_t)
            posterior_mean = mean_coef_1 * x_start + mean_coef_2 * x_t          
            
            if self._variance_mode == VarianceMode.LEARNED:
                model_log_variance = model_var
            else:
                log_schedule = equalize_shape_of_first(torch.log(self._schedule)[ts], x_t)
                # Output is [-1, 1] -> Normalize to [0, 1]
                
                # Fix for single exploding var values leading to nan and inf
                # model_var = torch.clamp(model_var, -2.0, 2.0)
                model_var = (model_var + 1) / 2
                model_log_variance = model_var * log_schedule + (1 - model_var) * posterior_log_variance
            
            recip_alphas_cum = equalize_shape_of_first(self._sqrt_recip_alphas_cumprod[ts], x_t)
            recipm1_alphas_cum = equalize_shape_of_first(self._sqrt_recipm1_alphas_cumprod[ts], x_t)
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

            if DEBUG and self._step % 10 == 0:
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
    
    def sample_data(self, model, sample_shape, data_embs=None, cfg_scale=3, x_appendex=None, clamp_var=False, **unet_kwargs):
        # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py
        model.eval()
        with torch.no_grad():
            x_t = torch.randn(sample_shape).to(self._device)
            for i in (pbar := tqdm(list(reversed(range(0, self._steps))), position=1)):
                pbar.set_description(f"Sampling Data: ")
                ts = (torch.ones(sample_shape[0])*i).long().to(self._device)
                tse = torch.stack([self.get_pos_encoding(step) for step in ts]).to(self._device)
                x_t_app= torch.cat((x_t, x_appendex), dim=1) if x_appendex is not None else x_t
                pred = model(x_t_app, tse, data_embs, **unet_kwargs) 
                
                if cfg_scale > 0:
                    uncondtional_pred = model(x_t_app, tse, None)
                    pred = torch.lerp(uncondtional_pred, pred, cfg_scale)
                                
                alphas = equalize_shape_of_first(self._alphas[ts], x_t)
                alphas_cum = equalize_shape_of_first(self._alphas_cum[ts], x_t)
                noise_schedule = equalize_shape_of_first(self._schedule[ts], x_t)
                
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                    
                if self._variance_mode == VarianceMode.LEARNED or self._variance_mode == VarianceMode.LEARNED_SCALE:
                    predicted, model_var = pred.chunk(2, dim=1)

                    mean_coef_1 = equalize_shape_of_first(self._posterior_mean_coef_1[ts], x_t)
                    mean_coef_2 = equalize_shape_of_first(self._posterior_mean_coef_2[ts], x_t)
                    posterior_log_variance = equalize_shape_of_first(self._posterior_log_variance_clipped[ts], x_t)

                    if self._variance_mode == VarianceMode.LEARNED:
                        model_log_variance = model_var
                        model_var = torch.exp(model_log_variance)
                    else:
                        log_schedule = equalize_shape_of_first(torch.log(self._schedule)[ts], x_t)
                        # Output is [-1, 1] -> Normalize to [0, 1]

                        if clamp_var:
                            model_var = torch.clamp(model_var, -2.0, 2.0)

                        model_var = (model_var + 1) / 2
                        model_log_variance = model_var * log_schedule + (1 - model_var) * posterior_log_variance
                        
                    recip_alphas_cum = equalize_shape_of_first(self._sqrt_recip_alphas_cumprod[ts], x_t)
                    recipm1_alphas_cum = equalize_shape_of_first(self._sqrt_recipm1_alphas_cumprod[ts], x_t)
                    pred_x_start = recip_alphas_cum * x_t - recipm1_alphas_cum * predicted
                    if self._clamp_x_start_in_sample:
                        pred_x_start = pred_x_start.clamp(-1, 1)
                    model_mean = mean_coef_1 * pred_x_start + mean_coef_2 * x_t
                    x_var = torch.exp(0.5 * model_log_variance) * noise
                else:
                    x_var = torch.sqrt(noise_schedule) * noise                                            
                    model_mean = (1 / torch.sqrt(alphas)) * (x_t - (noise_schedule / (torch.sqrt(1 - alphas_cum))) * pred)
                    
                x_t = model_mean + x_var

                if DEBUG:
                    wandb.log(
                        {
                            "sam_step": self._step*1000+i,
                            "val_predicted": predicted, 
                            "val_model_mean'": model_mean, 
                            "val_model_log_var": model_log_variance, 
                            "val_model_var": model_var,
                            "val_posterior_mean_1": mean_coef_1, 
                            "val_posterior_mean_2": mean_coef_2, 
                            "val_posterior_log_variance": posterior_log_variance,
                            "val_posterior_variance": torch.exp(posterior_log_variance)
                        }
                    )
                
        model.train()

        print(torch.min(x_t), torch.max(x_t), torch.mean(x_t))
        x_t = x_t.clamp(-1, 1)
        return x_t
    
