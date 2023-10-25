from abc import ABC, abstractmethod
from enum import Enum

import torch
import numpy as np
from tqdm import tqdm
import wandb

from Configs import RunConfig
from DiffusionModules.Modules import VLBDiffusionLoss


def equalize_shape_of_first(t1, t2):
    """
    Equalizes the dimensions of the first tensor to the second tensor by adding dimensions to the first tensor.

    :param t1: Tensor to equalize dimensions of.
    :param t2: Tensor to equalize dimensions to.
    :return: Tensor with equalized dimensions.
    """    
    while len(t1.shape) < len(t2.shape):
        t1 = t1[..., None]
    return t1
    

class NoiseScheduler(ABC):
    @abstractmethod
    def get_noise_schedule(self, timesteps):
        """
        Abstract method to get the noise variance for each timestep starting at 0 and ending at timesteps.

        :param timesteps: Number of timesteps to get the noise variance for.
        """        
        pass
    

class CosineScheduler(NoiseScheduler):
    def __init__(self, offset=0.008, tau=1):
        """
        Intializes cosine noise schedule as described in https://arxiv.org/pdf/2102.09672.pdf
        and https://huggingface.co/blog/annotated-diffusion.

        Tau inspired as factor by https://arxiv.org/pdf/2301.10972.pdf.

        :param offset: Small offset of variances to prevent the values of being to low near 0, defaults to 0.008
        :param tau: Scale factor of the variances. This was only used in tests, defaults to 1
        """        
        super().__init__()
        self._offset = offset
        self._tau = np.clip(tau, 0, 1)
        
    def get_noise_schedule(self, timesteps):
        """
        Gets the cosine distributed noise variance for each timestep starting at 0 and ending at timesteps.

        :param timesteps: Number of timesteps to get the noise variance for.
        :return: Tensor of noise variances for each timestep.
        """        
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        betas = torch.cos(((x / timesteps) + self._offset) / (1 + self._offset) * torch.pi * 0.5) ** 2
        # Not needed as: (ft / fo) / (ft-1 / fo) = ft / ft-1
        # betas = betas / betas[0]
        betas = self._tau*(1 - (betas[1:] / betas[:-1]))
        return torch.clip(betas, max=0.9999)
    

class LinearScheduler(NoiseScheduler):
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        """
        Intializes linear noise schedule as described in the original DDPM paper
        https://arxiv.org/pdf/2006.11239.pdf.

        :param beta_start: Min of the variances at 0, defaults to 0.0001
        :param beta_end: Max of the variances at T, defaults to 0.02
        """        
        super().__init__()
        self._beta_start = beta_start
        self._beta_end = beta_end
        
    def get_noise_schedule(self, timesteps):
        """
        Gets the linear distribued noise variance for each timestep starting at 0 and ending at timesteps.

        :param timesteps: Number of timesteps to get the noise variance for.
        :return: Tensor of noise variances for each timestep.
        """        
        return torch.linspace(self._beta_start, self._beta_end, timesteps)


class SigmoidScheduler(NoiseScheduler):
    def __init__(self, beta_start = 0.0001, beta_end=0.02):
        """
        Intializes sigmoid noise schedule as described in https://arxiv.org/pdf/2301.10972.pdf
        and https://huggingface.co/blog/annotated-diffusion.

        :param beta_start: Min of the variances at 0 before sigmoid scaling, defaults to 0.0001
        :param beta_end: Max of the variances at T before sigmoid scaling, defaults to 0.02
        """        
        super().__init__()
        self._beta_start = beta_start
        self._beta_end = beta_end
    
    def get_noise_schedule(self, timesteps):
        """
        Gets the sigmoid distribued noise variance for each timestep starting at 0 and ending at timesteps.

        :param timesteps: Number of timesteps to get the noise variance for.
        :return: Tensor of noise variances for each timestep.
        """        
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (self._beta_end - self._beta_start) + self._beta_start


class VarianceMode(Enum):
    """
    Enum for the variance mode of the diffusion model.

    NOT_LEARNED: No variance is learned. This is decribed in the original DDPM paper https://arxiv.org/pdf/2006.11239.pdf.
    LEARNED: The variance is learned as described in https://arxiv.org/pdf/2102.09672.pdf.
             No scaling is applied to the variance. Since the Paper states that this method
             is not recommended, it was not used in the experiments and was not tested.
    LEARNED_SCALE: The variance is learned as a linear interpolation between the minimum 
                   and maximum variance as described in https://arxiv.org/pdf/2102.09672.pdf.
                   Since this method is recommended in the paper, it was used in the experiments.
    """    
    NOT_LEARNED = "NOT_LEARNED",
    LEARNED = "LEARNED",
    LEARNED_SCALE = "LEARNED_SCALE"
    

class DiffusionTools():
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
        """
        Intializes the diffusion tools. This class is used to train and sample from the diffusion model.
        It defines the entire diffusion process depending on a given schedule and variance mode.
        The processes were described in https://arxiv.org/pdf/2006.11239.pdf and https://arxiv.org/pdf/2102.09672.pdf.

        The implementation is based on and inspired by OpenAI's Guided Diffusion implementation:
        https://github.com/openai/guided-diffusion/tree/main

        :param Size of the sinusodial timestep encoding: , defaults to 256
        :param steps: Number of steps of the diffusion process, defaults to 1000
        :param noise_scheduler: The Noise-Scheduler to use. Should be an instance of NoiseScheduler, defaults to CosineScheduler
        :param variance_mode: The variances modes to use as decribed in the documentation of VarianceMode, defaults to VarianceMode.LEARNED_SCALE
        :param variance_lambda: Defines the influence of the variance loss on the total loss, defaults to 0.001
        :param clamp_x_start_in_sample: Clamps the results of x_start in the sample process to the range of -1 and 1. 
                                        This is helpful in the early stages of the learning process, defaults to True
        :param device: Device to put the Tensors on, defaults to "cuda" if cuda is available else "cpu"
        """        
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._t_enc_size = t_enc_size
        self._variance_mode = variance_mode
        self._variance_lambda = variance_lambda
        self._steps = steps
        self._clamp_x_start_in_sample = clamp_x_start_in_sample
        self._noise_scheduler = CosineScheduler() if not isinstance(noise_scheduler, NoiseScheduler) else noise_scheduler
        # Get noise schedule for each timestep.
        self._schedule = self._noise_scheduler.get_noise_schedule(steps).to(self._device)
        # Get alphas for each timestep.
        self._alphas = 1.0 - self._schedule
        # Calculate cumulative product of the alphas.
        self._alphas_cum = torch.cumprod(self._alphas, dim=0)
        # Calculate cumulative product of the alphas of the previous timestep. This is needed for the variance calculation.
        self._alphas_cum_prev = torch.cat((torch.Tensor([1.0]).to(self._device), self._alphas_cum[:-1]))
        # Calculate the square root of the cumulative product of the alphas.
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cum)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cum)
        # Calculate the posterior variance beta_t~ as described in https://arxiv.org/pdf/2006.11239.pdf in Formula 7.
        self._posterior_variance  = self._schedule * (1. - self._alphas_cum_prev) / (1. - self._alphas_cum)
        self._posterior_log_variance_clipped = torch.log(torch.cat((self._posterior_variance[1:2], self._posterior_variance[1:])))
        # Calculate the posterior mean coefficients as described in https://arxiv.org/pdf/2006.11239.pdf in Formula 7.
        self._posterior_mean_coef_1 = self._schedule * torch.sqrt(self._alphas_cum_prev) / (1.0 - self._alphas_cum)
        self._posterior_mean_coef_2 = (1 - self._alphas_cum_prev) * torch.sqrt(self._alphas) / (1.0 - self._alphas_cum) 
        self._sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self._alphas_cum)
        self._sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self._alphas_cum - 1)

        self._step = 0
    
    def get_pos_encoding(self, t, m=10000):
        """
        Calculates the sinusoidal positional encoding for a given timestep t.

        :param t: Timestep to get the positional encoding for.
        :param m: Parameter given by the Paper Attention is all you need, defaults to 10000
        :return: Sinusoidal positional encoding for the given timestep.
        """        
        enc_size = self._t_enc_size
        wk = (t/m**((2*torch.arange(0, enc_size//2, step=1).to(self._device))/enc_size))
        pe = torch.cat((torch.sin(wk), torch.cos(wk)), dim=0)
        
        return pe
    
    def get_pos_encoding_size(self):
        """
        Returns the size of the sinusoidal positional encoding.

        :return: Size of the sinusoidal positional encoding.
        """        
        return self._t_enc_size
    
    def noise_data(self, xs, ts):
        """
        Forward pass of the diffusion process. Adds noise to the given data in a 
        single step. This is described in https://arxiv.org/pdf/2006.11239.pdf on page 3.

        :param xs: Data to add noise depending on the timestep to.
        :param ts: Timesteps to add noise for.
        :return: Tuple of the noised data and the noise.
        """        
        noise = torch.randn_like(xs)
        sqrt_alpha_cumprod = equalize_shape_of_first(self._sqrt_alphas_cumprod[ts], xs)
        sqrt_one_minus_alphas_cumprod = equalize_shape_of_first(self._sqrt_one_minus_alphas_cumprod[ts], xs)
                                          
        return sqrt_alpha_cumprod * xs + sqrt_one_minus_alphas_cumprod * noise, noise

    def sample_timesteps(self, num):
        """
        Gets n random timesteps to train on.

        :param num: Number of timesteps to get.
        :return: Tensor of timesteps.
        """        
        return torch.randint(low=0, high=self._steps, size=(num,))

    def train_step(self, unet, loss, x_start, data_embs, x_unnoised_appendex=None, **unet_kwargs):
        """
        A single training step of the diffusion model. 

        :param unet: The U-Net model to train.
        :param loss: The loss function to use.
        :param x_start: The data to start the diffusion process with.
        :param data_embs: The data embeddings to use.
        :param x_unnoised_appendex: Additional inputs to the U-Net that are not noised, defaults to None
        :return: The loss of the training step.
        """        
        # Get the timesteps to train on.
        ts = self.sample_timesteps(x_start.shape[0]).to(self._device)
        # Forward pass of the diffusion process to the given timestep.
        x_t, target = self.noise_data(x_start, ts)
        # Append additional inputs if given.
        x_t_app = torch.cat((x_t, x_unnoised_appendex), dim=1) if x_unnoised_appendex is not None else x_t
        # Get the positional encoding for each timestep.
        tse = torch.stack([self.get_pos_encoding(t) for t in ts]).to(self._device)
        # Forward pass of the U-Net.
        out = unet(x_t_app, tse, data_embs, **unet_kwargs)
        loss_vlb = 0
        self._step += 1
        
        # Is the variance learned?
        if self._variance_mode == VarianceMode.LEARNED or self._variance_mode == VarianceMode.LEARNED_SCALE:
            # Get predicted mean and variance from the U-Net output.
            predicted, model_var = out.chunk(2, dim=1)  
            # Detach the U-Net output for the vlb loss calculation.
            predicted_frozen = predicted.detach()

            # Value of beta_t~ for the given timestep.
            posterior_log_variance = equalize_shape_of_first(self._posterior_log_variance_clipped[ts], x_t)

            # Calculate the posterior mean for the given timestep as described in https://arxiv.org/pdf/2102.09672.pdf in Formula 11.
            mean_coef_1 = equalize_shape_of_first(self._posterior_mean_coef_1[ts], x_t)
            mean_coef_2 = equalize_shape_of_first(self._posterior_mean_coef_2[ts], x_t)
            posterior_mean = mean_coef_1 * x_start + mean_coef_2 * x_t          
            
            if self._variance_mode == VarianceMode.LEARNED:
                # f the variance is learned without a scale, the log variance is the output of the U-Net.
                model_log_variance = model_var
            else:
                # If the variance is learned with a scale, the output of the U-Net is a value between -1 and 1
                # normalized to 0 and 1. The result is used for a linear interpolation between beta_t and beta_t~.
                log_schedule = equalize_shape_of_first(torch.log(self._schedule)[ts], x_t)
                model_var = (model_var + 1) / 2
                model_log_variance = model_var * log_schedule + (1 - model_var) * posterior_log_variance
            
            # Calculate the KL-Divergence between the real and the model distributions for every timestep except the first.
            k1 = VLBDiffusionLoss.kl_divergence(
                posterior_mean, 
                posterior_log_variance,
                model_mean, 
                model_log_variance
            )
            k1_mean = k1.mean(dim=list(range(1, len(k1.shape)))) / np.log(2.0)
            
            # Calculate the Gaussian Log Likelihood for the first timestep.
            gll = -VLBDiffusionLoss.discretized_gaussian_log_likelihood(
                x_start, 
                predicted_frozen, 
                0.5 * model_log_variance
            )
            gll_mean = gll.mean(dim=list(range(1, len(gll.shape)))) / np.log(2.0)
            
            # The vlb loss is the mean of the KL-Divergence taken for all timesteps except the first
            # and the Gaussian Log Likelihood taken for the first timestep.
            loss_vlb = torch.where((ts == 0), gll_mean, k1_mean).mean()

            # These outputs are only for debugging purposes.
            if RunConfig.debug and self._step % 10 == 0:
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

        # If the variance is not learned, the loss is calculated as the given Loss between the predicted noise and the added noise.
        # Since loss_vlb is 0 in that case, the loss is only the given Loss. The given Loss is usually the MSE loss.
        loss = loss(target, out) + self._variance_lambda * loss_vlb * self._steps
            
        return loss
    
    def sample_data(self, model, sample_shape, data_embs=None, cfg_scale=3, x_appendex=None, clamp_var=False, **unet_kwargs):
        """
        Samples data from the diffusion model for a given number of timesteps. 

        :param model: The U-Net model to sample from.
        :param sample_shape: The shape of the data to sample.
        :param data_embs: The data embeddings to use.
        :param cfg_scale: The value for the classifier free guidance extrapolation, defaults to 3
        :param x_appendex: Additional inputs to the U-Net which are given every step, defaults to None
        :param clamp_var: Clamps the variance to the range of -2 and 2 to stop exploding variances in early timesteps.
                          This was mainly used for debugging purposes, defaults to False
        :return: The sampled data.
        """        
        model.eval()
        with torch.no_grad():
            # Get the initial noise.
            x_t = torch.randn(sample_shape).to(self._device)
            # Sample the data for each timestep.
            for i in (pbar := tqdm(list(reversed(range(0, self._steps))), position=1)):
                pbar.set_description(f"Sampling Data: ")
                # Get the timesteps and the corresponding positional encodings.
                ts = (torch.ones(sample_shape[0])*i).long().to(self._device)
                tse = torch.stack([self.get_pos_encoding(step) for step in ts]).to(self._device)
                # Append additional inputs if given.
                x_t_app= torch.cat((x_t, x_appendex), dim=1) if x_appendex is not None else x_t
                # Forward pass of the U-Net.
                pred = model(x_t_app, tse, data_embs, **unet_kwargs) 
                
                # If the classifier free guidance extrapolation is used, make a unconditional prediction and 
                # interpolate between the unconditional and the conditional prediction depending on the given scale.
                # Since the cfg scale is bigger than 1, this leads to an extrapolation of the prediction.
                if cfg_scale > 0:
                    uncondtional_pred = model(x_t_app, tse, None)
                    pred = torch.lerp(uncondtional_pred, pred, cfg_scale)
                                
                alphas = equalize_shape_of_first(self._alphas[ts], x_t)
                alphas_cum = equalize_shape_of_first(self._alphas_cum[ts], x_t)
                noise_schedule = equalize_shape_of_first(self._schedule[ts], x_t)
                
                # The noise of the current timestep. If the timestep is 0, the noise is 0.
                if i > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                    
                # Is the variance learned?
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
                        

                        if clamp_var:
                            model_var = torch.clamp(model_var, -2.0, 2.0)
                            
                        # Output is [-1, 1] -> Normalize to [0, 1]
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
                    # No variance is learned. 
                    # This is described in the original DDPM paper https://arxiv.org/pdf/2006.11239.pdf in Formula 11.
                    x_var = torch.sqrt(noise_schedule) * noise                                            
                    model_mean = (1 / torch.sqrt(alphas)) * (x_t - (noise_schedule / (torch.sqrt(1 - alphas_cum))) * pred)
                    
                x_t = model_mean + x_var

                # These outputs are only for debugging purposes.
                if RunConfig.debug:
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

        # Information about the sampled data before clamping to the valid range. 
        # This can be used to find mistakes in the training or sampling process.
        print(torch.min(x_t), torch.max(x_t), torch.mean(x_t))
        x_t = x_t.clamp(-1, 1)
        return x_t
    
