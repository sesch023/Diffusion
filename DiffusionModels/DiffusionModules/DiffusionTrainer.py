import copy
import os
import shutil
import sys
from einops import rearrange
from enum import Enum

import lightning.pytorch as pl
import torch
import torchvision.transforms as transforms
from diffusers import LDMSuperResolutionPipeline
from super_image import DrlnModel, ImageLoader
from torch import nn, optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

import DiffusionModules.ClipTranslatorModules as tools
from DiffusionModules.FVD import FVDLoss
from DiffusionModules.DiffusionModels import ExponentialMovingAverage
from DiffusionModules.ModelLoading import load_udm
from DiffusionModules.EmbeddingTools import ClipTextEmbeddingProvider, ClipEmbeddingProvider
from Configs import ModelLoadConfig

sys.modules['ClipTranslatorModules'] = tools


class UpscalerMode(Enum):
    """
    Defines the method of upscaling the data.

    :param NONE: No upscaling is done.
    :param DRLN: DRLN from Package super_image is used for upscaling.
    :param LDM: Latent diffusion model by Stability AI is used for upscaling.
    :param UDM: Custom diffusion upscaler is used for upscaling.
    """    
    NONE="NONE",
    DRLN="DRLN",
    LDM="LDM",
    # Due to a bug with model loading and enum changes this had to be removed. 
    # Instead for now a string is used!
    # UDM="UDM"


"""
All pl.LightningModules in this file are highly redundant and should be refactored. Since all training methods are pretty close
but have slight changes in many places, it was difficult to find a good way to refactor this. As a proof of concept the
given solution is enough but in future works a better solution including torch lightning CLI should be used.
In that way yaml config files could define the process and react to different inputs. This would also remove the need
for different Train files.

Some other simple refractors could also include the extractions of similar methods and the creation of a base class. It should be said though 
that with regards to the many small changes this could be highly ineffective as a solution for redundancy.
"""

def get_fid(fid, samples, real):
    """
    Calculates the FID score of the samples and the real images
    with the given FID metric.

    :param fid: Instance of the FID metric.
    :param samples: Samples to calculate the FID score for.
    :param real: Real images to calculate the FID score to.
    :return: FID score of the samples compared to the real images.
    """    
    fid.reset()
    fid.update((((samples + 1)/2)*255).byte(), real=False)
    fid.update((((real + 1)/2)*255).byte(), real=True)
    fid = fid.compute()
    fid.reset()
    return fid

class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self, 
        unet, 
        diffusion_tools, 
        transformable_data_module, 
        loss=None, 
        val_score=None, 
        embedding_provider=None, 
        alt_validation_emb_provider=None, 
        ema_beta=0.9999, 
        cfg_train_ratio=0.1, 
        cfg_scale=3, 
        captions_preprocess=None, 
        optimizer=None, 
        sample_upscaler_mode=UpscalerMode.LDM, 
        sample_scale_factor=4, 
        checkpoint_every_val_epochs=10, 
        no_up_samples_out=True, 
        sample_images_out_base_path="samples/", 
        c_device="cpu"
    ):
        """
        Trainer for a basic diffusion model. This trainer is used to train the diffusion model and validate it
        using samples every val_step. The trainer also saves the model every checkpoint_every_val_epochs epochs
        if the validation score is lower than the previous checkpoint. This is also done every epoch but the
        result of the last epoch is overwritten. It also trains a expoential moving average model if ema_beta is not None.  

        :param unet: U-Net model to train.
        :param diffusion_tools: DiffusionTools instance to use. This defines the scheduler and diffusion steps.
        :param transformable_data_module: TransformableDataModule instance to use. This defines the transformations and data.
        :param loss: Loss function to use in train process, defaults to nn.MSELoss()
        :param val_score: Validation score function to use in validation process. This should only be changed from none in a
                          few edge cases since the whole process is not tested with other validation scores. Val score should
                          always return a dict with atleast the key fid_score in it. defaults to a FID score and a CLIP score.
        :param embedding_provider: Instance of a BaseEmbeddingProvider to use for getting the embeddings. Defaults to a ClipEmbeddingProvider
        :param alt_validation_emb_provider: Alternative instance of a BaseEmbeddingProvider to use for getting the embeddings in the validation process. 
                                            Defaults to the same as embedding_provider.
        :param ema_beta: beta value for the exponential moving average model. If None no ema model is trained, defaults to 0.9999
        :param cfg_train_ratio: Ratio of unconditional training steps to conditional training steps, defaults to 0.1
        :param cfg_scale: Extrapolation factor of the sample process with classifier free guidance, defaults to 3
        :param captions_preprocess: Additional function for preprocessing the captions, defaults to None
        :param optimizer: Optimizer to use in train process, defaults to optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        :param sample_upscaler_mode: Defines the method of upscaling the data, defaults to UpscalerMode.LDM
        :param sample_scale_factor: Defines the scale factor of the upscaling this should not be higher than 4, defaults to 4
        :param checkpoint_every_val_epochs: Check and output a checkpoint every checkpoint_every_val_epochs epochs, defaults to 10
        :param no_up_samples_out: Output the samples without upscaling, defaults to True
        :param sample_images_out_base_path: Output path of the samples, defaults to "samples/"
        :param c_device: Device to work on. This is needed separetly with the final target since some models dont move with the "to" method, 
                         defaults to "cpu"
        """        
        super().__init__()
        self.unet = unet
        self.diffusion_tools = diffusion_tools
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-4, weight_decay=0.0) if optimizer is None else optimizer
        self.sample_scale_factor = sample_scale_factor
        self.sample_upscaler_mode = sample_upscaler_mode
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        self.no_up_samples_out = no_up_samples_out
        self.c_device = c_device
        
        self.upscaler = None
        if self.sample_upscaler_mode is not None and self.sample_upscaler_mode != UpscalerMode.NONE:
            if self.sample_upscaler_mode == UpscalerMode.DRLN:
                self.up_model_pipeline = DrlnModel.from_pretrained('eugenesiow/drln', scale=sample_scale_factor)
                self.upscaler = lambda image, caption: self.up_model_pipeline(ImageLoader.load_image(image).to(self.c_device)).detach().to(self.c_device)     
                self.save_images = lambda image, path: ImageLoader.save_image(image, path)
            elif self.sample_upscaler_mode == UpscalerMode.LDM:
                model_id = "CompVis/ldm-super-resolution-4x-openimages"
                self.up_model_pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id).to(self.c_device)
                self.upscaler = lambda image, caption: self.up_model_pipeline(ImageLoader.load_image(image).to(self.c_device), num_inference_steps=100, eta=1).images[0].detach().cpu() 
                self.save_images = lambda image, path: image.save(path)
            # This should later Change to a Enum value, but changing a enum with pickled models does not work and results in load errors.
            elif self.sample_upscaler_mode == "UDM":
                model_path = ModelLoadConfig.upscaler_model_path
                self.up_model_pipeline = load_udm(model_path, self.c_device, self.transformable_data_module.img_in_target_size*sample_scale_factor)
                self.up_model_pipeline.eval()
                self.upscaler = lambda image, caption: self.up_model_pipeline([image], [caption], ema=True)[0]
                self.save_images = lambda image, path: image.save(path)
        
        # Create EMA model if ema_beta is not None
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
        
        # This whole definition is not a very nice solution but it works for now.
        if val_score is None:
            
            self.fid = FrechetInceptionDistance(feature=2048)
            
            self.clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").eval()
            
            self.val_score = lambda samples, real, captions: {
                "clip_score": self.clip_model((((samples + 1)/2)*255).int(), captions),
                "fid_score": get_fid(self.fid, samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
        
    
    def training_step(self, batch, batch_idx):
        """
        A single training step. Returns the loss.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the loss.
        """        
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        # Get the embeddings for the images and captions if the cfg_train_ratio is not reached
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.device)
        else:
            i_embs = None
        # Transform the images
        images = self.transformable_data_module.transform_batch(images).to(self.device)  
        # Step
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        """
        Updates the exponential moving average model at the end of a train epoch if ema_beta is not None.
        """        
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        """
        A single validation step. Samples images from the model and calculates the validation score.
        The sampled images are saved to the output path.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the validation score.
        """         
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        # Get the embeddings for the images and captions    
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        # Transform the images
        images = self.transformable_data_module.transform_batch(images).to(self.device)    
        # Sample the images
        sampled_images = self.diffusion_tools.sample_data(self.unet, images.shape, i_embs, self.cfg_scale)
        self.save_sampled_images(sampled_images, captions, batch_idx, "normal")
        if self.upscaler is not None and self.no_up_samples_out:
            self.save_sampled_images(sampled_images, captions, batch_idx, "normal_no_up", no_upscale=True)

        # Sample the images with the ema model if ema_beta is not None
        if self.ema is not None:
            sampled_images = self.diffusion_tools.sample_data(self.ema_unet, images.shape, i_embs, self.cfg_scale)
            self.save_sampled_images(sampled_images, captions, batch_idx, "ema")
            if self.upscaler is not None and self.no_up_samples_out:
                self.save_sampled_images(sampled_images, captions, batch_idx, "ema_no_up", no_upscale=True)
        
        try:
            val_score = self.val_score(sampled_images, images, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        """
        The end of the validation epoch. Calculates the average of the validation scores and logs them.
        Also saves a checkpoint if the average FID score is lower than the previous checkpoint, if
        the checkpoint_every_val_epochs is reached. It also saves a checkpoint every epoch but overwrites
        the last one.
        """        
        avg_dict = dict()
        outs = self.validation_step_outputs
        
        for key in outs[0].keys():
            values = [outs[i][key] for i in range(len(outs)) if key in outs[i]]
            avg = sum(values) / len(values)
            avg_dict[key] = avg

        self.log_dict(avg_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        # Since FID is the only sensible score to use for now only this is used for checkpointing.
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_score"]
        
        path = f"{self.sample_images_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None, no_upscale=False):
        """
        Saves the sampled images and captions to the output path.

        :param sampled_images: Sampled images to save.
        :param captions: Captions or class labels of the sampled images.
        :param batch_idx: Index of the batch.
        :param note: Additional note to add to the output path, defaults to None
        :param no_upscale: If true no upscaling is done, defaults to False
        """        
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
            
        if self.upscaler is not None and not no_upscale:
            sampled_images = [self.upscaler(image, caption) for image, caption in zip(sampled_images, captions)]
            
        for image_id in range(len(sampled_images)):
            self.save_images(sampled_images[image_id], path_folder + f"img_{image_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for the training process.

        :return: Returns a dict with the optimizer and scheduler.
        """        
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }
    
    
class UpscalerDiffusionTrainer(pl.LightningModule):
    def __init__(
        self, 
        unet, 
        diffusion_tools, 
        transformable_data_module, 
        start_size=64, 
        target_size=256, 
        loss=None, 
        val_score=None, 
        embedding_provider=None, 
        alt_validation_emb_provider=None, 
        ema_beta=0.9999, 
        cfg_train_ratio=0.1, 
        cfg_scale=3, 
        captions_preprocess=None, 
        optimizer=None, 
        checkpoint_every_val_epochs=10, 
        sample_images_out_base_path="samples_upscale/"
    ):
        """
        Trainer for a upscaler diffusion model. This trainer is used to train the diffusion model and validate it
        using samples every val_step. The trainer also saves the model every checkpoint_every_val_epochs epochs.
        It also trains a expoential moving average model if ema_beta is not None.  

        :param unet: U-Net model to train.
        :param diffusion_tools: DiffusionTools instance to use. This defines the scheduler and diffusion steps.
        :param transformable_data_module: TransformableDataModule instance to use. This defines the transformations and data.
        :param loss: Loss function to use in train process, defaults to nn.MSELoss()
        :param val_score: Validation score function to use in validation process. This should only be changed from none in a
                          few edge cases since the whole process is not tested with other validation scores. Val score should
                          always return a dict with atleast the key fid_score in it. defaults to a FID score.
        :param embedding_provider: Instance of a BaseEmbeddingProvider to use for getting the embeddings. Defaults to a ClipEmbeddingProvider
        :param alt_validation_emb_provider: Alternative instance of a BaseEmbeddingProvider to use for getting the embeddings in the validation process. 
                                            Defaults to the same as embedding_provider.
        :param ema_beta: beta value for the exponential moving average model. If None no ema model is trained, defaults to 0.9999
        :param cfg_train_ratio: Ratio of unconditional training steps to conditional training steps, defaults to 0.1
        :param cfg_scale: Extrapolation factor of the sample process with classifier free guidance, defaults to 3
        :param captions_preprocess: Additional function for preprocessing the captions, defaults to None
        :param optimizer: Optimizer to use in train process, defaults to optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        :param checkpoint_every_val_epochs: Check and output a checkpoint every checkpoint_every_val_epochs epochs, defaults to 10
        :param sample_images_out_base_path: Output path of the upscaled samples, defaults to "samples/"
        """        
        super().__init__()
        self.unet = unet 
        self.diffusion_tools = diffusion_tools  
        self.start_size = start_size
        self.target_size = target_size
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-5, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        
        # Extra pipeline for creating low res images
        self.transform_low_res = transforms.Compose([
            transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            transforms.Resize(start_size),
            transforms.CenterCrop(start_size),
            transforms.Resize(target_size),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
            
        if val_score is None:
            self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
            self.val_score = lambda samples, real, captions: {
                "fid_score": get_fid(self.fid, samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
        

    def forward(self, images, captions, ema=True):
        """
        Upscales the images with the model.

        :param images: Upscales the images to the target size.
        :param captions: Captions or class labels of the images.
        :param ema: If true the ema model is used if available, defaults to True
        :return: Upcaled images.
        """        
        model = self.ema_unet if ema and self.ema is not None else self.unet
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        low_res = torch.stack([self.transform_low_res(image) for image in images]).to(self.device)
        sampled_images = self.diffusion_tools.sample_data(model, low_res.shape, i_embs, self.cfg_scale, x_appendex=low_res)
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
        return sampled_images


    def training_step(self, batch, batch_idx):
        """
        A single training step. Returns the loss.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the loss.
        """        
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        # Get the embeddings for the images and captions if the cfg_train_ratio is not reached
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = self.embedding_provider.get_embedding(images, captions).to(self.device)
        else:
            i_embs = None
        
        # Create the low res images
        low_res = torch.stack([self.transform_low_res(image).to(self.device) for image in images]).to(self.device)  
        images = self.transformable_data_module.transform_batch(images).to(self.device)   
        loss = self.diffusion_tools.train_step(self.unet, self.loss, images, i_embs, x_unnoised_appendex=low_res)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        """
        Updates the exponential moving average model at the end of a train epoch if ema_beta is not None.
        """        
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        """
        A single validation step. Upscales images with the model and calculates the validation score.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the validation score.
        """        
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        # Create the low res images
        low_res = torch.stack([self.transform_low_res(image).to(self.device) for image in images])   
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        images = self.transformable_data_module.transform_batch(images).to(self.device)
        sampled_images = self.diffusion_tools.sample_data(self.unet, images.shape, i_embs, self.cfg_scale, x_appendex=low_res)
        self.save_sampled_images(sampled_images, captions, batch_idx, "normal") 

        # Upscale the images with the ema model if ema_beta is not None
        if self.ema is not None:
            sampled_images = self.diffusion_tools.sample_data(self.ema_unet, images.shape, i_embs, self.cfg_scale, x_appendex=low_res)
            self.save_sampled_images(sampled_images, captions, batch_idx, "ema")

        self.save_sampled_images(low_res, captions, batch_idx, "low_res")
        
        try:
            val_score = self.val_score(sampled_images, images, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        """
        The end of the validation epoch. Calculates the average of the validation scores and logs them.
        Also saves a checkpoint if the average FID score is lower than the previous checkpoint, if
        the checkpoint_every_val_epochs is reached. It also saves a checkpoint every epoch but overwrites
        the last one.
        """        
        avg_dict = dict()
        outs = self.validation_step_outputs
        
        for key in outs[0].keys():
            values = [outs[i][key] for i in range(len(outs)) if key in outs[i]]
            avg = sum(values) / len(values)
            avg_dict[key] = avg
       
        self.log_dict(avg_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_images_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_score"]
        
        path = f"{self.sample_images_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_images(self, sampled_images, captions, batch_idx, note=None):
        """
        Saves the upscaled images and captions to the output path.

        :param sampled_images: Upscaled images to save.
        :param captions: Captions or class labels of the upscaled images.
        :param batch_idx: Index of the batch.
        :param note: Additional note to add to the output path, defaults to None
        """        
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_images_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        
        sampled_images = self.transformable_data_module.reverse_transform_batch(sampled_images.detach().cpu())
        sampled_images = [ImageLoader.load_image(image) for image in sampled_images]
            
        for image_id in range(len(sampled_images)):
            self.save_images(sampled_images[image_id], path_folder + f"img_{image_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for the training process.

        :return: Returns a dict with the optimizer and scheduler.
        """        
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }


class SpatioTemporalDiffusionTrainer(pl.LightningModule):
    def __init__(
        self, 
        unet, 
        diffusion_tools, 
        transformable_data_module, 
        loss=None, 
        val_score=None, 
        embedding_provider=None, 
        temporal_embedding_provider=None,
        alt_validation_emb_provider=None, 
        alt_validation_temporal_emb_provider=None,
        ema_beta=0.9999, 
        cfg_train_ratio=0.1, 
        cfg_scale=3, 
        captions_preprocess=None, 
        optimizer=None, 
        checkpoint_every_val_epochs=10, 
        sample_data_out_base_path="samples_spatio_temporal/",
        disable_temporal_embs=True,
        temporal=True,
        after_load_fvd=False
    ):
        """
        Trainer for a spatio temporal diffusion model. This trainer is used to train the diffusion model and validate it
        using samples every val_step. The trainer also saves the model every checkpoint_every_val_epochs epochs.
        It also trains a expoential moving average model if ema_beta is not None.

        This trainer can work with temporal and non temporal data. If temporal is set to True the trainer expects
        a temporal dataset and uses the temporal embedding provider. If temporal is False the trainer expects a
        non temporal dataset and uses the normal embedding provider. 

        :param unet: U-Net model to train.
        :param diffusion_tools: DiffusionTools instance to use. This defines the scheduler and diffusion steps.
        :param transformable_data_module: TransformableDataModule instance to use. This defines the transformations and data.
        :param loss: Loss function to use in train process, defaults to nn.MSELoss()
        :param val_score: Validation score function to use in validation process. This should only be changed from none in a
        :param embedding_provider: Instance of a BaseEmbeddingProvider to use for getting the embeddings. 
                                   Defaults to a ClipEmbeddingProvider
        :param temporal_embedding_provider: Instance of a BaseEmbeddingProvider to use for getting the temporal embeddings. 
                                            Defaults to a ClipTextEmbeddingProvider
        :param alt_validation_emb_provider: Alternative instance of a BaseEmbeddingProvider to use for getting the embeddings in the validation process.
                                            Defaults to the same as embedding_provider.
        :param alt_validation_temporal_emb_provider: Alternative instance of a BaseEmbeddingProvider to use for getting the temporal embeddings in the validation process.
                                                     Defaults to the same as temporal_embedding_provider.
        :param ema_beta: beta value for the exponential moving average model. If None no ema model is trained, defaults to 0.9999
        :param cfg_train_ratio: Ratio of unconditional training steps to conditional training steps, defaults to 0.1
        :param cfg_scale: Extrapolation factor of the sample process with classifier free guidance, defaults to 3
        :param captions_preprocess: Additional function for preprocessing the captions, defaults to None
        :param optimizer: Optimizer to use in train process, defaults to optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        :param checkpoint_every_val_epochs: Check and output a checkpoint every checkpoint_every_val_epochs epochs, defaults to 10
        :param sample_data_out_base_path: Output path of the sampled data, defaults to "samples/"
        :param disable_temporal_embs: If true the temporal embeddings are not used in the training process, as described in the 
                                              make a video paper, defaults to True
        :param temporal: If true the trainer expects a temporal dataset and uses the temporal embedding provider. If false the trainer expects a
                         non temporal dataset and uses the normal embedding provider. Both dataset and parameter can be changed after initialization,
                         defaults to True
        :param after_load_fvd: Load the FVD model after the intialization of the trainer. This is purely for compatibility reasons, defaults to False
        """    
        super().__init__()
        self.unet = unet
        self.diffusion_tools = diffusion_tools
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.temporal_embedding_provider = ClipTextEmbeddingProvider() if temporal_embedding_provider is None else temporal_embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.alt_validation_temporal_emb_provider = self.temporal_embedding_provider if alt_validation_temporal_emb_provider is None else alt_validation_temporal_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_data_out_base_path = sample_data_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=3e-5, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.temporal = temporal
        self.writer_module = self.transformable_data_module.t_data if self.transformable_data_module.t_data is not None else self.transformable_data_module.v_data
        if self.writer_module is None:
            print("Warning: No transformable dataset module found for saving samples!")
        self.save_videos= lambda video, path: self.writer_module.write_video(video, path)
        self.save_images = lambda image, path: image.save(path)
        self.disable_temporal_embs = disable_temporal_embs
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
        
        if val_score is None:
            self.fid = FrechetInceptionDistance(feature=2048)
            self.clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").eval()
            if not after_load_fvd:
                self.load_fvd()

            self.val_score = lambda samples, real, captions: {
                "clip_score": self.get_clip_score(samples, captions),
                "fid_fvd_score": self.get_fvd_fid(samples, real),
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet"])
    
    def load_fvd(self):
        """
        Loads the FVD model. This is purely for compatibility reasons. This must be
        done if after_load_fvd is set to True.
        """        
        self.fvd = FVDLoss(self.device)

    def get_fvd_fid(self, s_data, r_data):
        """
        Gets the FVD or FID score of the sampled data and real data
        depending on the dimensionality of the data.

        :param s_data: Sampled data.
        :param r_data: Real data.
        :return: Returns the FVD or FID score.
        """        
        if s_data.ndim == 5:
            score = self.fvd(s_data, r_data)
            return score

        return get_fid(self.fid, s_data, r_data)

    def get_clip_score(self, s_data, captions):
        """
        Gets the CLIP score of the sampled data and captions.
        If the dimensionality of the data is 4 the data is rearranged
        and the scores are calculated for each frame and averaged.

        :param s_data: Sampled data.
        :param captions: Captions of the sampled data.
        :return: Returns the CLIP score.
        """        
        scores = []
        if s_data.ndim == 4:
            s_data = rearrange(s_data, 'b c h w -> b 1 c h w')
        else:
            s_data = rearrange(s_data, 'b c t h w -> b t c h w')
            
        for b_id in range(s_data.shape[0]):
            for f_id in range(s_data.shape[1]):
                score = self.clip_model((((s_data[b_id][f_id] + 1)/2)*255).int(), captions[b_id])
                scores.append(score)

        return sum(scores) / len(scores)
    

    def training_step(self, batch, batch_idx):
        """
        A single training step. Returns the loss.

        :param batch: Batch of data and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the loss.
        """        
        data, captions, *other = batch  
        # If there is additional data in the batch, use it
        fps = other[1] if len(other) > 1 else None
        length = other[0] if len(other) > 0 else None
        f_emb = None
        transformed_data = self.transformable_data_module.transform_batch(data).to(self.device) 
        temporal = self.temporal and transformed_data.ndim == 5
        
        # Get the temporal embeddings if we have temporal data
        if temporal:
            f_emb = torch.stack([self.diffusion_tools.get_pos_encoding(f) for f in fps]).to(self._device) 

        # Use the temporal embedding provider if we have temporal data
        embedding_provider = self.embedding_provider
        if transformed_data.ndim == 5:
            embedding_provider = self.temporal_embedding_provider

        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        
        # Get the embeddings for the data and captions if the cfg_train_ratio is not reached
        if torch.rand(1)[0] > self.cfg_train_ratio and not (temporal and self.disable_temporal_embs):
            d_embs = embedding_provider.get_embedding(data, captions).to(self.device)
        else:
            d_embs = None
 
        # Train step
        loss = self.diffusion_tools.train_step(self.unet, self.loss, transformed_data, d_embs, f_emb=f_emb, temporal=temporal)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=transformed_data.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Updates the exponential moving average model at the end of a train epoch if ema_beta is not None.
        """        
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        """
        A single validation step. Samples data with the model and calculates the validation score.

        :param batch: Batch of data and captions.
        :param batch_idx: Index of the batch.
        :return: Returns the validation score.
        """        
        data, captions, *other = batch  
        # If there is additional data in the batch, use it
        fps = other[1] if len(other) > 1 else None
        length = other[0] if len(other) > 0 else None
        f_emb = None
        transformed_data = self.transformable_data_module.transform_batch(data).to(self.device) 
        temporal = self.temporal and transformed_data.ndim == 5

        # Get the temporal embeddings if we have temporal data
        if temporal:
            f_emb = torch.stack([self.diffusion_tools.get_pos_encoding(f) for f in fps]).to(self._device) 

        # Use the temporal embedding provider if we have temporal data
        notes = ("normal", "ema")
        embedding_provider = self.embedding_provider
        if transformed_data.ndim == 5:
            embedding_provider = self.temporal_embedding_provider
            notes = ("normal_temp", "ema_temp")

        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        d_embs = embedding_provider.get_embedding(data, captions).to(self.device)

        # Sample the data
        normal_sampled = self.diffusion_tools.sample_data(self.unet, transformed_data.shape, d_embs, self.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=temporal)
        # Sample the data with the ema model if ema_beta is not None
        if self.ema is not None:
            ema_sampled = self.diffusion_tools.sample_data(self.ema_unet, transformed_data.shape, d_embs, self.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=temporal)
        
        self.save_sampled_data(normal_sampled, captions, batch_idx, notes[0])
        self.save_sampled_data(ema_sampled, captions, batch_idx, notes[1])
            
        try:
            val_score = self.val_score(ema_sampled, transformed_data, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        """
        The end of the validation epoch. Calculates the average of the validation scores and logs them.
        Also saves a checkpoint if the average fid_fvd_score is lower than the previous checkpoint, if
        the checkpoint_every_val_epochs is reached. It also saves a checkpoint every epoch but overwrites
        the last one.
        """        
        avg_dict = dict()
        outs = self.validation_step_outputs
        
        for key in outs[0].keys():
            values = [outs[i][key] for i in range(len(outs)) if key in outs[i]]
            avg = sum(values) / len(values)
            avg_dict[key] = avg

        self.log_dict(avg_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch += 1
        if self.val_epoch % self.checkpoint_every_val_epochs == 0 and avg_dict["fid_fvd_score"] < self.prev_checkpoint_val_avg:
            epoch = self.current_epoch
            path = f"{self.sample_data_out_base_path}/{str(epoch)}_model.ckpt"
            print(f"Saving Checkpoint at: {path}")
            self.trainer.save_checkpoint(path)
            
            if self.prev_checkpoint is not None:
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = path
            self.prev_checkpoint_val_avg = avg_dict["fid_fvd_score"]
        
        path = f"{self.sample_data_out_base_path}/latest.ckpt"
        print(f"Saving Checkpoint at: {path}")
        self.trainer.save_checkpoint(path)

        self.validation_step_outputs.clear() 
            
        
    def save_sampled_data(self, sampled_data, captions, batch_idx, note=None):
        """
        Saves the sampled data and captions to the output path. Depending on the dimensionality
        of the data the data is saved as a video or as images.

        :param sampled_data: Sampled data to save.
        :param captions: Captions or class labels of the sampled data.
        :param batch_idx: Index of the batch.
        :param note: Additional note to add to the output path, defaults to None
        """        
        epoch = self.current_epoch
        note = f"_{note}" if note is not None else ""
        path_folder = f"{self.sample_data_out_base_path}/{str(epoch)}_{str(batch_idx)}{note}/"
        path_cap = f"{path_folder}/{str(epoch)}_{str(batch_idx)}.txt"
        
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder)
        temporal = sampled_data.ndim == 5
        
        sampled_data = self.transformable_data_module.reverse_transform_batch(sampled_data.detach().cpu())
            
        for data_id in range(len(sampled_data)):
            if temporal:
                self.save_videos(sampled_data[data_id], path_folder + f"vid_{data_id}.mp4")
            else:
                self.save_images(sampled_data[data_id], path_folder + f"img_{data_id}.png")
        
        with open(path_cap, "w") as f:
            for cap in captions:
                f.write(cap)
                f.write("\n")
        
    
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for the training process.

        :return: Returns a dict with the optimizer and scheduler.
        """        
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10, gamma = lr_decay)      
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=2, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "fid_fvd_score",
                "interval": "epoch",
                "frequency": 200,
                "strict": False
            }
        }