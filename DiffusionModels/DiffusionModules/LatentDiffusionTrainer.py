import copy
import os
import shutil

import lightning.pytorch as pl
import torch
from super_image import ImageLoader
from torch import nn, optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

from DiffusionModules.DiffusionModels import ExponentialMovingAverage
from DiffusionModules.DiffusionTrainer import get_fid_score
from DiffusionModules.EmbeddingTools import ClipEmbeddingProvider


class LatentDiffusionTrainer(pl.LightningModule):
    def __init__(
        self, 
        unet, 
        vqgan, 
        latent_shape,
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
        checkpoint_every_val_epochs=10,
        quantize_after_sample=True,
        sample_images_out_base_path="samples/"
    ):
        """
        Creates a LatentDiffusionTrainer for training a latent diffusion model
        with a given unet and vqgan model.

        :param unet: UNet model.
        :param vqgan: VQGAN model.
        :param latent_shape: Shape of the latent space produced by the VQGAN model.
        :param diffusion_tools: DiffusionTools object.
        :param transformable_data_module: TransformableDataModule object.
        :param loss: Loss function, defaults to MSELoss
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
        :param optimizer: Optimizer to use in train process, defaults to optim.AdamW(model.parameters(), lr=9.6e-5, weight_decay=0.0)
        :param checkpoint_every_val_epochs: Check and output a checkpoint every checkpoint_every_val_epochs epochs 
                                            if the validation score is lower than the previous checkpoint, defaults to 10
        :param quantize_after_sample: Should the sampled latents be quantized after sampling, defaults to True
        :param sample_images_out_base_path: Base path for saving the sampled images, defaults to "samples/"
        """        
        super().__init__()
        self.unet = unet
        self.vqgan = vqgan.eval()
        self.diffusion_tools = diffusion_tools
        self.transformable_data_module = transformable_data_module
        self.loss = nn.MSELoss() if loss is None else loss
        self.embedding_provider = ClipEmbeddingProvider() if embedding_provider is None else embedding_provider
        self.alt_validation_emb_provider = self.embedding_provider if alt_validation_emb_provider is None else alt_validation_emb_provider
        self.cfg_scale = cfg_scale
        self.cfg_train_ratio = cfg_train_ratio
        self.captions_preprocess = captions_preprocess
        self.sample_images_out_base_path = sample_images_out_base_path
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=9.6e-5, weight_decay=0.0) if optimizer is None else optimizer
        self.val_epoch = 0
        self.checkpoint_every_val_epochs = checkpoint_every_val_epochs
        self.prev_checkpoint = None
        self.prev_checkpoint_val_avg = float("inf")
        self.validation_step_outputs = []
        self.save_images = lambda image, path: ImageLoader.save_image(image, path)
        self.latent_shape = latent_shape
        self.quantize_after_sample = quantize_after_sample
        self.codebook_max = self.vqgan.quantize.embedding.weight.max()
        self.codebook_min = self.vqgan.quantize.embedding.weight.min()
        
        if ema_beta is not None:   
            self.ema = ExponentialMovingAverage(ema_beta)
            self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)
        else:
            self.ema = None
        
        if val_score is None:
            self.fid = FrechetInceptionDistance(feature=2048)
            
            self.clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").eval()
            
            self.val_score = lambda samples, real, captions: {
                "clip_score": self.clip_model((((samples + 1)/2)*255).int(), captions),
                "fid_score": get_fid_score(self.fid, samples, real)
            }
        else:
            self.val_score = val_score
            
        self.save_hyperparameters(ignore=["embedding_provider", "unet", "vqgan"])
    
    def diff_norm_tensor(self, in_tensor):
        """
        Normalizes the tensor to the range [-1, 1] from the range [codebook_min, codebook_max].
        This is done after encoding the images and before training with the diffusion process.

        :param in_tensor: Input tensor.
        :return: Normalized tensor.
        """        
        min_val, max_val = self.codebook_min, self.codebook_max
        scaled_tensor = ((in_tensor - min_val) / (max_val - min_val)) * 2 - 1
        return scaled_tensor

    def codebook_norm_tensor(self, in_tensor):
        """
        Normalizes the tensor to the range [codebook_min, codebook_max] from the range [-1, 1].
        This is done before decoding the sampled latents.

        :param in_tensor: Input tensor.
        :return: 
        """        
        min_val, max_val = self.codebook_min, self.codebook_max
        scaled_tensor = (((in_tensor + 1) / 2) * (max_val - min_val)) + min_val
        return scaled_tensor

    def training_step(self, batch, batch_idx):
        """
        Training step for the latent diffusion process.

        :param batch: Batch of images and captions.
        :param batch_idx: Batch index.
        :return: Loss value.
        """        
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions
        i_embs_req = self.embedding_provider.get_embedding(images, captions).to(self.device)
        if torch.rand(1)[0] > self.cfg_train_ratio:
            i_embs = i_embs_req
        else:
            i_embs = None
        images = self.transformable_data_module.transform_batch(images).to(self.device)    
        with torch.no_grad():
            # Encode the images, quantize the latents and normalize the latents.
            latents, _, _ = self.vqgan.encode(images, emb=i_embs_req)
            latents, _, _ = self.vqgan.quantize(latents)
            latents = self.diff_norm_tensor(latents)
        loss = self.diffusion_tools.train_step(self.unet, self.loss, latents, i_embs)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Updates the exponential moving average model after every training batch.
        """        
        if self.ema is not None:
            self.ema.step_ema(self.ema_unet, self.unet)
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the latent diffusion process.

        :param batch: Batch of images and captions.
        :param batch_idx: Batch index.
        :return: Validation score.
        """        
        images, captions = batch
        captions = self.captions_preprocess(captions) if self.captions_preprocess is not None else captions    
        i_embs = self.alt_validation_emb_provider.get_embedding(images, captions).to(self.device)
        images = self.transformable_data_module.transform_batch(images).to(self.device)    

        latent_batch_shape = (images.shape[0], *self.latent_shape)
        sampled_latents = self.diffusion_tools.sample_data(self.unet, latent_batch_shape, i_embs, self.cfg_scale)
        # Normalize the sampled latents to the codebook range.
        sampled_latents = self.codebook_norm_tensor(sampled_latents)
        # If quantize_after_sample is true, quantize the sampled latents.
        if self.quantize_after_sample:
            sampled_latents, _, _ = self.vqgan.quantize(sampled_latents)
        samples_images = self.vqgan.decode(sampled_latents, emb=i_embs, clamp=True)
        self.save_sampled_images(samples_images, captions, batch_idx, "normal")

        # Repeat the same process for the exponential moving average model if it exists.
        if self.ema is not None:
            ema_sampled_latents = self.diffusion_tools.sample_data(self.ema_unet, latent_batch_shape, i_embs, self.cfg_scale)
            ema_sampled_latents = self.codebook_norm_tensor(ema_sampled_latents)
            if self.quantize_after_sample:
                ema_sampled_latents, _, _ = self.vqgan.quantize(ema_sampled_latents)
            samples_images = self.vqgan.decode(ema_sampled_latents, emb=i_embs, clamp=True)
            self.save_sampled_images(samples_images, captions, batch_idx, "ema")
            
        try:
            val_score = self.val_score(samples_images.clamp(-1, 1), images, captions)
        except RuntimeError as e:
            val_score = {"score": 0}
            print(f"Error with Score:  {e}")
        
        self.validation_step_outputs.append(val_score)
        return val_score
    
    def on_validation_epoch_end(self):
        """
        Saves a checkpoint if the validation score is lower than the previous checkpoint
        and the current epoch is a multiple of checkpoint_every_val_epochs. Also
        saves a checkpoint at the end of every validation epoch which overwrites the previous one.
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
        Saves the sampled images and captions to a folder.

        :param sampled_images: Sampled images.
        :param captions: Captions for the sampled images.
        :param batch_idx: Batch index.
        :param note: Additional note to add to the folder name, defaults to None
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
        Configures the optimizers and learning rate schedulers for the training process.

        :return: Dictionary with the optimizer and the learning rate scheduler.
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