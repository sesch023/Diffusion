import torch
import torch.nn as nn
from torch import optim
import lightning.pytorch as pl

class ClipTranslator(nn.Module):    
    def __init__(self, in_out_dim=512, mid_dim=1024, num_mid=30, dropout=0.1): 
        """
        Creates a ClipTranslator model that takes in a clip text embedding and outputs an image embedding
        if the model is trained correctly. The model repetitively applies a linear layer, gelu activation, dropout, 
        linear layer, dropout, and residual connection. The number of times this is repeated is determined by num_mid.

        :param in_out_dim: Size of the input and output, defaults to 512
        :param mid_dim: Size of the output dimensions of the mid blocks, defaults to 1024
        :param num_mid: Number of repeated blocks, defaults to 30
        :param dropout: Probability of dropout, defaults to 0.1
        """        
        super().__init__()
        self._in_out_dim = in_out_dim
        self._in_layer = nn.Sequential(
            nn.Linear(in_out_dim, in_out_dim),
            nn.LayerNorm(in_out_dim)
        )
        self._sequential_mids = nn.ModuleList([nn.Sequential(
            nn.Linear(in_out_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mid_dim, in_out_dim),
            nn.Dropout(p=dropout)
        ) for i in range(num_mid)])
        
        self._out_layer = nn.Sequential(
            nn.LayerNorm(in_out_dim),
            nn.Linear(in_out_dim, in_out_dim)
        )
        
    def forward(self, x):
        """
        Forward pass of the model. Takes in a clip text embedding and outputs an image embedding.

        :param x: Clip text embedding.
        :return: Image embedding.
        """        
        current_out = self._in_layer(x)
        for layer in self._sequential_mids:
            current_out = layer(current_out) + current_out
        return self._out_layer(current_out)
      
        
class ClipTranslatorTrainer(pl.LightningModule):
    def __init__(self, model, device=None, loss=None, optimizer=None, clip_tools=None, model_out="clip_translator/model.ckpt"):
        """
        Creates a ClipTranslatorTrainer that trains a ClipTranslator model. The model is trained using a MSE loss function
        and given a default value an AdamW optimizer. The model is trained using a learning rate scheduler that reduces the
        learning rate by a factor of 0.5 if the validation loss does not improve for 5000 steps. The model is saved at the
        end of each epoch if the validation loss is lower than the previous best validation loss.

        :param model: A ClipTranslator model.
        :param device: Device to use, defaults to cuda if available else cpu.
        :param loss: Loss of the Trainer, defaults to nn.MSELoss()
        :param optimizer: Optimizer of the Trainer, defaults to optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
        :param clip_tools: Used ClipTools of the Trainer for interacting with Clip, defaults to ClipTools(device=self.dev)
        :param model_out: Output path of the training process, defaults to "clip_translator/model.ckpt"
        """        
        super().__init__()
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model.to(self.dev)
        
        from DiffusionModules.EmbeddingTools import ClipTools
        
        self.clip_tools = ClipTools(device=self.dev) if clip_tools is None else clip_tools
        self.clip_tools.eval()
        self.loss = nn.MSELoss() if loss is None else loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0001) if optimizer is None else optimizer
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_identity_outputs = []
        self.best_loss = float("inf")
        self.model_out = model_out
        self.save_hyperparameters()
    
    def eval_items(self, images, captions, ret_emb=False):
        """
        Evaluates the model on a batch of images and captions. Returns the loss of the model on the batch and optionally
        the image and caption embeddings.

        :param images: Images to evaluate on.
        :param captions: Captions to evaluate on.
        :param ret_emb: Returns the embeddings if true, defaults to False
        :return: Loss of the model on the batch and optionally the image and caption embeddings. 
                 A) (loss, cap_emb, img_emb) if ret_emb is True else B) loss.
        """        
        cap_emb = self.clip_tools.get_clip_emb_text(captions).to(self.dev) 
        img_emb = self.clip_tools.get_clip_emb_images(images).to(self.dev) 
        model_out = self.model(cap_emb.detach())
        loss = self.loss(img_emb.detach(), model_out)
        return (loss, cap_emb, img_emb) if ret_emb else loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step of the model. Evaluates the model on a batch of images and captions and returns the loss of the model.
        Also logs the loss and identity loss of the model every 100 steps.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Loss of the model on the batch.
        """        
        images, captions = batch
        loss, cap_emb, img_emb = self.eval_items(images, captions, True)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                identity_loss = self.loss(img_emb, cap_emb)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
            self.log("identity_loss", identity_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model. Evaluates the model on a batch of images and captions and returns the loss of the model.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Loss of the model on the batch.
        """        
        images, captions = batch
        loss = self.eval_items(images, captions)
        self.validation_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step of the model. Evaluates the model on a batch of images and captions and returns the loss of the model.
        Also calculates the identity loss of the model on the batch.

        :param batch: Batch of images and captions.
        :param batch_idx: Index of the batch.
        :return: Loss of the model on the batch.
        """        
        images, captions = batch
        loss, cap_emb, img_emb = self.eval_items(images, captions, True)  
        with torch.no_grad():
            identity_loss = self.loss(img_emb, cap_emb)
            self.test_step_identity_outputs.append(identity_loss)
        self.test_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        """
        Hook that is called at the end of the validation epoch. Calculates the average loss of the model on the validation set.
        Logs the average loss of the model on the validation set. If the average loss is lower than the previous best loss,
        the model is saved.
        """        
        avg_loss = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            print(f"Saving Checkpoint at: {self.model_out}")
            self.trainer.save_checkpoint(self.model_out)
        
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        """
        Hook that is called at the end of the test epoch. Calculates the average loss and identity loss of the model on the test set.
        Logs the average loss and identity loss of the model on the test set.
        """        
        avg_loss = sum(self.test_step_outputs) / len(self.test_step_outputs)
        avg_identity_loss = sum(self.test_step_identity_outputs) / len(self.test_step_identity_outputs)
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_id_loss", avg_identity_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()
        self.test_step_identity_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler of the model. The optimizer is an AdamW optimizer by default with a learning
        rate of 1e-3 and weight decay of 0.0001. The learning rate scheduler is a ReduceLROnPlateau scheduler that reduces the
        learning rate by a factor of 0.5 if the validation loss does not improve for 5000 steps.
        :return: Optimizer and learning rate scheduler of the model.
        """        
        lr = self.optimizer.param_groups[-1]['lr']
        sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5000, min_lr=lr/100)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss_step"
            }
        }
    