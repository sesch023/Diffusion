import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_cf10

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "CF10_report/"
path = "../samples_cifar_final/4499_model.ckpt"
model = load_cf10(path, device)
model.sample_images_out_base_path = report_path

def sample_from_diffusion_trainer(trainer, captions, device, samples_per_caption=10):
    image_shape = (len(captions), 3, 64, 64)
    embs = trainer.alt_validation_emb_provider.get_embedding(None, captions).to(device)

    for i in range(samples_per_caption):
        sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale)
        trainer.save_sampled_images(sampled_images, captions, i, "up")
        trainer.save_sampled_images(sampled_images, captions, i, "no_up", no_upscale=True)


sample_from_diffusion_trainer(model, CIFAR10DataModule.classes, device)