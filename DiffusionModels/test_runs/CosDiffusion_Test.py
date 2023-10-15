import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_wdm

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "CosDiffusion_report/"
path = "../samples_cos_diffusion/3999_model.ckpt"
model = load_wdm(path, device, alt_prov_mode="TEXT")
model.sample_images_out_base_path = report_path

def sample_from_diffusion_trainer(trainer, captions, images, device, samples_per_item=10):
    image_shape = (len(captions), 3, 64, 64)
    embs_translator = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)

    for i in range(samples_per_item):
        sampled_images_translator = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs_translator, trainer.cfg_scale)
        sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale)
        trainer.save_sampled_images(sampled_images_translator, captions, i, "up_translator")
        trainer.save_sampled_images(sampled_images_translator, captions, i, "no_up_translator", no_upscale=True)
        trainer.save_sampled_images(sampled_images, captions, i, "up")
        trainer.save_sampled_images(sampled_images, captions, i, "no_up", no_upscale=True)

dl = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    batch_size=4,
    num_workers=4
).val_dataloader()
images, captions = next(iter(dl))

sample_from_diffusion_trainer(model, captions, images, device)