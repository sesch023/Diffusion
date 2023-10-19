import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_udm

gpus=[2]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "UpscaleDiffusion_report/"
path = "~/upscaler.ckpt"
model = load_udm(path, device, upscale_size=256)
model.sample_images_out_base_path = report_path
start_n = 0
n = 1000
batch_size = 4

scores = []

def calculate_mean(items, key):
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx):
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    low_res = torch.stack([trainer.transform_low_res(image).to(device) for image in images])  
    images = trainer.transformable_data_module.transform_batch(images).to(device)
    image_shape = images.shape

    sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale, x_appendex=low_res)
    score = trainer.val_score(sampled_images, images, captions)
    scores.append(score)
    trainer.save_sampled_images(sampled_images, captions, batch_idx, f"{str(i)}_up")
    trainer.save_sampled_images(low_res, captions, batch_idx, f"{str(i)}_low_res")
    trainer.save_sampled_images(images, captions, batch_idx, f"{str(i)}_real")

    mean_fid = calculate_mean(scores, "fid_score")
    print(scores)
    print(f"Mean FID: {mean_fid}")

dl = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00041..00059}.tar"],
    batch_size=batch_size,
    num_workers=4,
    img_in_target_size=256
).test_dataloader()

limit_batches = n//batch_size + 1
i = start_n

for images, captions in dl:
    sample_from_diffusion_trainer(model, captions, images, device, i)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid = calculate_mean(scores, "fid_score")
    f.write(f"Mean FID: {mean_fid}")