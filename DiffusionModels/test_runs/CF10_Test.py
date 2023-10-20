import sys
sys.path.append("../")

import torch
from functools import reduce

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_cf10


gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "CF10_report/"
path = "../samples_cifar_final/4499_model.ckpt"
model = load_cf10(path, device)
model.sample_images_out_base_path = report_path
batch_size = 16
start_n = 0
n = 1000

scores = []

def calculate_mean(items, key):
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, samples_per_caption=1):
    images = trainer.transformable_data_module.transform_batch(images).to(device)
    image_shape = images.shape
    embs = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)

    for i in range(samples_per_caption):
        sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale)
        score = trainer.val_score(sampled_images, images, captions)
        scores.append({key: value.item() for key, value in score.items()})
        # trainer.save_sampled_images(sampled_images, captions, batch_idx, f"{str(i)}_up")
        trainer.save_sampled_images(sampled_images, captions, batch_idx, f"{str(i)}_no_up", no_upscale=True)
        trainer.save_sampled_images(images, captions, batch_idx, f"{str(i)}_no_up_real", no_upscale=True)

    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(f"Mean FID: {mean_fid}")
    print(f"Mean CLIP score: {mean_clip}")

vd = CIFAR10DataModule(batch_size=batch_size, num_workers=4).test_dataloader()
limit_batches = n//batch_size + 1
i = start_n

for images, captions in vd:
    sample_from_diffusion_trainer(model, captions, images, device, i)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    f.write(f"Mean FID: {mean_fid}\n")
    f.write(f"Mean CLIP score: {mean_clip}")


