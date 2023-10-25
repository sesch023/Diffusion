import sys
import os
sys.path.append("../")
import torch
import math

from DiffusionModules.DataModules import CIFAR10DataModule
from DiffusionModules.ModelLoading import load_cf10
from DiffusionModules.EmbeddingTools import ClipTools, ClipTextEmbeddingProvider
from DiffusionModules.Util import calculate_mean

"""
Tests for a cifar10 diffusion model.

The results of this model were described in the chapter:
7.1. Diffusion mit dem CIFAR-10 Datensatz
"""

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
report_path = "CF10_report_2/"
path = ModelLoadConfig.cf10_diffusion_path
batch_size = 16
start_n = 0
n = 1000

if not os.path.exists(report_path):
    os.makedirs(report_path)

# Load the model
model = load_cf10(path, device)
model.sample_images_out_base_path = report_path

scores = []

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, samples_per_caption=1):
    """
    Sample from the CIFAR10 diffusion model.

    :param trainer: Trainer instance to sample from.
    :param captions: Captions for Class-Embeddings. Also used for calculating the CLIP score.
    :param images: Original Images for getting the shape, FID and saving the original images to disk.
    :param device: Device to use.
    :param batch_idx: Batch index for saving the images.
    :param samples_per_caption: How many samples should be generated per caption.
    """    
    images = trainer.transformable_data_module.transform_batch(images).to(device)
    image_shape = images.shape
    embs = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)

    for i in range(samples_per_caption):
        sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale)
        score = trainer.val_score(sampled_images, images, captions)
        scores.append({key: value.item() for key, value in score.items()})
        trainer.save_sampled_images(sampled_images, captions, batch_idx, f"{str(i)}_no_up", no_upscale=True)
        trainer.save_sampled_images(images, captions, batch_idx, f"{str(i)}_no_up_real", no_upscale=True)

    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(f"Mean FID: {mean_fid}")
    print(f"Mean CLIP score: {mean_clip}")

# Get a test dataloader
vd = CIFAR10DataModule(batch_size=batch_size, num_workers=4).test_dataloader()
limit_batches = math.ceil(n/batch_size)
i = start_n

# Sample from the model
for images, captions in vd:
    sample_from_diffusion_trainer(model, captions, images, device, i)
    i += 1
    print(f"Batch {i} of {limit_batches} done.")
    if i >= limit_batches:
        break

# Write the scores to a file
with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    f.write(f"Mean FID: {mean_fid}\n")
    f.write(f"Mean CLIP score: {mean_clip}")
