import sys
import os
import math
sys.path.append("../")

import torch

from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools, ClipTextEmbeddingProvider
from DiffusionModules.ModelLoading import load_wdm
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig
from DiffusionModules.Util import calculate_mean

"""
Tests for a diffusion model with cosine scheduling.

The results of this model were described in the chapter:
7.2. Diffusion mit Cosine-Schedule
"""

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "CosDiffusion_report/"
path = ModelLoadConfig.diffusion_path
batch_size = 4
start_n = 0
n = 1000

# Load the model
model = load_wdm(path, device, alt_prov_mode="TRANSLATOR")
model.sample_images_out_base_path = report_path

if not os.path.exists(report_path):
    os.makedirs(report_path)

scores = []
scores_translator = []
scores_text = []

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, text_emb_provider):
    """
    Sample from the diffusion model with cosine scheduling.

    :param trainer: Trainer instance to sample from.
    :param captions: Captions for Text- and Translator-Embeddings. Also used for calculating the CLIP score.
    :param images: Original Images for getting the shape, FID and saving the original images to disk. 
                   Also to provides Image-Embeddings.
    :param device: Device to use.
    :param batch_idx: Batch index for saving the images.
    :param text_emb_provider: Text-Embedding provider for Text-Embeddings.
    """    
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    embs_translator = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)
    embs_text = text_emb_provider.get_embedding(images, captions).to(device)
    images = trainer.transformable_data_module.transform_batch(images).to(device)
    image_shape = images.shape

    # Sample from the diffusion model with cosine scheduling and translator embeddings.
    sampled_images_translator = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs_translator, trainer.cfg_scale)
    score_translator = trainer.val_score(sampled_images_translator, images, captions)
    scores_translator.append({key: value.item() for key, value in score_translator.items()})

    # Sample from the diffusion model with cosine scheduling and image embeddings.
    sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale)
    score = trainer.val_score(sampled_images, images, captions)
    scores.append({key: value.item() for key, value in score.items()})

    # Sample from the diffusion model with cosine scheduling and text embeddings.
    sampled_images_text = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs_text, trainer.cfg_scale)
    score_text = trainer.val_score(sampled_images_text, images, captions)
    scores_text.append({key: value.item() for key, value in score_text.items()})

    trainer.save_sampled_images(sampled_images_translator, captions, batch_idx, "translator", no_upscale=True)
    trainer.save_sampled_images(sampled_images, captions, batch_idx, "img_emb", no_upscale=True)
    trainer.save_sampled_images(sampled_images_text, captions, batch_idx, "text_emb", no_upscale=True)
    trainer.save_sampled_images(images, captions, batch_idx, "real", no_upscale=True)

    # Calculate the mean FID and CLIP score for image-embs.
    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(scores)
    print(f"Mean FID: {mean_fid}")
    print(f"Mean CLIP score: {mean_clip}")

    # Calculate the mean FID and CLIP score for translator-embs.
    mean_fid = calculate_mean(scores_translator, "fid_score")
    mean_clip = calculate_mean(scores_translator, "clip_score")
    print(scores_translator)
    print(f"Mean Translator FID: {mean_fid}")
    print(f"Mean Translator CLIP score: {mean_clip}")

    # Calculate the mean FID and CLIP score for text-embs.
    mean_fid = calculate_mean(scores_text, "fid_score")
    mean_clip = calculate_mean(scores_text, "clip_score")
    print(scores_text)
    print(f"Mean Text FID: {mean_fid}")
    print(f"Mean Text CLIP score: {mean_clip}")

dl = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    num_workers=4
).test_dataloader()

clip_tools = ClipTools(device=device)
text_emb_provider = ClipTextEmbeddingProvider(clip_tools=clip_tools)

limit_batches = math.ceil(n/batch_size)
i = start_n

# Sample from the diffusion model with cosine scheduling.
for images, captions in dl:
    sample_from_diffusion_trainer(model, captions, images, device, i, text_emb_provider)
    i += 1
    print(f"Batch {i} of {limit_batches} done.")
    if i >= limit_batches:
        break

# Save the scores
with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    f.write(f"Mean FID: {mean_fid}\n")
    f.write(f"Mean CLIP score: {mean_clip}\n")

    mean_fid = calculate_mean(scores_translator, "fid_score")
    mean_clip = calculate_mean(scores_translator, "clip_score")
    f.write(f"Mean Translator FID: {mean_fid}\n")
    f.write(f"Mean Translator CLIP score: {mean_clip}\n")

    mean_fid = calculate_mean(scores_text, "fid_score")
    mean_clip = calculate_mean(scores_text, "clip_score")
    f.write(f"Mean Text FID: {mean_fid}\n")
    f.write(f"Mean Text CLIP score: {mean_clip}\n")
