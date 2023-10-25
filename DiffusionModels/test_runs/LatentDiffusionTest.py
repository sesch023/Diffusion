import sys
import os
import math
sys.path.append("../")

import torch

from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.EmbeddingTools import ClipTools, ClipTranslatorEmbeddingProvider
from DiffusionModules.ModelLoading import load_latent_diffusion
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig
from DiffusionModules.Util import calculate_mean

"""
Tests for a latent diffusion model.

The results of this model were described in the chapter:
7.4. VQGANs und latente Diffusion
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
report_path = "LatentDiffusion_report_2/"
batch_size = 4
start_n = 0
n = 8

if not os.path.exists(report_path):
    os.makedirs(report_path)

# Load the model
path = ModelLoadConfig.latent_diffusion_path
vqgan_path = ModelLoadConfig.vqgan_path
model = load_latent_diffusion(path, vqgan_path, device, alt_prov_mode="TEXT")
model.sample_images_out_base_path = report_path

scores = []
scores_translator = []
scores_text = []

def sample(trainer, latent_shape, embs, captions, images, batch_idx, note):
    """
    Sample from the latent diffusion model with the given embeddings.

    :param trainer: Trainer instance to sample from.
    :param latent_shape: Shape of the latent space.
    :param embs: Embeddings to use for sampling.
    :param captions: Captions for outputs. Also used for calculating the CLIP score.
    :param images: Original Images for getting the FID and saving the original images to disk.
    :param batch_idx: Batch index for saving the images.
    :param note: Note for saving the images.
    :return: Scores of the sampled images.
    """    
    sampled_latents = trainer.diffusion_tools.sample_data(trainer.ema_unet, latent_shape, embs, trainer.cfg_scale)
    sampled_latents = trainer.codebook_norm_tensor(sampled_latents)
    if trainer.quantize_after_sample:
        sampled_latents, _, _ = trainer.vqgan.quantize(sampled_latents)
    sampled_images = trainer.vqgan.decode(sampled_latents, emb=embs, clamp=True)
    score = trainer.val_score(sampled_images, images, captions)
    trainer.save_sampled_images(sampled_images, captions, batch_idx, note)
    return score

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, translator_emb_provider):
    """
    Sample from the latent diffusion model.

    :param trainer: Trainer instance to sample from.
    :param captions: Captions for Text- and Translator-Embeddings. Also used for calculating the CLIP score.
    :param images: Original Images for getting the shape, FID and saving the images. Also to provided Image-Embeddings.
    :param device: Device to use.
    :param batch_idx: Batch index for saving the images.
    :param translator_emb_provider: Translator-Embedding provider for Translator-Embeddings.
    """    
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    embs_translator = translator_emb_provider.get_embedding(images, captions).to(device)
    embs_text = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)

    images = trainer.transformable_data_module.transform_batch(images).to(device)
    latent_batch_shape = (images.shape[0], *trainer.latent_shape)

    # Sample from the latent diffusion model with translator embeddings.
    score_translator = sample(trainer, latent_batch_shape, embs_translator, captions, images, batch_idx, "translator")
    scores_translator.append({key: value.item() for key, value in score_translator.items()})

    # Sample from the latent diffusion model with image embeddings.
    score = sample(trainer, latent_batch_shape, embs, captions, images, batch_idx, "img_emb")
    scores.append({key: value.item() for key, value in score.items()})

    # Sample from the latent diffusion model with text embeddings.
    score_text = sample(trainer, latent_batch_shape, embs_text, captions, images, batch_idx, "text_emb")
    scores_text.append({key: value.item() for key, value in score_text.items()})

    trainer.save_sampled_images(images, captions, batch_idx, "real")

    # Print the mean scores of image embeddings.
    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(f"Mean FID: {mean_fid}")
    print(f"Mean CLIP score: {mean_clip}")

    # Print the mean scores of translator embeddings.
    mean_fid = calculate_mean(scores_translator, "fid_score")
    mean_clip = calculate_mean(scores_translator, "clip_score")
    print(f"Mean Translator FID: {mean_fid}")
    print(f"Mean Translator CLIP score: {mean_clip}")

    # Print the mean scores of text embeddings.
    mean_fid = calculate_mean(scores_text, "fid_score")
    mean_clip = calculate_mean(scores_text, "clip_score")
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
translator_model_path = ModelLoadConfig.translator_model_path
translator_emb_provider = ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)

limit_batches = math.ceil(n/batch_size)
i = start_n

# Sample from the diffusion model.
for images, captions in dl:
    sample_from_diffusion_trainer(model, captions, images, device, i, translator_emb_provider)
    i += 1
    print(f"Batch {i} of {limit_batches} done.")
    if i >= limit_batches:
        break

# Save the scores.
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
