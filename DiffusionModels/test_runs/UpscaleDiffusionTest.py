import sys
import os
import math
sys.path.append("../")

import torch

from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.ModelLoading import load_udm
from DiffusionModules.EmbeddingTools import ClipTools, ClipTextEmbeddingProvider
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig
from DiffusionModules.Util import calculate_mean

"""
Tests for a upscaler diffusion model with linear scheduling.

The results of this model were described in the chapter:
7.3. Upscaling mittels Diffusion
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
start_n = 0
n = 1000
batch_size = 4
low_res_embs = True
report_path = "UpscaleDiffusion_report/"

if not os.path.exists(report_path):
    os.makedirs(report_path)

# Load the model
path = ModelLoadConfig.upscaler_model_path
model = load_udm(path, device, upscale_size=256)
model.sample_images_out_base_path = report_path

scores = []
scores_text = []
scores_low_res = []
scores_no_emb = []

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, text_emb_provider, low_res_embs=True):
    """
    Upscale with the upscaler diffusion model
    using different embeddings.

    :param trainer: Trainer instance to upscale from.
    :param captions: Captions for Text-Embeddings. 
    :param images: Original Images in high res. Used for getting Image-Embeddings, FID, Low-Res-Images, Low-Res-Embedding 
                   and saving the original images to disk.
    :param device: Device to use.
    :param batch_idx: Batch index for saving the images.
    :param text_emb_provider: Text-Embedding provider for Text-Embeddings.
    :param low_res_embs: Whether to upscale with low res embeddings or not.
    """    
    # Create low res images
    low_res = torch.stack([trainer.transform_low_res(image).to(device) for image in images])  

    # If low_res_embs is True, create low res embeddings
    if low_res_embs:
        low_res_image = trainer.transformable_data_module.reverse_transform_batch(low_res.cpu())
        low_embs = trainer.embedding_provider.get_embedding(low_res_image, captions).to(device)
    
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    text_embs = text_emb_provider.get_embedding(images, captions).to(device)

    images = trainer.transformable_data_module.transform_batch(images).to(device)
    image_shape = images.shape

    # Upscale with the upscaler diffusion model using no embeddings.
    sampled_images_no_emb = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, None, trainer.cfg_scale, x_appendex=low_res)
    score_no_emb = trainer.val_score(sampled_images_no_emb, images, captions)
    scores_no_emb.append({key: value.item() for key, value in score_no_emb.items()})

    # Upscale with the upscaler diffusion model using image embeddings.
    sampled_images = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, embs, trainer.cfg_scale, x_appendex=low_res)
    score = trainer.val_score(sampled_images, images, captions)
    scores.append({key: value.item() for key, value in score.items()})

    # Upscale with the upscaler diffusion model using text embeddings.
    sampled_images_text = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, text_embs, trainer.cfg_scale, x_appendex=low_res)
    score_text = trainer.val_score(sampled_images_text, images, captions)
    scores_text.append({key: value.item() for key, value in score_text.items()})

    # Upscale with the upscaler diffusion model using low res embeddings, if low_res_embs is True. Also save the sampled images.
    if low_res_embs:
        sampled_images_low_res = trainer.diffusion_tools.sample_data(trainer.ema_unet, image_shape, low_embs, trainer.cfg_scale, x_appendex=low_res)
        score_low_res = trainer.val_score(sampled_images_low_res, images, captions)
        scores_low_res.append({key: value.item() for key, value in score_low_res.items()})
        trainer.save_sampled_images(sampled_images_low_res, captions, batch_idx, "up_low_emb")

    # Save the sampled results.
    trainer.save_sampled_images(sampled_images_text, captions, batch_idx, "up_text_emb")
    trainer.save_sampled_images(sampled_images, captions, batch_idx, "up_img_emb")
    trainer.save_sampled_images(sampled_images_no_emb, captions, batch_idx, "up_no_emb")
    trainer.save_sampled_images(low_res, captions, batch_idx, "low_res")
    trainer.save_sampled_images(images, captions, batch_idx, "real")

    # Calculate mean fid of upscaled images with high res image embeddings.
    mean_fid = calculate_mean(scores, "fid_score")
    print(f"Mean FID: {mean_fid}")
    # Calculate mean fid of upscaled images with text embeddings.
    mean_fid = calculate_mean(scores_text, "fid_score")
    print(f"Mean FID Text: {mean_fid}")
    # Calculate mean fid of upscaled images with no embeddings.
    mean_fid = calculate_mean(scores_no_emb, "fid_score")
    print(f"Mean FID No Emb: {mean_fid}")

    # Calculate mean fid of upscaled images with low res embeddings, if low_res_embs is True.
    if low_res_embs:
        mean_fid = calculate_mean(scores_low_res, "fid_score")
        print(f"Mean FID Low Res: {mean_fid}")

dl = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    num_workers=4,
    img_in_target_size=256
).test_dataloader()

clip_tools = ClipTools(device=device)
text_emb_provider = ClipTextEmbeddingProvider(clip_tools=clip_tools)

limit_batches = math.ceil(n/batch_size)
i = start_n

# Sample from the upscaler diffusion model with different embeddings.
for images, captions in dl:
    sample_from_diffusion_trainer(model, captions, images, device, i, text_emb_provider, low_res_embs)
    i += 1
    print(f"Batch {i} of {limit_batches} done.")
    if i >= limit_batches:
        break

# Save the mean fids to a file.
with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid = calculate_mean(scores, "fid_score")
    f.write(f"Mean FID: {mean_fid}")
    mean_fid = calculate_mean(scores_text, "fid_score")
    f.write(f"Mean FID Text: {mean_fid}")
    mean_fid = calculate_mean(scores_no_emb, "fid_score")
    f.write(f"Mean FID No Emb: {mean_fid}")
    if low_res_embs:
        mean_fid = calculate_mean(scores_low_res, "fid_score")
        f.write(f"Mean FID Low Res: {mean_fid}")