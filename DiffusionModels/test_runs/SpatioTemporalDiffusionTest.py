import sys
import os
import math
sys.path.append("../")

import torch
from einops import rearrange

from DiffusionModules.DataModules import VideoDatasetDataModule, WebdatasetDataModule
from DiffusionModules.ModelLoading import load_spatio_temporal
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

"""
Tests for a spatio-temporal diffusion model.

The results of this model were described in the chapter:
7.5. Spatiotemporal Decoder des Make-A-Video-Systems
"""
gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
start_n = 0
n = 200
batch_size = 2
report_path = "SpatioTemporalDiffusion_report_2/"

if not os.path.exists(report_path):
    os.makedirs(report_path)

# Load the model
path = ModelLoadConfig.spatio_temporal_diffusion_path
model = load_spatio_temporal(path, device, False)
model.sample_data_out_base_path = report_path

scores = []

def sample_from_diffusion_trainer(trainer, captions, videos, device, batch_idx, fps, temporal_dm, non_temporal_dm):
    """
    Sample from the spatio-temporal diffusion model.

    :param trainer: Trainer instance to sample from.
    :param captions: Captions for Text-Embeddings. Also used for calculating the CLIP score.
    :param videos: Original Videos for getting the shape, FVD and saving the original videos to disk.
    :param device: Device to use.
    :param batch_idx: Batch index for saving the videos.
    :param fps: FPS of the videos.
    :param temporal_dm: Temporal DataModule for temporal transformations.
    :param non_temporal_dm: Non-Temporal DataModule for non-temporal outputs of frames.
    """    
    trainer.transformable_data_module = temporal_dm
    embs = trainer.temporal_embedding_provider.get_embedding(videos, captions).to(device)
    videos = trainer.transformable_data_module.transform_batch(videos).to(device)
    # Get the fps embeddings
    f_emb = torch.stack([trainer.diffusion_tools.get_pos_encoding(f) for f in fps]).to(device) 
    videos_shape = videos.shape

    # Sample from the spatio-temporal diffusion model.
    sampled_videos = trainer.diffusion_tools.sample_data(trainer.ema_unet, videos_shape, embs, trainer.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=True)
    score = trainer.val_score(sampled_videos.detach(), videos.detach(), captions)
    score["clip_score"] = score["clip_score"].item()
    scores.append(score)

    # Save the sampled videos to disk.
    trainer.save_sampled_data(sampled_videos, captions, batch_idx, "fake")
    trainer.save_sampled_data(videos, captions, batch_idx, "real")

    # Reshape the videos to frames and set the non-temporal DataModule.
    trainer.transformable_data_module = non_temporal_dm
    videos_frame = rearrange(videos, "b c t h w -> (b t) c h w")
    sampled_videos_frame = rearrange(sampled_videos, "b c t h w -> (b t) c h w")

    # Save the sampled videos as frames to disk.
    trainer.save_sampled_data(sampled_videos_frame, captions, batch_idx, "fake_frame")
    trainer.save_sampled_data(videos_frame, captions, batch_idx, "real_frame")
    trainer.transformable_data_module = temporal_dm

    # Calculate the mean FVD score and CLIP score.
    mean_fid_fvd = calculate_mean(scores, "fid_fvd_score")
    mean_clip = calculate_mean(scores, "clip_score")
    # Print the mean FVD score and CLIP score.
    print(f"Mean FVD: {mean_fid_fvd}")
    print(f"Mean CLIP: {mean_clip}")

# Temporal DataModule for temporal transformations.
temporal_dataset = VideoDatasetDataModule(
    None,
    None,
    DatasetLoadConfig.webvid_10m_val_csv,
    DatasetLoadConfig.webvic_10m_val_data,
    DatasetLoadConfig.webvid_10m_val_csv,
    DatasetLoadConfig.webvic_10m_val_data,
    batch_size=batch_size,
    num_workers=4,
    nth_frames=1,
    max_frames_per_part=16,
    min_frames_per_part=4,
    first_part_only=True
)

# Non-Temporal DataModule for non-temporal outputs of frames. No data is loaded.
dm = WebdatasetDataModule(
    [""],
    [""],
    batch_size=batch_size,
    num_workers=1
)  
model.transformable_data_module = temporal_dataset
model.writer_module = temporal_dataset.v_data
dl = temporal_dataset.val_dataloader()

limit_batches = math.ceil(n / batch_size)
i = start_n

# Sample from the spatio-temporal diffusion model.
for videos, captions, lengths, fps in dl:
    sample_from_diffusion_trainer(model, captions, videos, device, i, fps, temporal_dataset, dm)
    i += 1
    print(f"Batch {i} of {limit_batches} done.")
    if i >= limit_batches:
        break

# Save the scores to disk.
with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid_fvd = calculate_mean(scores, "fid_fvd_score")
    mean_clip = calculate_mean(scores, "clip_score")
    f.write(f"Mean FVD: {mean_fid_fvd}\n")
    f.write(f"Mean CLIP: {mean_clip}\n")