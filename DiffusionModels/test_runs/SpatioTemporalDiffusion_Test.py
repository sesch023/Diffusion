import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_spatio_temporal

from einops import rearrange

gpus=[2]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "SpatioTemporalDiffusion_report/"
path = "../samples_spatio_temporal/10799_model.ckpt"
model = load_spatio_temporal(path, device)
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

def sample_from_diffusion_trainer(trainer, captions, videos, device, batch_idx, fps):
    embs = trainer.temporal_embedding_provider.get_embedding(videos, captions).to(device)
    videos = trainer.transformable_data_module.transform_batch(videos).to(device)
    f_emb = torch.stack([trainer.diffusion_tools.get_pos_encoding(f) for f in fps]).to(device) 
    videos_shape = videos.shape
    sampled_videos = trainer.diffusion_tools.sample_data(trainer.ema_unet, videos_shape, embs, trainer.cfg_scale, clamp_var=True, f_emb=f_emb, temporal=True)
    score = trainer.val_score(sampled_videos, videos, captions)

    scores.append(score)

    trainer.save_sampled_data(sampled_videos, captions, batch_idx, "fake")
    trainer.save_sampled_data(videos, captions, batch_idx, "real")

    videos_frame = rearrange(videos, "b t c h w -> (b t) c h w")
    sampled_videos_frame = rearrange(sampled_videos, "b t c h w -> (b t) c h w")

    trainer.save_sampled_data(sampled_videos_frame, captions, batch_idx, "fake_frame")
    trainer.save_sampled_data(videos_frame, captions, batch_idx, "real_frame")

    mean_fid_fvd = calculate_mean(scores, "fid_fvd_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(f"Mean FVD: {mean_fid_fvd}")
    print(f"Mean CLIP: {mean_clip}")

temporal_dataset = VideoDatasetDataModule(
    None,
    None,
    "/home/shared-data/webvid/results_10M_val.csv", 
    "/home/shared-data/webvid/data_val/videos",
    "/home/shared-data/webvid/results_10M_val.csv", 
    "/home/shared-data/webvid/data_val/videos",
    batch_size=batch_size,
    num_workers=4,
    nth_frames=1,
    max_frames_per_part=16,
    min_frames_per_part=4,
    first_part_only=True
)
dl = temporal_dataset.val_dataloader()

limit_batches = n//batch_size + 1
i = start_n

for videos, captions, lengths, fps in dl:
    sample_from_diffusion_trainer(model, captions, videos, device, i, fps)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

with open(f"{report_path}/scores.txt", "w") as f:
    mean_fid_fvd = calculate_mean(scores, "fid_fvd_score")
    mean_clip = calculate_mean(scores, "clip_score")
    f.write(f"Mean FVD: {mean_fid_fvd}\n")
    f.write(f"Mean CLIP: {mean_clip}\n")