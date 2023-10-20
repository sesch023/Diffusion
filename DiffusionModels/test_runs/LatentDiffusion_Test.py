import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
from DiffusionModules.ModelLoading import load_latent_diffusion

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "LatentDiffusion_report/"
path = "../samples_laten_diffusion/7999_model.ckpt"
vqgan_path = "../vqgan.ckpt"
model = load_latent_diffusion(path, vqgan_path, device, alt_prov_mode="TEXT")
model.sample_images_out_base_path = report_path
batch_size = 4
start_n = 112
n = 1000

scores = []
scores_translator = []
scores_text = []

def calculate_mean(items, key):
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)

def sample(trainer, latent_shape, embs, captions, images, batch_idx, note):
    sampled_latents = trainer.diffusion_tools.sample_data(trainer.ema_unet, latent_shape, embs, trainer.cfg_scale)
    sampled_latents = trainer.codebook_norm_tensor(sampled_latents)
    if trainer.quantize_after_sample:
        sampled_latents, _, _ = trainer.vqgan.quantize(sampled_latents)
    sampled_images = trainer.vqgan.decode(sampled_latents, emb=embs, clamp=True)
    score = trainer.val_score(sampled_images, images, captions)
    trainer.save_sampled_images(sampled_images, captions, batch_idx, note)
    return score

def sample_from_diffusion_trainer(trainer, captions, images, device, batch_idx, translator_emb_provider):
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    embs_translator = translator_emb_provider.get_embedding(images, captions).to(device)
    embs_text = trainer.alt_validation_emb_provider.get_embedding(images, captions).to(device)

    images = trainer.transformable_data_module.transform_batch(images).to(device)
    latent_batch_shape = (images.shape[0], *trainer.latent_shape)

    score_translator = sample(trainer, latent_batch_shape, embs_translator, captions, images, batch_idx, "translator")
    scores_translator.append({key: value.item() for key, value in score_translator.items()})

    score = sample(trainer, latent_batch_shape, embs, captions, images, batch_idx, "img_emb")
    scores.append({key: value.item() for key, value in score.items()})

    score_text = sample(trainer, latent_batch_shape, embs_text, captions, images, batch_idx, "text_emb")
    scores_text.append({key: value.item() for key, value in score_text.items()})

    trainer.save_sampled_images(images, captions, batch_idx, "real")

    mean_fid = calculate_mean(scores, "fid_score")
    mean_clip = calculate_mean(scores, "clip_score")
    print(f"Mean FID: {mean_fid}")
    print(f"Mean CLIP score: {mean_clip}")

    mean_fid = calculate_mean(scores_translator, "fid_score")
    mean_clip = calculate_mean(scores_translator, "clip_score")
    print(f"Mean Translator FID: {mean_fid}")
    print(f"Mean Translator CLIP score: {mean_clip}")

    mean_fid = calculate_mean(scores_text, "fid_score")
    mean_clip = calculate_mean(scores_text, "clip_score")
    print(f"Mean Text FID: {mean_fid}")
    print(f"Mean Text CLIP score: {mean_clip}")

dl = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00041..00059}.tar"],
    batch_size=batch_size,
    num_workers=4
).test_dataloader()

clip_tools = ClipTools(device=device)
translator_model_path = "../clip_translator/model.ckpt"
text_emb_provider = ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)

limit_batches = n//batch_size + 1
i = start_n

for images, captions in dl:
    sample_from_diffusion_trainer(model, captions, images, device, i, text_emb_provider)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

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
