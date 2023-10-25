import sys
import os
sys.path.append("../")

import torch

from DiffusionModules.DataModules import WebdatasetDataModule
from DiffusionModules.ModelLoading import load_vqgan
from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
from Configs import ModelLoadConfig, DatasetLoadConfig, RunConfig

gpus=[0]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"
report_path = "VQGAN_report_3/"
batch_size = 4
start_n = 0
n = 1000

if not os.path.exists(report_path):
    os.makedirs(report_path)

load_path = ModelLoadConfig.vqgan_path
model = load_vqgan(load_path, device=device)
model.reconstructions_out_base_path = report_path


scores = []
scores_text = []

def calculate_mean(items, key):
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)

def forward_vqgan(trainer, captions, images, device, batch_idx, text_emb_provider):
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    text_embs = text_emb_provider.get_embedding(images, captions).to(device)
    images = trainer.transformable_data_module.transform_batch(images).to(device)

    reconstructions, qloss = trainer(images, emb=embs)

    aeloss, log_dict_ae = trainer.loss(qloss, images, reconstructions, 0, trainer.global_step, last_layer=trainer.get_last_layer(), split="test")
    scores.append(log_dict_ae)

    reconstructions_text, qloss = trainer(images, emb=text_embs)
    aeloss, log_dict_ae = trainer.loss(qloss, images, reconstructions_text, 0, trainer.global_step, last_layer=trainer.get_last_layer(), split="test")
    scores_text.append(log_dict_ae)

    trainer.save_reconstructions(reconstructions, batch_idx, captions=captions, note="recons")
    trainer.save_reconstructions(reconstructions_text, batch_idx, captions=captions, note="recons_text")
    trainer.save_reconstructions(images, batch_idx, captions=captions, note="orig")

    mean_total = calculate_mean(scores, "test/total_loss")
    print(f"Mean Total Loss: {mean_total}")
    mean_rec = calculate_mean(scores, "test/rec_loss")
    print(f"Mean Rec Loss: {mean_rec}")
    mean_total_text = calculate_mean(scores_text, "test/total_loss")
    print(f"Mean Total Loss Text: {mean_total_text}")
    mean_rec_text = calculate_mean(scores_text, "test/rec_loss")
    print(f"Mean Rec Loss Text: {mean_rec_text}")

dm = WebdatasetDataModule(
    DatasetLoadConfig.cc_3m_12m_paths,
    DatasetLoadConfig.coco_val_path,
    DatasetLoadConfig.coco_test_path,
    batch_size=batch_size,
    num_workers=4,
    img_in_target_size=256
)
dl = dm.test_dataloader()

clip_tools = ClipTools(device=device)
text_emb_provider = ClipTextEmbeddingProvider(clip_tools=clip_tools)
limit_batches = n//batch_size + 1
i = start_n

for images, captions in dl:
    forward_vqgan(model, captions, images, device, i, text_emb_provider)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

with open(f"{report_path}/scores.txt", "w") as f:
    mean_total = calculate_mean(scores, "test/total_loss")
    f.write(f"Mean Total Loss: {mean_total}")
    mean_rec = calculate_mean(scores, "test/rec_loss")
    f.write(f"Mean Rec Loss: {mean_rec}")
    mean_total_text = calculate_mean(scores_text, "test/total_loss")
    f.write(f"Mean Total Loss Text: {mean_total_text}")
    mean_rec_text = calculate_mean(scores_text, "test/rec_loss")
    f.write(f"Mean Rec Loss Text: {mean_rec_text}")

