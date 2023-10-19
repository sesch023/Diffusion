import sys
sys.path.append("../")

import torch

from DiffusionModules.Diffusion import *
from DiffusionModules.DiffusionTrainer import *
from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *
from DiffusionModules.ModelLoading import load_vqgan

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

report_path = "VQGAN_report/"
load_path = "../vqgan.ckpt"
model = load_vqgan(load_path, device=device)
model.reconstructions_out_base_path = report_path
batch_size = 4
start_n = 0
n = 1000

scores = []

def calculate_mean(items, key):
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)

def forward_vqgan(trainer, captions, images, device, batch_idx):
    embs = trainer.embedding_provider.get_embedding(images, captions).to(device)
    images = trainer.transformable_data_module.transform_batch(images).to(device)

    reconstructions, qloss = trainer(images, emb=embs)

    aeloss, log_dict_ae = trainer.loss(qloss, images, reconstructions, 0, trainer.global_step, last_layer=trainer.get_last_layer(), split="test")
    scores.append(log_dict_ae)

    trainer.save_reconstructions(reconstructions, batch_idx, captions=captions, note="recons")
    trainer.save_reconstructions(images, batch_idx, captions=captions, note="orig")

    mean_total = calculate_mean(scores, "test/total_loss")
    print(f"Mean Total Loss: {mean_total}")

dm = WebdatasetDataModule(
    ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"],
    ["/home/archive/CocoWebdatasetFullScale/mscoco/{00041..00059}.tar"],
    batch_size=batch_size,
    num_workers=4,
    img_in_target_size=256
)
dl = dm.test_dataloader()

limit_batches = n//batch_size + 1
i = start_n

for images, captions in dl:
    forward_vqgan(model, captions, images, device, i)
    print(f"Batch {i} of {limit_batches - 1} done.")
    i += 1
    if i >= limit_batches:
        break

with open(f"{report_path}/scores.txt", "w") as f:
    mean_total = calculate_mean(scores, "test/total_loss")
    f.write(f"Mean Total Loss: {mean_total}")
