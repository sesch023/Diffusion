from glob import glob
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

real_paths = sorted(glob("UpscaleDiffusion_report_2/*_low_res/*.png"))
fake_paths = sorted(glob("UpscaleDiffusion_report_2/*_up_no_emb/*.png"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fid = FrechetInceptionDistance(feature=2048).to(device)
convert_tensor = transforms.ToTensor()

for real, fake in tqdm(zip(real_paths, fake_paths), total=len(real_paths)):
    print(real, fake)
    fake = convert_tensor(Image.open(fake).convert("RGB")).unsqueeze(0).to(device)
    real = convert_tensor(Image.open(real).convert("RGB")).unsqueeze(0).to(device)

    fid.update((fake*255).byte(), real=False)
    fid.update((real*255).byte(), real=True)

fid_v = fid.compute()
print(fid_v)