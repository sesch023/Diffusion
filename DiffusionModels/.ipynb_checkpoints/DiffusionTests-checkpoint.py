import torch
import torch.nn as nn
from torchinfo import summary
import math
import clip
from abc import ABC, abstractmethod
from Diffusion import *

def test_unet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicUNet().to(device)
    batch_size = 16
    summary(model, input_size=[(batch_size, 3, 64, 64), (batch_size, 256), (batch_size, 512)], device=device)
    
def test_timestep_encoding():
    import matplotlib.pyplot as plt
    import numpy as np

    limit = 1000
    pos_enc = []

    for i in range(limit):
        enc = DiffusionTools().timestep_encoding(i, 128)
        enc_copy = enc.clone()

        for i in range(0, 128, 2):
            enc[i] = enc_copy[i//2]
            enc[i+1] = enc_copy[i//2+64]

        pos_enc.append(enc)

    pos_enc = torch.stack(pos_enc).numpy()
    fig, ax = plt.subplots()

    c = ax.pcolor(pos_enc)
    ax.set_title('Diffusion Timestep Encoding')

    plt.savefig("out.png")
    
    
def test_noise_images():
    from PIL import Image
    import requests

    urls = ['http://images.cocodataset.org/val2017/000000039769.jpg', "http://images.cocodataset.org/train2017/000000218589.jpg"]
    images =[]
    for url in urls:
        images.append(Image.open(requests.get(url, stream=True).raw))
        
    from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

    image_size = 512
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
        Lambda(lambda t: (t * 2) - 1),

    ])
    
    import numpy as np
    
    out_size = 512
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
        Resize(out_size)
    ])

    images = torch.stack([transform(image) for image in images])
    
    steps = 300
    step = steps // 10
    dt = DiffusionTools(noise_scheduler=LinearScheduler(), steps = steps)
    
    outs = []
    for t in range(0, steps, step):
        ts = torch.Tensor([t, t+step//2]).long()
        outs.append(dt.noise_images(images, ts)[0])
        
    for i in range(len(outs)):
        for x in range(outs[i].shape[0]):
            reverse_transform(outs[i][x]).save(f"outs/noise/{i}_{x}_lin.jpg")
            
    dt = DiffusionTools(noise_scheduler=CosineScheduler(), steps = steps)
    
    outs = []
    for t in range(0, steps, step):
        ts = torch.Tensor([t, t+step//2]).long()
        outs.append(dt.noise_images(images, ts)[0])
        
    for i in range(len(outs)):
        for x in range(outs[i].shape[0]):
            reverse_transform(outs[i][x]).save(f"outs/noise/{i}_{x}_cos.jpg")
            
test_noise_images()
