from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *

def load_udm(path, device, upscale_size=256):
    ds = WebdatasetDataModule(
        ["/home/archive/CC12M/cc12m/{00000..01242}.tar", "/home/archive/CC3M/cc3m/{00000..00331}.tar"],
        ["/home/archive/CocoWebdataset/mscoco/{00000..00059}.tar"],
        batch_size=1,
        img_in_target_size=upscale_size
    )  
    unet = UpscalerUNet(device=device).to(device)
    from DiffusionModules.DiffusionTrainer import UpscalerDiffusionTrainer
    up_model_pipeline = UpscalerDiffusionTrainer.load_from_checkpoint(path, unet=unet, transformable_data_module=ds, map_location=device).to(device)
    up_model_pipeline._device = device
    up_model_pipeline.diffusion_tools._device = device
    return up_model_pipeline