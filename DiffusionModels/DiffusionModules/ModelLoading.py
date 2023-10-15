from DiffusionModules.DiffusionModels import *
from DiffusionModules.DataModules import *

wds_path = ""
translator_model_path = "~/clip_translator.ckpt"
upscaler_model_path = "~/upscaler.ckpt"

def load_udm(path, device, upscale_size=256):
    ds = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=1,
        img_in_target_size=upscale_size
    )  
    unet = UpscalerUNet(device=device).to(device)
    from DiffusionModules.DiffusionTrainer import UpscalerDiffusionTrainer
    up_model_pipeline = UpscalerDiffusionTrainer.load_from_checkpoint(path, unet=unet, transformable_data_module=ds, map_location=device, device=device).to(device)
    up_model_pipeline._device = device
    up_model_pipeline.diffusion_tools._device = device
    up_model_pipeline.eval()
    return up_model_pipeline

def load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size, alt_emb_prov=None):
    unet = BasicUNet(i_emb_size=i_emb_size, device=device).to(device)
    from DiffusionModules.DiffusionTrainer import DiffusionTrainer
    model = DiffusionTrainer.load_from_checkpoint(
        path, 
        unet=unet, 
        embedding_provider=emb_prov,
        alt_validation_emb_provider=None,
        transformable_data_module=dm,
        map_location=device
    ).to(device)
    if alt_emb_prov is not None:
        model.alt_validation_emb_provider=alt_emb_prov
    model.diffusion_tools._device = device
    model.c_devide = device
    model.eval()
    model.up_model_pipeline = load_udm(upscaler_model_path, device, upscale_size)
    model.up_model_pipeline.eval()
    model.up_model_pipeline.to(device)
    return model

def load_wdm(path, device, alt_prov_mode="TRANSLATOR"):
    batch_size = 1
    upscale_size = 256
    dm = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=batch_size,
        num_workers=1
    )  
    from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
    clip_tools = ClipTools(device=device)
    i_emb_size = clip_tools.get_clip_emb_size()
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)
    alt_emb_prov = None
    if alt_prov_mode == "TRANSLATOR":
        alt_emb_prov = ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
    elif alt_prov_mode == "TEXT":
        alt_emb_prov = ClipTextEmbeddingProvider(clip_tools=clip_tools)

    return load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size, alt_emb_prov=alt_emb_prov)

def load_cf10(path, device):
    batch_size = 1
    dm = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=batch_size,
        num_workers=1
    )  
    i_emb_size = len(CIFAR10DataModule.classes)
    upscale_size = 256
    from DiffusionModules.EmbeddingTools import CF10EmbeddingProvider
    emb_prov = CF10EmbeddingProvider()
    return load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size)


