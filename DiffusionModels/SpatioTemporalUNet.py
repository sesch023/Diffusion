from DiffusionModules.DiffusionModels import SpatioTemporalUNet
import os
import torch
from torchinfo import summary

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WDS_VERBOSE_CACHE"] = "1"

gpus=[1]
device = f"cuda:{str(gpus[0])}" if torch.cuda.is_available() else "cpu"

unet_in_channels = 3
unet_in_size = 64
unet = SpatioTemporalUNet(
    base_channels=256,
    base_channel_mults=(1, 2, 3, 4),
    res_blocks_per_resolution=3,
    use_res_block_scale_shift_norm=True,
    attention_at_downsample_factor=(2, 4, 8),
    mha_heads=(4, 4, 4),
    mha_head_channels=None,
    i_emb_size=512,
    t_emb_size=256,
    mid_emb_size=1024,
    in_channels=unet_in_channels,
    out_channels=unet_in_channels*2,
    device=device
).to(device)

summary(unet, [(1, unet_in_channels, 32, 64, 64), (1, 256), (1, 512), (1, 256)], verbose=1)