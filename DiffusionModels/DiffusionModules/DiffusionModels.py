import torch
import torch.nn as nn
from DiffusionModules.Modules import MultiParamSequential, ResBlock, ResBlockSampleMode, AdaptivePseudo3DConv
from DiffusionModules.Modules import SelfAttentionResBlock, zero_module, AdaptiveSpatioTemporalResBlock, AdaptiveSpatioTemporalSelfAttentionResBlock
import wandb

# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
class ExponentialMovingAverage():
    def __init__(self, beta, step_start_ema=2000):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.step_start_ema = step_start_ema

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        return new if old is None else old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class BasicUNet(nn.Module):
    def __init__(self, i_emb_size=512, t_emb_size=256, mid_emb_size=1024, out_channels=6, device=None):
        super().__init__()
        self._model = UNet(
            i_emb_size=i_emb_size,
            t_emb_size=t_emb_size,
            mid_emb_size=mid_emb_size,
            out_channels=out_channels,
            device=device,
            mha_head_channels=(64, 64, 64),
            use_sample_conv=True,
            use_functional_att_norm_out=False
        )

    def forward(self, x, t_emb=None, i_emb=None):
        return self._model(x, t_emb=t_emb, i_emb=i_emb)


class UpscalerUNet(nn.Module):
    def __init__(self, i_emb_size=512, t_emb_size=256, mid_emb_size=1024, out_channels=6, device=None):
        super().__init__()
        self._model = UNet(
            i_emb_size=i_emb_size,
            t_emb_size=t_emb_size,
            mid_emb_size=mid_emb_size,
            out_channels=out_channels,
            device=device,
            in_channels=6,
            res_blocks_per_resolution=2,
            #mha_heads=(4, 4, 4),
            mha_head_channels=(64, 64, 64),
            base_channels=192, 
            base_channel_mults=(1, 1, 2, 2, 4, 4),
            attention_at_downsample_factor=(8, 16, 32),
            use_sample_conv=False,
            use_functional_att_norm_out=True,
            torch_mha=True
        )

    def forward(self, x, t_emb=None, i_emb=None):
        return self._model(x, t_emb=t_emb, i_emb=i_emb)


class UNet(nn.Module):
    def __init__(
        self,
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
        in_channels=3,
        out_channels=6,
        device=None,
        use_sample_conv=True,
        use_functional_att_norm_out=False,
        torch_mha=False
    ):
        super().__init__()

        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._base_channels = base_channels
        self._base_channel_mults = base_channel_mults
        self._res_blocks_per_resolution = res_blocks_per_resolution
        self._attention_at_downsample_factor = attention_at_downsample_factor
        self._mha_heads = mha_heads
        self._mha_head_channels = mha_head_channels
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._use_res_block_scale_shift_norm = use_res_block_scale_shift_norm
        self._i_emb_size = i_emb_size
        self._t_emb_size = t_emb_size
        self._mid_emb_size = mid_emb_size
        self._dropout = 0.1
        self._use_sample_conv = use_sample_conv

        if t_emb_size is not None:
            self._time_emb_seq = nn.Sequential(
                nn.Linear(t_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )

        if i_emb_size is not None:
            self._img_emb_seq = nn.Sequential(
                nn.Linear(i_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )

        self._in_conv = nn.Conv2d(self._in_channels, self._base_channels, stride=1, padding=1, kernel_size=3)
        self._out_conv = nn.Sequential(
            nn.GroupNorm(32, self._base_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(self._base_channels, self._out_channels, stride=1, padding=1, kernel_size=3))
        )

        self._down_blocks = nn.ModuleList()
        self._up_blocks = nn.ModuleList()

        current_attention_block = 0 
        channels_out = base_channel_mults[0] * base_channels
        current_down_factor = 1
        channel_skip_dim = []
        
        for i, channels in enumerate(base_channel_mults):
            channels_in = channels_out
            channels_out = channels * base_channels
            if current_down_factor in self._attention_at_downsample_factor:
                curr_mha_head_channel = mha_head_channels[current_attention_block] if mha_head_channels is not None else None
                curr_mha_head = mha_heads[current_attention_block] if mha_heads is not None and curr_mha_head_channel is None else None
                current_attention_block += 1
                get_block = lambda curr_ch_in: SelfAttentionResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        mha_head_channels=curr_mha_head_channel,
                        mha_heads=curr_mha_head,
                        use_sample_conv=use_sample_conv,
                        use_functional_norm_out=use_functional_att_norm_out,
                        torch_mha=torch_mha
                )
            else:
                get_block = lambda curr_ch_in: ResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        use_sample_conv=use_sample_conv
                )

            layer_blocks = []
            for _ in range(res_blocks_per_resolution):
                layer_blocks.append(get_block(channels_in))
                channels_in = channels_out
            channel_skip_dim.append(channels_out)
            self._down_blocks.append(MultiParamSequential(*layer_blocks))

            if i < len(base_channel_mults) - 1:
                self._down_blocks.append(ResBlock(
                        in_channels=channels_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        sample_mode=ResBlockSampleMode.DOWNSAMPLE2X,
                        use_sample_conv=use_sample_conv
                ))
                channel_skip_dim.append(channels_out)
                current_down_factor *= 2

        self._mid = MultiParamSequential(
            get_block(channels_in),
            ResBlock(
                in_channels=channels_in,
                out_channels=channels_out,
                emb_size=self._mid_emb_size,
                dropout=self._dropout,
                skip_con=True,
                use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                use_sample_conv=use_sample_conv
            )
        )

        for i, channels in reversed(list(enumerate(base_channel_mults))):
            channels_in = channels_out
            channels_out = channels * base_channels

            if current_down_factor in self._attention_at_downsample_factor:
                current_attention_block -= 1
                curr_mha_head_channel = mha_head_channels[current_attention_block] if mha_head_channels is not None else None
                curr_mha_head = mha_heads[current_attention_block] if mha_heads is not None and curr_mha_head_channel is None else None
                get_block = lambda curr_ch_in: SelfAttentionResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        mha_head_channels=curr_mha_head_channel,
                        mha_heads=curr_mha_head,
                        use_sample_conv=use_sample_conv,
                        use_functional_norm_out=use_functional_att_norm_out,
                        torch_mha=torch_mha
                )
            else:
                get_block = lambda curr_ch_in: ResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        use_sample_conv=use_sample_conv
                )

            layer_blocks = []
            channels_in = channels_in + channel_skip_dim.pop()
            for _ in range(res_blocks_per_resolution):
                layer_blocks.append(get_block(channels_in))
                channels_in = channels_out
            self._up_blocks.append(MultiParamSequential(*layer_blocks))

            if i > 0:
                added_channels = channel_skip_dim.pop()
                self._up_blocks.append(ResBlock(
                        in_channels=channels_in + added_channels,
                        out_channels=channels_out,
                        emb_size=mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        sample_mode=ResBlockSampleMode.UPSAMPLE2X,
                        use_sample_conv=use_sample_conv
                ))
                current_down_factor /= 2

    def forward(self, x, t_emb=None, i_emb=None):
        if self._i_emb_size is None and self._t_emb_size is None:
            emb = None
        else:
            zeros = torch.zeros(self._mid_emb_size).to(self._device)
            i_emb = zeros if i_emb is None else self._img_emb_seq(i_emb)
            t_emb = zeros if t_emb is None else self._time_emb_seq(t_emb)       
            
            emb = i_emb + t_emb

        x = self._in_conv(x)
        outs = []
        i = 0
        for block in self._down_blocks:
            x = block(x, emb)
            outs.append(x)
            i += 1

        x = self._mid(x, emb)

        i = 0
        for block in self._up_blocks:
            xp = outs.pop()
            x = torch.cat((xp, x), dim=1)
            x = block(x, emb)
            i += 1

        return self._out_conv(x)

        
class SpatioTemporalUNet(nn.Module):
    def __init__(
        self,
        base_channels=256,
        base_channel_mults=(1, 2, 3, 4),
        res_blocks_per_resolution=3,
        use_res_block_scale_shift_norm=True,
        attention_at_downsample_factor=(2, 4, 8),
        mha_heads=(4, 4, 4),
        mha_head_channels=None,
        i_emb_size=512,
        t_emb_size=256,
        f_emb_size=256,
        mid_emb_size=1024,
        in_channels=3,
        out_channels=6,
        device=None
    ):
        super().__init__()

        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._base_channels = base_channels
        self._base_channel_mults = base_channel_mults
        self._res_blocks_per_resolution = res_blocks_per_resolution
        self._attention_at_downsample_factor = attention_at_downsample_factor
        self._mha_heads = mha_heads
        self._mha_head_channels = mha_head_channels
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._use_res_block_scale_shift_norm = use_res_block_scale_shift_norm
        self._i_emb_size = i_emb_size
        self._t_emb_size = t_emb_size
        self._f_emb_size = f_emb_size
        self._mid_emb_size = mid_emb_size
        self._dropout = 0.1

        if t_emb_size is not None:
            self._time_emb_seq = nn.Sequential(
                nn.Linear(t_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )

        if i_emb_size is not None:
            self._img_emb_seq = nn.Sequential(
                nn.Linear(i_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )

        if f_emb_size is not None:
            self._fps_emb_seq = nn.Sequential(
                nn.Linear(f_emb_size, mid_emb_size),
                nn.SiLU(),
                nn.Linear(mid_emb_size, mid_emb_size)
            )

        self._in_conv = AdaptivePseudo3DConv(self._in_channels, self._base_channels, stride=1, padding=1, kernel_size=3, temp_kernel_size=3)
        self._pre_out_conv = nn.Sequential(
            nn.GroupNorm(32, self._base_channels),
            nn.SiLU()
        )
        self._out_conv = zero_module(AdaptivePseudo3DConv(self._base_channels, self._out_channels, stride=1, padding=1, kernel_size=3, temp_kernel_size=3))

        self._down_blocks = nn.ModuleList()
        self._up_blocks = nn.ModuleList()

        current_attention_block = 0 
        channels_out = base_channel_mults[0] * base_channels
        current_down_factor = 1
        channel_skip_dim = []
        
        for i, channels in enumerate(base_channel_mults):
            channels_in = channels_out
            channels_out = channels * base_channels
            if current_down_factor in self._attention_at_downsample_factor:
                curr_mha_head_channel = mha_head_channels[current_attention_block] if mha_head_channels is not None else None
                curr_mha_head = mha_heads[current_attention_block] if mha_heads is not None and curr_mha_head_channel is None else None
                current_attention_block += 1
                get_block = lambda curr_ch_in: AdaptiveSpatioTemporalSelfAttentionResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        mha_head_channels=curr_mha_head_channel,
                        mha_heads=curr_mha_head
                )
            else:
                get_block = lambda curr_ch_in: AdaptiveSpatioTemporalResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm
                )

            layer_blocks = []
            for _ in range(res_blocks_per_resolution):
                layer_blocks.append(get_block(channels_in))
                channels_in = channels_out
            channel_skip_dim.append(channels_out)
            self._down_blocks.append(MultiParamSequential(*layer_blocks))

            if i < len(base_channel_mults) - 1:
                self._down_blocks.append(AdaptiveSpatioTemporalResBlock(
                        in_channels=channels_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        sample_mode=ResBlockSampleMode.DOWNSAMPLE2X
                ))
                channel_skip_dim.append(channels_out)
                current_down_factor *= 2

        self._mid = MultiParamSequential(
            get_block(channels_in),
            AdaptiveSpatioTemporalResBlock(
                in_channels=channels_in,
                out_channels=channels_out,
                emb_size=self._mid_emb_size,
                dropout=self._dropout,
                skip_con=True,
                use_scale_shift_norm=self._use_res_block_scale_shift_norm
            )
        )

        for i, channels in reversed(list(enumerate(base_channel_mults))):
            channels_in = channels_out
            channels_out = channels * base_channels

            if current_down_factor in self._attention_at_downsample_factor:
                current_attention_block -= 1
                curr_mha_head_channel = mha_head_channels[current_attention_block] if mha_head_channels is not None else None
                curr_mha_head = mha_heads[current_attention_block] if mha_heads is not None and curr_mha_head_channel is None else None
                get_block = lambda curr_ch_in: AdaptiveSpatioTemporalSelfAttentionResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        mha_head_channels=curr_mha_head_channel,
                        mha_heads=curr_mha_head
                )
            else:
                get_block = lambda curr_ch_in: AdaptiveSpatioTemporalResBlock(
                        in_channels=curr_ch_in,
                        out_channels=channels_out,
                        emb_size=self._mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm
                )

            layer_blocks = []
            channels_in = channels_in + channel_skip_dim.pop()
            for _ in range(res_blocks_per_resolution):
                layer_blocks.append(get_block(channels_in))
                channels_in = channels_out
            self._up_blocks.append(MultiParamSequential(*layer_blocks))

            if i > 0:
                added_channels = channel_skip_dim.pop()
                self._up_blocks.append(AdaptiveSpatioTemporalResBlock(
                        in_channels=channels_in + added_channels,
                        out_channels=channels_out,
                        emb_size=mid_emb_size,
                        dropout=self._dropout,
                        skip_con=True,
                        use_scale_shift_norm=self._use_res_block_scale_shift_norm,
                        sample_mode=ResBlockSampleMode.UPSAMPLE2X
                ))
                current_down_factor /= 2

    def forward(self, x, t_emb=None, i_emb=None, f_emb=None, temporal=True):
        if self._i_emb_size is None and self._t_emb_size is None and self._f_emb_size is None:
            emb = None
        else:
            zeros = torch.zeros(self._mid_emb_size).to(self._device)
            i_emb = zeros if i_emb is None else self._img_emb_seq(i_emb)
            t_emb = zeros if t_emb is None else self._time_emb_seq(t_emb)       
            f_emb = zeros if f_emb is None else self._fps_emb_seq(f_emb)
            
            emb = i_emb + t_emb + f_emb

        x = self._in_conv(x, temporal)
        outs = []
        i = 0
        for block in self._down_blocks:
            x = block(x, emb, temporal)
            outs.append(x)
            i += 1

        x = self._mid(x, emb, temporal)

        for block in self._up_blocks:
            xp = outs.pop()
            x = torch.cat((xp, x), dim=1)
            x = block(x, emb, temporal)

        return self._out_conv(self._pre_out_conv(x), temporal)
