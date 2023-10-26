from DiffusionModules.DiffusionModels import BasicUNet, UNet, UpscalerUNet, SpatioTemporalUNet
from DiffusionModules.DataModules import WebdatasetDataModule, CIFAR10DataModule, VideoDatasetDataModule
from torchinfo import summary
from Configs import ModelLoadConfig, DatasetLoadConfig

# We dont need data for loading the model, but we need to initialize the DataModule for transforming the data
wds_path = ""

def load_udm(path, device, upscale_size=256):
    """
    Loads a trained UpscalerDiffusionTrainer model from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis, 
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param device: Device to load the model on.
    :param upscale_size: Size of the output image, default is 256.
    :return: UpscalerDiffusionTrainer.
    """    
    ds = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=1,
        img_in_target_size=upscale_size
    )  
    unet = UpscalerUNet(device=device).to(device)
    from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
    from DiffusionModules.DiffusionTrainer import UpscalerDiffusionTrainer
    clip_tools = ClipTools(device=device)
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)
    up_model_pipeline = UpscalerDiffusionTrainer.load_from_checkpoint(
        path, 
        unet=unet, 
        transformable_data_module=ds, 
        map_location=device, 
        embedding_provider=emb_prov,
        device=device).to(device)
    up_model_pipeline.embedding_provider = emb_prov
    up_model_pipeline.embedding_provider.to(device)
    up_model_pipeline._device = device
    up_model_pipeline.diffusion_tools._device = device
    up_model_pipeline.eval()
    return up_model_pipeline

def load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size, alt_emb_prov=None):
    """
    Loads a trained DiffusionTrainer model from a checkpoint. 
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param dm: DataModule to load the model on.
    :param device: Device to load the model on.
    :param emb_prov: EmbeddingProvider to load the model on.
    :param i_emb_size: Size of the input embedding.
    :param upscale_size: Size of the output image.
    :param alt_emb_prov: Alternative EmbeddingProvider to load the model on.
    :return: DiffusionTrainer.
    """    
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
    upscaler_model_path = ModelLoadConfig.upscaler_model_path
    model.up_model_pipeline = load_udm(upscaler_model_path, device, upscale_size)
    model.up_model_pipeline.eval()
    model.up_model_pipeline.to(device)
    return model

def load_wdm(path, device, alt_prov_mode="TRANSLATOR"):
    """
    Loads a trained DiffusionTrainer based on the WebdatasetDataModule from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param device: Device to load the model on.
    :param alt_prov_mode: Alternative EmbeddingProvider mode, either "TRANSLATOR" or "TEXT" for the 
                          ClipTranslatorEmbeddingProvider or ClipTextEmbeddingProvider. Default is "TRANSLATOR".
    :return: _description_
    """    
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
        translator_model_path = ModelLoadConfig.translator_model_path
        alt_emb_prov = ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
    elif alt_prov_mode == "TEXT":
        alt_emb_prov = ClipTextEmbeddingProvider(clip_tools=clip_tools)

    return load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size, alt_emb_prov=alt_emb_prov)

def load_cf10(path, device):
    """
    Loads a trained DiffusionTrainer based on the CIFAR10DataModule from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param device: Device to load the model on.
    :return: DiffusionTrainer.
    """    
    batch_size = 1
    dm = CIFAR10DataModule(
        cifar_path=DatasetLoadConfig.cifar_10_64_path,
        batch_size=batch_size,
        num_workers=1
    )
    i_emb_size = len(CIFAR10DataModule.classes)
    upscale_size = 256
    from DiffusionModules.EmbeddingTools import CF10EmbeddingProvider
    emb_prov = CF10EmbeddingProvider()
    return load_dm(path, dm, device, emb_prov, i_emb_size, upscale_size)


def load_vqgan(path, device, img_size=256, z_channels=3, shared_args=None, encoder_args=None, decoder_args=None, discriminator_args=None, loss_args=None):
    """
    Loads a trained VQGAN model from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param device: Device to load the model on.
    :param img_size: Size of the input image, default is 256.
    :param z_channels: Number of channels of the latent space, default is 3.
    :param shared_args: Shared arguments for the Encoder and Decoder, default is:
                        z_channels=z_channels,
                        ch=128,
                        resolution=256,
                        num_res_blocks=2,
                        attn_resolutions=[],
                        ch_mult=(1,2,4),
                        dropout=0.0,
                        emb_size=512,
                        out_emb_size=1024
    :param encoder_args: Additional arguments for the Encoder, default is:
                         in_channels=3,
                         double_z=False
    :param decoder_args: Additional arguments for the Decoder, default is:
                         out_channels=3
    :param discriminator_args: Arguments for the Discriminator, default is:
                               input_nc=decoder.out_channels,
                               n_layers=3,
                               ndf=64
    :param loss_args: Arguments for the Loss, default is:
                      disc_start=0,
                      disc_weight=0.75,
                      codebook_weight=1.0,
                      disc_conditional=False
    :return: VQModel.
    """    
    from DiffusionModules.LatentVQGANModules import Encoder, Decoder, NLayerDiscriminator
    from DiffusionModules.LatentVQGANModel import VQModel
    from DiffusionModules.VQGANLosses import VQLPIPSWithDiscriminator
    from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider

    ds = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=1,
        img_in_target_size=img_size
    ) 

    # Shared args between Encoder and Decoder
    shared_args = dict(
        z_channels=z_channels,
        ch=128,
        resolution=256,
        num_res_blocks=2,
        attn_resolutions=[],
        ch_mult=(1,2,4),
        dropout=0.0,
        emb_size=512,
        out_emb_size=1024
    ) if shared_args is None else shared_args

    # Build Encoder Args
    encoder_args = dict(
        in_channels=3,
        double_z=False
    ) if encoder_args is None else encoder_args

    # Build Encoder
    encoder = Encoder(
        **encoder_args,
        **shared_args
    ).to(device)

    print("Encoder")
    summary(encoder, [(1, encoder.in_channels, shared_args["resolution"], shared_args["resolution"]), (1, 512)], verbose=1)

    # Build Decoder
    decoder = Decoder(
        out_channels=3,
        **shared_args
    ).to(device) if decoder_args is None else Decoder(**decoder_args, **shared_args).to(device)

    decoder_in_res = shared_args["resolution"] // (2 ** (len(shared_args["ch_mult"])-1))
    print("Decoder")
    summary(decoder, [(1, z_channels, decoder_in_res, decoder_in_res), (1, 512)], verbose=1)

    # Build Discriminator Args
    discriminator_args = dict(
        input_nc=decoder.out_channels,
        n_layers=3,
        ndf=64
    ) if discriminator_args is None else discriminator_args

    # Build Discriminator
    discriminator = NLayerDiscriminator(
        **discriminator_args
    ).to(device)

    print("Discriminator")
    summary(discriminator, (1, decoder.out_channels, shared_args["resolution"], shared_args["resolution"]), verbose=1)

    # Build Loss Args
    loss_args = dict(
        discriminator=discriminator,
        disc_start=0,
        disc_weight=0.75,
        codebook_weight=1.0,
        disc_conditional=False
    ) if loss_args is None else loss_args

    # Build Loss
    loss = VQLPIPSWithDiscriminator(
        **loss_args
    ).to(device)

    clip_tools = ClipTools(device=device)
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)

    reconstructions_out_base_path = "emb_reconstructions/"
    return VQModel.load_from_checkpoint(
        path, 
        device=device, 
        encoder=encoder, 
        decoder=decoder, 
        loss=loss, 
        transformable_data_module=ds,
        embedding_provider=emb_prov,
        strict=False,
        map_location=device
    ).to(device)


def load_latent_diffusion(path, vqgan_path, device, img_size=256, alt_prov_mode="TRANSLATOR", unet_in_channels=3, vqgan_args=None):
    """
    Loads a trained LatentDiffusionTrainer model from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param vqgan_path: Path to the VQGAN checkpoint.
    :param device: Device to load the model on.
    :param img_size: Size of the input image, default is 256.
    :param alt_prov_mode: Alternative EmbeddingProvider mode, either "TRANSLATOR" or "TEXT" for the 
                          ClipTranslatorEmbeddingProvider or ClipTextEmbeddingProvider. Default is "TRANSLATOR".
    :param unet_in_channels: Number of input channels for the UNet, default is 3.
    :param vqgan_args: Arguments for the VQGAN, default is None.
    :return: LatentDiffusionTrainer.
    """    
    ds = WebdatasetDataModule(
        [wds_path],
        [wds_path],
        batch_size=1,
        img_in_target_size=img_size
    )  
    unet = UNet(
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
    )
    from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
    from DiffusionModules.LatentDiffusionTrainer import LatentDiffusionTrainer
    clip_tools = ClipTools(device=device)
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)
    alt_emb_prov = None
    if alt_prov_mode == "TRANSLATOR":
        alt_emb_prov = ClipTranslatorEmbeddingProvider(clip_tools=clip_tools, translator_model_path=translator_model_path)
    elif alt_prov_mode == "TEXT":
        alt_emb_prov = ClipTextEmbeddingProvider(clip_tools=clip_tools)


    vqgan_args = dict() if vqgan_args is None else vqgan_args
    vqgan = load_vqgan(vqgan_path, device=device, **vqgan_args)

    latent_diffusion_trainer = LatentDiffusionTrainer.load_from_checkpoint(
        path, 
        vqgan=vqgan,
        unet=unet, 
        transformable_data_module=ds, 
        map_location=device, 
        embedding_provider=emb_prov,
        device=device,
        alt_validation_emb_provider=alt_emb_prov
    ).to(device)
    latent_diffusion_trainer.embedding_provider = emb_prov
    latent_diffusion_trainer.embedding_provider.to(device)
    latent_diffusion_trainer.vqgan.to(device)
    latent_diffusion_trainer._device = device
    latent_diffusion_trainer.diffusion_tools._device = device
    latent_diffusion_trainer.eval()
    return latent_diffusion_trainer


def load_spatio_temporal(path, device, after_load_fvd=False):
    """
    Loads a trained SpatioTemporalDiffusionTrainer model from a checkpoint.
    This was specifically written for the Model trained in my Master's Thesis,
    therefore most of the parameters are hardcoded.

    :param path: Path to the checkpoint.
    :param device: Device to load the model on.
    :param after_load_fvd: Whether to load the FVD after loading the model, default is False.
    :return: SpatioTemporalDiffusionTrainer.
    """    
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
        f_emb_size=256,
        mid_emb_size=1024,
        in_channels=unet_in_channels,
        out_channels=unet_in_channels*2,
        device=device
    ).to(device)

    temporal_dataset = VideoDatasetDataModule(
        None,
        None,
        None,
        None,
        batch_size=1,
        num_workers=1,
        nth_frames=1,
        max_frames_per_part=16,
        min_frames_per_part=4,
        first_part_only=True
    )

    from DiffusionModules.EmbeddingTools import ClipTools, ClipEmbeddingProvider, ClipTranslatorEmbeddingProvider, ClipTextEmbeddingProvider
    from DiffusionModules.DiffusionTrainer import SpatioTemporalDiffusionTrainer

    clip_tools = ClipTools(device=device)
    emb_prov = ClipEmbeddingProvider(clip_tools=clip_tools)
    text_emb_prov = ClipTextEmbeddingProvider(clip_tools=clip_tools)

    spatio_temporal_trainer = SpatioTemporalDiffusionTrainer.load_from_checkpoint(
        path,
        transformable_data_module=temporal_dataset,
        unet=unet,
        embedding_provider=emb_prov,
        temporal_embedding_provider=text_emb_prov,
        device=device,
        map_location=device,
        temporal=True,
        after_load_fvd=after_load_fvd
    ).to(device)
    spatio_temporal_trainer.diffusion_tools._device = device
    spatio_temporal_trainer.eval()
    spatio_temporal_trainer.load_fvd()
    return spatio_temporal_trainer

