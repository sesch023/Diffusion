class ModelLoadConfig:
    # Path to the Upscaler model
    upscaler_model_path = "/home/archive/schmidt_models/upscaler.ckpt"
    # Path to the CLIP-Translator model
    translator_model_path = "/home/archive/schmidt_models/clip_translator.ckpt"
    # Path to the VQGAN model
    vqgan_path = "/home/archive/schmidt_models/vqgan.ckpt"
    # Path to the Latent Diffusion model
    latent_diffusion_path = "/home/archive/schmidt_models/latent_diff_model.ckpt"
    # Path to the Diffusion model
    diffusion_path = "/home/archive/schmidt_models/diffusion_model.ckpt"
    # Pato the the CIFAR-10 Diffusion model
    cf10_diffusion_path = "/home/archive/schmidt_models/cf10_diffusion_model.ckpt"
    # Path to the Spatio-Temporal Diffusion model
    spatio_temporal_path = "/home/archive/schmidt_models/spatio_temporal_diff_model.ckpt"

class DatasetLoadConfig:
    # Path to the CIFAR-10 dataset
    cifar_10_64_path = "/home/archive/cifar10-64"
    # Path to the cc3m and cc12m datasets, which were often combined for training
    cc_3m_12m_paths = ["/home/archive/CC12M_HIGH_RES/cc12m/{00000..01242}.tar", "/home/archive/CC3M_HIGH_RES/cc3m/{00000..00331}.tar"]
    # Path to the COCO dataset, which was used for validation and testing. A different path was used for each purpose.
    coco_val_path = ["/home/archive/CocoWebdatasetFullScale/mscoco/{00000..00040}.tar"]
    coco_test_path = ["/home/archive/CocoWebdatasetFullScale/mscoco/{00041..00059}.tar"]
    # Path to the Webvid-10M dataset csv file
    webvid_10m_train_csv = "/home/shared-data/webvid/results_10M_train.csv"
    # Path to the Webvid-10M dataset train video folder
    webvid_10m_train_data = "/home/shared-data/webvid/data/videos"
    # Path to the validation csv file of the Webvid-10M dataset
    webvid_10m_val_csv = "/home/shared-data/webvid/results_10M_val.csv"
    # Path to the validation video folder of the Webvid-10M dataset
    webvic_10m_val_data = "/home/shared-data/webvid/data_val/videos"
    
class RunConfig:
    # Enable debugging mode for certain wandb outputs
    debug = False