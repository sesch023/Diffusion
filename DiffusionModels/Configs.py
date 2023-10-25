class ModelLoadConfig:
    upscaler_model_path = "/home/archive/schmidt_models/upscaler.ckpt"
    translator_model_path = "/home/archive/schmidt_models/clip_translator.ckpt"
    vqgan_path = "/home/archive/schmidt_models/vqgan.ckpt"

class DatasetLoadConfig:
    cifar_10_64_path = "/home/archive/cifar10-64"
    
class RunConfig:
    debug = False