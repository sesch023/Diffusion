from lightning.pytorch import LightningDataModule
from DiffusionModules.Util import ImageTransformer

class CIFAR10DataProvider(LightningDataModule):
    transform = ImageTransformer(img_target_size=64)
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, batch_size=16, cifar_path="/home/archive/cifar10-64", num_workers=4, img_target_size=64):
        self.batch_size = batch_size
        self.cifar_path = cifar_path
        
    
    def train_dataloader(self):
        trainset = torchvision.datasets.ImageFolder(f"{cifar_path}/train")
        return torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    def val_dataloader(self):
        valset = torchvision.datasets.ImageFolder(f"{cifar_path}/test")
        return torch.utils.data.DataLoader(valset, batch_size=2, shuffle=True, num_workers=0)

    